"""
Cross-Platform Execution
========================

Places the buy and sell legs of a Polymarket ↔ Kalshi arbitrage opportunity.
Respects dry_run mode and risk limits.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

from polymarket_client.api import PolymarketClient
from polymarket_client.models import OrderSide, TokenType
from kalshi_client.api import KalshiClient
from core.cross_platform_arb import CrossPlatformOpportunity
from core.risk_manager import RiskManager
from core.portfolio import Portfolio


logger = logging.getLogger(__name__)


@dataclass
class CrossPlatformExecResult:
    """Result of attempting a cross-platform trade."""
    success: bool
    opportunity_id: str
    buy_order_id: Optional[str] = None
    sell_order_id: Optional[str] = None
    size: float = 0.0
    error: Optional[str] = None
    dry_run: bool = True


class CrossPlatformExecutor:
    """
    Executes cross-platform arbitrage by placing legs on both venues.

    Buy leg first, then sell leg. If sell fails after buy succeeds, logs a
    warning (one-legged exposure) — full auto-unwind can be added later.
    """

    def __init__(
        self,
        polymarket: PolymarketClient,
        kalshi: KalshiClient,
        risk_manager: RiskManager,
        portfolio: Portfolio,
        default_order_size: float = 5.0,
        max_order_size: float = 10.0,
        dry_run: bool = True,
        enabled: bool = True,
    ):
        self.polymarket = polymarket
        self.kalshi = kalshi
        self.risk_manager = risk_manager
        self.portfolio = portfolio
        self.default_order_size = default_order_size
        self.max_order_size = max_order_size
        self.dry_run = dry_run
        self.enabled = enabled
        self._executed_ids: set[str] = set()
        self.stats = {
            "attempts": 0,
            "successes": 0,
            "failures": 0,
            "skipped_duplicates": 0,
        }

    def _size_for(self, opp: CrossPlatformOpportunity) -> float:
        size = opp.suggested_size or self.default_order_size
        size = min(size, self.default_order_size, self.max_order_size)
        if opp.max_size > 0:
            size = min(size, opp.max_size)
        return max(size, 0.0)

    async def execute(self, opp: CrossPlatformOpportunity) -> CrossPlatformExecResult:
        """Execute a cross-platform opportunity (or simulate in dry_run)."""
        self.stats["attempts"] += 1

        if not self.enabled:
            return CrossPlatformExecResult(
                success=False,
                opportunity_id=opp.opportunity_id,
                error="cross-platform execution disabled",
                dry_run=self.dry_run,
            )

        if opp.opportunity_id in self._executed_ids:
            self.stats["skipped_duplicates"] += 1
            return CrossPlatformExecResult(
                success=False,
                opportunity_id=opp.opportunity_id,
                error="already executed this opportunity id",
                dry_run=self.dry_run,
            )

        size = self._size_for(opp)
        if size <= 0:
            self.stats["failures"] += 1
            return CrossPlatformExecResult(
                success=False,
                opportunity_id=opp.opportunity_id,
                error="size too small",
                dry_run=self.dry_run,
            )

        if not self.risk_manager.within_global_limits():
            self.stats["failures"] += 1
            return CrossPlatformExecResult(
                success=False,
                opportunity_id=opp.opportunity_id,
                error="risk limits breached",
                dry_run=self.dry_run,
            )

        token = TokenType.YES if opp.token.upper() == "YES" else TokenType.NO
        pair = opp.market_pair

        logger.info(
            f"{'[DRY RUN] ' if self.dry_run else ''}"
            f"Executing cross-platform: buy {opp.token} on {opp.buy_platform} "
            f"@ {opp.buy_price:.3f}, sell on {opp.sell_platform} @ {opp.sell_price:.3f} "
            f"size={size:.2f}"
        )

        try:
            buy_order_id, sell_order_id = await self._place_legs(
                opp=opp,
                token=token,
                size=size,
                poly_market_id=pair.polymarket_id,
                kalshi_ticker=pair.kalshi_ticker,
            )
        except Exception as e:
            self.stats["failures"] += 1
            logger.exception(f"Cross-platform execution failed: {e}")
            return CrossPlatformExecResult(
                success=False,
                opportunity_id=opp.opportunity_id,
                size=size,
                error=str(e),
                dry_run=self.dry_run,
            )

        self._executed_ids.add(opp.opportunity_id)
        self.stats["successes"] += 1
        return CrossPlatformExecResult(
            success=True,
            opportunity_id=opp.opportunity_id,
            buy_order_id=buy_order_id,
            sell_order_id=sell_order_id,
            size=size,
            dry_run=self.dry_run,
        )

    async def _place_legs(
        self,
        opp: CrossPlatformOpportunity,
        token: TokenType,
        size: float,
        poly_market_id: str,
        kalshi_ticker: str,
    ) -> tuple[Optional[str], Optional[str]]:
        """Place buy then sell legs. Returns (buy_order_id, sell_order_id)."""
        buy_id: Optional[str] = None
        sell_id: Optional[str] = None

        # --- BUY LEG ---
        if opp.buy_platform == "polymarket":
            buy_order = await self.polymarket.place_order(
                market_id=poly_market_id,
                token_type=token,
                side=OrderSide.BUY,
                price=opp.buy_price,
                size=size,
                strategy_tag="cross_platform_buy",
            )
            buy_id = buy_order.order_id
        else:
            buy_order = await self.kalshi.place_order(
                ticker=kalshi_ticker,
                token_type=token,
                side=OrderSide.BUY,
                price=opp.buy_price,
                size=size,
                strategy_tag="cross_platform_buy",
            )
            buy_id = buy_order.order_id

        # --- SELL LEG ---
        try:
            if opp.sell_platform == "polymarket":
                sell_order = await self.polymarket.place_order(
                    market_id=poly_market_id,
                    token_type=token,
                    side=OrderSide.SELL,
                    price=opp.sell_price,
                    size=size,
                    strategy_tag="cross_platform_sell",
                )
                sell_id = sell_order.order_id
            else:
                sell_order = await self.kalshi.place_order(
                    ticker=kalshi_ticker,
                    token_type=token,
                    side=OrderSide.SELL,
                    price=opp.sell_price,
                    size=size,
                    strategy_tag="cross_platform_sell",
                )
                sell_id = sell_order.order_id
        except Exception as e:
            logger.error(
                f"SELL LEG FAILED after buy {buy_id} succeeded — "
                f"one-legged exposure risk: {e}"
            )
            raise

        return buy_id, sell_id
