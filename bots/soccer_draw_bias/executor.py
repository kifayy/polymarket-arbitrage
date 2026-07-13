"""
Execution and fill verification for buying favorite No shares.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from polymarket_client.api import PolymarketClient
from polymarket_client.models import OrderSide, TokenType, Trade
from core.portfolio import Portfolio
from core.risk_manager import RiskManager

from bots.soccer_draw_bias.engine import DrawBiasConfig, MatchStateMachine
from bots.soccer_draw_bias.models import MatchRecord

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    success: bool
    order_id: Optional[str] = None
    price: Optional[float] = None
    shares: Optional[float] = None
    reason: str = ""
    trade: Optional[Trade] = None


class DrawBiasExecutor:
    """Places limit buys on favorite No when ask <= target max price."""

    STRATEGY_TAG = "soccer_draw_bias"

    def __init__(
        self,
        client: PolymarketClient,
        risk_manager: RiskManager,
        portfolio: Portfolio,
        config: DrawBiasConfig,
        state_machine: MatchStateMachine,
        immediate_fill_dry_run: bool = True,
    ):
        self.client = client
        self.risk_manager = risk_manager
        self.portfolio = portfolio
        self.config = config
        self.state_machine = state_machine
        self.immediate_fill_dry_run = immediate_fill_dry_run

    async def maybe_execute(self, match: MatchRecord) -> ExecutionResult:
        if not self.state_machine.should_evaluate_orderbook(match):
            return ExecutionResult(success=False, reason="gates_not_met")

        if self.risk_manager.state.kill_switch_triggered:
            return ExecutionResult(success=False, reason="kill_switch")

        if not match.token_id_favorite_no:
            return ExecutionResult(success=False, reason="missing_token")

        book = await self.client._fetch_token_orderbook(
            match.token_id_favorite_no, TokenType.NO
        )
        best_ask = book.best_ask
        best_ask_size = book.best_ask_size or 0.0

        if best_ask is None:
            return ExecutionResult(success=False, reason="empty_asks")

        if best_ask > self.config.target_max_buy_price:
            logger.debug(
                f"Match {match.match_id}: best ask {best_ask:.3f} > "
                f"{self.config.target_max_buy_price:.3f} — wait"
            )
            return ExecutionResult(success=False, reason="price_above_threshold")

        # Size: risk_unit / price (e.g. $15 / 0.15 = 100 shares)
        price = float(best_ask)
        shares = self.config.risk_unit_usd / price
        if best_ask_size > 0:
            shares = min(shares, best_ask_size)
        shares = round(shares, 2)
        if shares <= 0:
            return ExecutionResult(success=False, reason="insufficient_liquidity")

        token_type = await self._resolve_token_type(match)

        # Risk gate via a synthetic order check
        from polymarket_client.models import Order, OrderStatus

        probe = Order(
            order_id="probe",
            market_id=match.polymarket_market_id,
            token_type=token_type,
            side=OrderSide.BUY,
            price=price,
            size=shares,
            status=OrderStatus.PENDING,
            strategy_tag=self.STRATEGY_TAG,
        )
        if not self.risk_manager.check_order(probe):
            return ExecutionResult(success=False, reason="risk_rejected")

        try:
            order = await self.client.place_order(
                market_id=match.polymarket_market_id,
                token_type=token_type,
                side=OrderSide.BUY,
                price=price,
                size=shares,
                strategy_tag=self.STRATEGY_TAG,
            )
        except Exception as e:
            logger.error(f"Order place failed match={match.match_id}: {e}")
            return ExecutionResult(success=False, reason=f"place_failed:{e}")

        trade: Optional[Trade] = None
        if self.client.dry_run and self.immediate_fill_dry_run:
            trade = self.client.simulate_fill(order.order_id, fill_size=shares)
            if trade:
                self.portfolio.update_from_fill(trade)
                self.risk_manager.update_from_fill(trade)
        else:
            # Poll briefly for fill in live mode
            filled = await self._wait_for_fill(order.order_id, timeout_sec=15)
            if not filled:
                try:
                    await self.client.cancel_order(order.order_id)
                except Exception:
                    pass
                return ExecutionResult(
                    success=False,
                    order_id=order.order_id,
                    reason="not_filled",
                )

        self.state_machine.mark_executed(
            match, price=price, shares=shares, order_id=order.order_id
        )
        logger.info(
            f"EXECUTED match={match.match_id} buy No {shares}@${price:.3f} "
            f"favorite={match.favorite_team} minute={match.current_minute}"
        )
        return ExecutionResult(
            success=True,
            order_id=order.order_id,
            price=price,
            shares=shares,
            reason="filled",
            trade=trade,
        )

    async def _resolve_token_type(self, match: MatchRecord) -> TokenType:
        """Map favorite-No token id onto client's YES/NO enum for this market."""
        try:
            market = await self.client.get_market(match.polymarket_market_id)
            if match.token_id_favorite_no == market.no_token_id:
                return TokenType.NO
            if match.token_id_favorite_no == market.yes_token_id:
                return TokenType.YES
        except Exception as e:
            logger.warning(f"Token resolve fallback to NO: {e}")
        return TokenType.NO

    async def _wait_for_fill(self, order_id: str, timeout_sec: float = 15) -> bool:
        import asyncio

        elapsed = 0.0
        while elapsed < timeout_sec:
            orders = await self.client.get_open_orders()
            open_ids = {o.order_id for o in orders if o.is_open}
            if order_id not in open_ids:
                # Assume filled/cancelled; check trades
                trades = await self.client.get_trades(limit=50)
                if any(t.order_id == order_id for t in trades):
                    return True
                # In live, absence from open without trade may mean cancel
                return False
            await asyncio.sleep(1.0)
            elapsed += 1.0
        return False
