"""
Dashboard Integration
======================

Integrates the dashboard with the trading bot components.
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional

from dashboard.server import dashboard_state

logger = logging.getLogger(__name__)


class DashboardIntegration:
    """
    Integrates the trading bot with the dashboard.
    
    Updates the dashboard state with live data from the bot.
    """
    
    def __init__(
        self,
        data_feed=None,
        arb_engine=None,
        execution_engine=None,
        risk_manager=None,
        portfolio=None,
        mode: str = "dry_run",
        mm_enabled: bool = False,
        cross_platform_enabled: bool = True,
        soccer_enabled: bool = False,
    ):
        self.data_feed = data_feed
        self.arb_engine = arb_engine
        self.execution_engine = execution_engine
        self.risk_manager = risk_manager
        self.portfolio = portfolio
        self.soccer_bot = None
        
        dashboard_state.mode = mode
        dashboard_state.is_running = False
        dashboard_state.set_bot_status("bundle_arb", "running", enabled=True)
        dashboard_state.set_bot_status(
            "market_making",
            "running" if mm_enabled else "paused",
            enabled=mm_enabled,
        )
        dashboard_state.set_bot_status(
            "cross_platform",
            "running" if cross_platform_enabled else "paused",
            enabled=cross_platform_enabled,
        )
        dashboard_state.set_bot_status(
            "soccer_draw_bias",
            "running" if soccer_enabled else "paused",
            enabled=soccer_enabled,
        )
        
        self._update_task: Optional[asyncio.Task] = None
        self._running = False

    def attach_soccer_bot(self, soccer_bot) -> None:
        """Wire the soccer draw-bias bot for live dashboard cards."""
        self.soccer_bot = soccer_bot
        dashboard_state.set_bot_status("soccer_draw_bias", "running", enabled=True)
    
    async def start(self, update_interval: float = 1.0) -> None:
        """Start the dashboard integration."""
        self._running = True
        dashboard_state.is_running = True
        
        self._update_task = asyncio.create_task(
            self._update_loop(update_interval)
        )
        
        logger.info("Dashboard integration started")
    
    async def stop(self) -> None:
        """Stop the dashboard integration."""
        self._running = False
        dashboard_state.is_running = False
        
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Dashboard integration stopped")
    
    async def _update_loop(self, interval: float) -> None:
        """Periodically update the dashboard state."""
        while self._running:
            try:
                await self._update_state()
                await self._broadcast_update()
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Dashboard update error: {e}")
                await asyncio.sleep(interval)
    
    async def _update_state(self) -> None:
        """Update the dashboard state from bot components."""
        # Update markets
        if self.data_feed:
            markets = {}
            for market_id, state in self.data_feed.get_all_market_states().items():
                ob = state.order_book
                markets[market_id] = {
                    "market_id": market_id,
                    "question": state.market.question[:80] if state.market.question else market_id,
                    "best_bid_yes": ob.best_bid_yes,
                    "best_ask_yes": ob.best_ask_yes,
                    "best_bid_no": ob.best_bid_no,
                    "best_ask_no": ob.best_ask_no,
                    "total_ask": ob.total_ask,
                    "total_bid": ob.total_bid,
                    "spread_yes": ob.yes.spread if ob.yes else None,
                    "spread_no": ob.no.spread if ob.no else None,
                }
            dashboard_state.markets = markets
        
        # Update portfolio
        if self.portfolio:
            summary = self.portfolio.get_summary()
            dashboard_state.portfolio = summary
        
        # Update risk
        if self.risk_manager:
            dashboard_state.risk = self.risk_manager.get_summary()
        
        # Update orders
        if self.execution_engine:
            orders = self.execution_engine.get_open_orders()
            dashboard_state.orders = [
                {
                    "order_id": o.order_id,
                    "market_id": o.market_id,
                    "side": o.side.value,
                    "token_type": o.token_type.value,
                    "price": o.price,
                    "size": o.size,
                    "filled_size": o.filled_size,
                    "status": o.status.value,
                }
                for o in orders
            ]
            
            # Update stats
            stats = self.execution_engine.get_stats()
            dashboard_state.stats = {
                "orders_placed": stats.orders_placed,
                "orders_filled": stats.orders_filled,
                "orders_cancelled": stats.orders_cancelled,
                "signals_processed": stats.signals_processed,
            }
        
        # Update arb stats and timing
        if self.arb_engine:
            arb_stats = self.arb_engine.get_stats()
            dashboard_state.stats.update({
                "bundle_opportunities": arb_stats.bundle_opportunities_detected,
                "mm_opportunities": arb_stats.mm_opportunities_detected,
                "signals_generated": arb_stats.signals_generated,
            })
            
            # Update opportunity timing stats
            dashboard_state.timing = self.arb_engine.get_timing_stats()
        
        # Update operational stats
        if self.data_feed:
            markets_with_data = len([m for m in dashboard_state.markets.values() 
                                     if m.get("best_bid_yes") or m.get("best_ask_yes")])
            dashboard_state.operational = {
                "total_markets": len(self.data_feed.market_ids),
                "markets_with_orderbooks": len(dashboard_state.markets),
                "markets_with_prices": markets_with_data,
                "orderbook_updates": self.data_feed.update_count,
                "is_streaming": self.data_feed.is_running,
            }

        # Soccer draw-bias match snapshot
        if self.soccer_bot is not None:
            self._sync_soccer_state()
        
        dashboard_state.last_update = datetime.utcnow()

    def _sync_soccer_state(self) -> None:
        """Push soccer match universe into dashboard_state.soccer."""
        from bots.soccer_draw_bias.models import MatchStatus

        matches = list(getattr(self.soccer_bot, "matches", {}).values())
        counts = {
            "scheduled": 0,
            "monitoring": 0,
            "active_window": 0,
            "executed": 0,
            "settled": 0,
            "discarded": 0,
        }
        rows = []
        for m in matches:
            key = m.status.value.lower()
            if key in counts:
                counts[key] += 1
            rows.append({
                "match_id": m.match_id,
                "home_team": m.home_team,
                "away_team": m.away_team,
                "favorite": m.favorite_team,
                "status": m.status.value,
                "minute": m.current_minute,
                "score": list(m.current_score),
                "market_id": m.polymarket_market_id,
                "pnl": m.pnl,
                "outcome": m.outcome,
            })
            # Surface as market rows so Markets tab isn't empty in soccer-only mode
            mid = m.polymarket_market_id or str(m.match_id)
            if mid not in dashboard_state.markets:
                dashboard_state.markets[mid] = {
                    "market_id": mid,
                    "question": (
                        f"[Soccer {m.status.value}] {m.home_team} vs {m.away_team} "
                        f"(fav {m.favorite_team}) {m.current_score[0]}-{m.current_score[1]} "
                        f"{m.current_minute}'"
                    ),
                    "best_bid_yes": None,
                    "best_ask_yes": None,
                    "best_bid_no": None,
                    "best_ask_no": None,
                    "total_ask": None,
                    "total_bid": None,
                    "spread_yes": None,
                    "spread_no": None,
                }
        dashboard_state.soccer = {**counts, "matches": rows}
        if matches and dashboard_state.bots.get("soccer_draw_bias", {}).get("enabled"):
            # Keep card alive while universe is tracked
            if any(m.status not in (MatchStatus.SETTLED, MatchStatus.DISCARDED) for m in matches):
                dashboard_state.set_bot_status("soccer_draw_bias", "running", enabled=True)
    
    async def _broadcast_update(self) -> None:
        """Broadcast update to connected clients."""
        await dashboard_state.broadcast({
            "type": "update",
            "data": dashboard_state.to_dict()
        })
    
    def add_opportunity(
        self,
        opportunity_type: str,
        market_id: str,
        edge: float,
        **kwargs
    ) -> None:
        """Add an opportunity to the dashboard."""
        strategy = kwargs.pop("strategy", None) or opportunity_type
        opp = {
            "type": opportunity_type,
            "strategy": strategy,
            "market_id": market_id,
            "edge": edge,
            **kwargs
        }
        dashboard_state.add_opportunity(opp)
        
        # Broadcast immediately
        asyncio.create_task(dashboard_state.broadcast({
            "type": "opportunity",
            "data": opp
        }))
    
    def add_signal(
        self,
        action: str,
        market_id: str,
        **kwargs
    ) -> None:
        """Add a signal to the dashboard."""
        signal = {
            "action": action,
            "market_id": market_id,
            **kwargs
        }
        dashboard_state.add_signal(signal)
        
        asyncio.create_task(dashboard_state.broadcast({
            "type": "activity",
            "data": signal
        }))
    
    def add_trade(
        self,
        side: str,
        price: float,
        size: float,
        **kwargs
    ) -> None:
        """Add a trade to the dashboard (and strategy P/L ledger)."""
        trade = {
            "side": side,
            "price": price,
            "size": size,
            **kwargs
        }
        dashboard_state.add_trade(trade)
        
        asyncio.create_task(dashboard_state.broadcast({
            "type": "activity",
            "data": trade
        }))

