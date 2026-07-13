"""
Dashboard Server
=================

FastAPI-based web server for the trading dashboard.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger(__name__)


STRATEGY_ALIASES = {
    "bundle_long": "bundle_arb",
    "bundle_short": "bundle_arb",
    "bundle_arb": "bundle_arb",
    "mm_bid": "market_making",
    "mm_ask": "market_making",
    "mm": "market_making",
    "market_making": "market_making",
    "cross_platform": "cross_platform",
    "cross_platform_buy": "cross_platform",
    "cross_platform_sell": "cross_platform",
    "cross": "cross_platform",
    "soccer_draw_bias": "soccer_draw_bias",
    "soccer": "soccer_draw_bias",
    "draw_bias": "soccer_draw_bias",
}

KNOWN_STRATEGIES = (
    "bundle_arb",
    "market_making",
    "cross_platform",
    "soccer_draw_bias",
)


def normalize_strategy(raw: Optional[str]) -> str:
    if not raw:
        return "bundle_arb"
    key = str(raw).strip().lower()
    return STRATEGY_ALIASES.get(
        key, key if key in KNOWN_STRATEGIES else "bundle_arb"
    )


def _default_bots() -> dict:
    return {
        "bundle_arb": {
            "id": "bundle_arb",
            "name": "Bundle Arbitrage",
            "tagline": "YES + NO mispricing within a single venue",
            "venue": "Polymarket",
            "enabled": True,
            "status": "idle",
            "realized_pnl": 0.0,
            "trades": 0,
            "volume": 0.0,
            "wins": 0,
            "losses": 0,
            "opportunities": 0,
            "open_orders": 0,
        },
        "cross_platform": {
            "id": "cross_platform",
            "name": "Cross-Platform",
            "tagline": "Price gaps between Polymarket and Kalshi",
            "venue": "Polymarket · Kalshi",
            "enabled": True,
            "status": "idle",
            "realized_pnl": 0.0,
            "trades": 0,
            "volume": 0.0,
            "wins": 0,
            "losses": 0,
            "opportunities": 0,
            "open_orders": 0,
        },
        "market_making": {
            "id": "market_making",
            "name": "Market Making",
            "tagline": "Capture wide spreads with two-sided quotes",
            "venue": "Polymarket",
            "enabled": False,
            "status": "paused",
            "realized_pnl": 0.0,
            "trades": 0,
            "volume": 0.0,
            "wins": 0,
            "losses": 0,
            "opportunities": 0,
            "open_orders": 0,
        },
        "soccer_draw_bias": {
            "id": "soccer_draw_bias",
            "name": "Soccer Draw Bias",
            "tagline": "Buy favorite No when tied late (75–88')",
            "venue": "Polymarket · Soccer",
            "enabled": True,
            "status": "idle",
            "realized_pnl": 0.0,
            "trades": 0,
            "volume": 0.0,
            "wins": 0,
            "losses": 0,
            "opportunities": 0,
            "open_orders": 0,
        },
    }


class DashboardState:
    """Holds the current state for the dashboard."""

    def __init__(self):
        self.markets: dict = {}
        self.opportunities: list = []
        self.signals: list = []
        self.orders: list = []
        self.trades: list = []
        self.portfolio: dict = {}
        self.risk: dict = {}
        self.stats: dict = {}
        self.timing: dict = {}
        self.operational: dict = {}
        self.is_running: bool = False
        self.mode: str = "dry_run"
        self.last_update: datetime = datetime.utcnow()
        self.started_at: datetime = datetime.utcnow()

        self.bots: dict = _default_bots()
        self.nav = {
            "sections": [
                {"id": "overview", "label": "Overview"},
                {"id": "bots", "label": "Bots"},
                {"id": "markets", "label": "Markets"},
                {"id": "signals", "label": "Signals"},
                {"id": "cross", "label": "Cross"},
                {"id": "risk", "label": "Risk"},
            ]
        }

        self.cross_platform: dict = {
            "enabled": False,
            "kalshi_markets": 0,
            "polymarket_markets": 0,
            "matched_pairs": 0,
            "kalshi_orderbooks": 0,
            "cross_opportunities": [],
            "matched_pairs_data": [],
            "matching_progress": 0,
            "matching_checked": 0,
            "matching_total": 0,
            "matching_status": "idle",
        }
        self.soccer: dict = {
            "scheduled": 0,
            "monitoring": 0,
            "active_window": 0,
            "executed": 0,
            "settled": 0,
            "discarded": 0,
            "matches": [],
        }

        self._connections: list[WebSocket] = []

    def to_dict(self) -> dict:
        uptime = (datetime.utcnow() - self.started_at).total_seconds()
        bots_list = list(self.bots.values())
        total_bot_pnl = sum(b.get("realized_pnl", 0) for b in bots_list)
        return {
            "markets": self.markets,
            "opportunities": self.opportunities[-50:],
            "signals": self.signals[-50:],
            "orders": self.orders,
            "trades": self.trades[-100:],
            "portfolio": self.portfolio,
            "risk": self.risk,
            "stats": self.stats,
            "timing": self.timing,
            "operational": self.operational,
            "cross_platform": self.cross_platform,
            "soccer": self.soccer,
            "bots": self.bots,
            "bots_list": bots_list,
            "bots_total_pnl": total_bot_pnl,
            "nav": self.nav,
            "is_running": self.is_running,
            "mode": self.mode,
            "last_update": self.last_update.isoformat(),
            "started_at": self.started_at.isoformat(),
            "uptime_seconds": uptime,
        }

    async def broadcast(self, data: dict) -> None:
        if not self._connections:
            return
        message = json.dumps(data)
        disconnected = []
        for ws in self._connections:
            try:
                await ws.send_text(message)
            except Exception:
                disconnected.append(ws)
        for ws in disconnected:
            self._connections.remove(ws)

    def set_bot_status(self, strategy: str, status: str, enabled: Optional[bool] = None) -> None:
        sid = normalize_strategy(strategy)
        if sid not in self.bots:
            return
        self.bots[sid]["status"] = status
        if enabled is not None:
            self.bots[sid]["enabled"] = enabled

    def record_bot_opportunity(self, strategy: str) -> None:
        sid = normalize_strategy(strategy)
        if sid not in self.bots:
            return
        self.bots[sid]["opportunities"] += 1
        if self.bots[sid]["enabled"] and self.bots[sid]["status"] != "paused":
            self.bots[sid]["status"] = "running"

    def record_bot_trade(
        self,
        strategy: str,
        price: float,
        size: float,
        pnl: float = 0.0,
    ) -> None:
        sid = normalize_strategy(strategy)
        if sid not in self.bots:
            return
        bot = self.bots[sid]
        notional = abs(price * size)
        bot["trades"] += 1
        bot["volume"] += notional
        bot["realized_pnl"] += pnl
        if pnl > 0:
            bot["wins"] += 1
        elif pnl < 0:
            bot["losses"] += 1
        if bot["enabled"]:
            bot["status"] = "running"

    def add_opportunity(self, opportunity: dict) -> None:
        opportunity["timestamp"] = datetime.utcnow().isoformat()
        strategy = normalize_strategy(
            opportunity.get("strategy") or opportunity.get("type")
        )
        opportunity["strategy"] = strategy
        self.opportunities.append(opportunity)
        self.record_bot_opportunity(strategy)
        if len(self.opportunities) > 200:
            self.opportunities = self.opportunities[-100:]

    def add_signal(self, signal: dict) -> None:
        signal["timestamp"] = datetime.utcnow().isoformat()
        if "strategy" in signal:
            signal["strategy"] = normalize_strategy(signal["strategy"])
        self.signals.append(signal)
        if len(self.signals) > 200:
            self.signals = self.signals[-100:]

    def add_trade(self, trade: dict) -> None:
        trade["timestamp"] = datetime.utcnow().isoformat()
        strategy = normalize_strategy(
            trade.get("strategy") or trade.get("strategy_tag") or trade.get("type")
        )
        trade["strategy"] = strategy
        pnl = float(trade.get("pnl") or trade.get("realized_pnl") or 0.0)
        # Dry-run heuristic: attribute a small edge if none provided
        if trade.get("defer_pnl"):
            trade["pnl"] = pnl
        elif pnl == 0.0 and trade.get("edge") is not None:
            pnl = float(trade["edge"]) * float(trade.get("size") or 0)
            trade["pnl"] = pnl
        elif pnl == 0.0 and self.mode == "dry_run":
            # Simulate modest edge capture for demo visibility
            size = float(trade.get("size") or 0)
            price = float(trade.get("price") or 0)
            pnl = round(size * price * 0.008, 4)
            trade["pnl"] = pnl

        self.trades.append(trade)
        self.record_bot_trade(
            strategy,
            price=float(trade.get("price") or 0),
            size=float(trade.get("size") or 0),
            pnl=pnl,
        )
        if len(self.trades) > 500:
            self.trades = self.trades[-250:]

    def add_cross_platform_opportunity(self, opportunity: dict) -> None:
        opportunity["timestamp"] = datetime.utcnow().isoformat()
        opportunity["strategy"] = "cross_platform"
        self.cross_platform["cross_opportunities"].append(opportunity)
        self.record_bot_opportunity("cross_platform")
        if len(self.cross_platform["cross_opportunities"]) > 100:
            self.cross_platform["cross_opportunities"] = (
                self.cross_platform["cross_opportunities"][-50:]
            )

    def update_cross_platform_stats(
        self,
        kalshi_markets: int,
        polymarket_markets: int,
        matched_pairs: int,
        enabled: bool = True,
        matched_pairs_data: list = None,
    ) -> None:
        self.cross_platform["enabled"] = enabled
        self.cross_platform["kalshi_markets"] = kalshi_markets
        self.cross_platform["polymarket_markets"] = polymarket_markets
        self.cross_platform["matched_pairs"] = matched_pairs
        if matched_pairs_data is not None:
            self.cross_platform["matched_pairs_data"] = matched_pairs_data
        self.set_bot_status(
            "cross_platform",
            "running" if enabled else "paused",
            enabled=enabled,
        )


dashboard_state = DashboardState()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Arb Desk",
        description="Strategy desk for Polymarket / Kalshi bots",
        version="2.0.0",
    )

    static_dir = Path(__file__).parent / "static"
    static_dir.mkdir(exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    templates_dir = Path(__file__).parent / "templates"
    templates_dir.mkdir(exist_ok=True)

    @app.get("/", response_class=HTMLResponse)
    async def index():
        html_path = templates_dir / "index.html"
        if html_path.exists():
            return HTMLResponse(html_path.read_text(encoding="utf-8"))
        return HTMLResponse("<h1>Dashboard template missing</h1>", status_code=500)

    @app.get("/api/state")
    async def get_state():
        return dashboard_state.to_dict()

    @app.get("/api/bots")
    async def get_bots():
        return {
            "bots": dashboard_state.bots,
            "total_pnl": sum(b["realized_pnl"] for b in dashboard_state.bots.values()),
        }

    @app.get("/api/markets")
    async def get_markets():
        return {"markets": dashboard_state.markets}

    @app.get("/api/opportunities")
    async def get_opportunities():
        return {"opportunities": dashboard_state.opportunities[-50:]}

    @app.get("/api/portfolio")
    async def get_portfolio():
        return dashboard_state.portfolio

    @app.get("/api/risk")
    async def get_risk():
        return dashboard_state.risk

    @app.get("/api/timing")
    async def get_timing():
        return dashboard_state.timing

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        dashboard_state._connections.append(websocket)
        try:
            await websocket.send_text(json.dumps({
                "type": "initial",
                "data": dashboard_state.to_dict(),
            }))
            while True:
                try:
                    data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                    msg = json.loads(data)
                    if msg.get("type") == "ping":
                        await websocket.send_text(json.dumps({"type": "pong"}))
                except asyncio.TimeoutError:
                    await websocket.send_text(json.dumps({"type": "heartbeat"}))
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            if websocket in dashboard_state._connections:
                dashboard_state._connections.remove(websocket)

    return app


# Back-compat for imports that expect this symbol
def get_embedded_html() -> str:
    path = Path(__file__).parent / "templates" / "index.html"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return "<h1>Dashboard template missing</h1>"


app = create_app()
