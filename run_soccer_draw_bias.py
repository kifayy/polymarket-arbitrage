#!/usr/bin/env python3
"""
Soccer 90-Minute Draw Bias Bot
==============================

Paper-trades Polymarket favorite-No when a Tier-1/2 match is tied late.
Serves Arb Desk so Soccer Draw Bias appears as a bot card.

Usage:
    python run_soccer_draw_bias.py
    python run_soccer_draw_bias.py --port 8888
    python run_soccer_draw_bias.py --live
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys

import uvicorn

from utils.config_loader import load_config
from utils.logging_utils import setup_logging
from bots.soccer_draw_bias.bot import SoccerDrawBiasBot
from dashboard.server import app, dashboard_state
from dashboard.integration import DashboardIntegration


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Soccer draw-bias paper trading bot")
    p.add_argument("--config", default="config.yaml", help="Path to config YAML")
    p.add_argument(
        "--live",
        action="store_true",
        help="Place live Polymarket orders (default is dry_run from config)",
    )
    p.add_argument("--port", type=int, default=8888, help="Arb Desk port")
    p.add_argument("--no-dashboard", action="store_true", help="Skip web UI")
    return p.parse_args()


async def _run(bot: SoccerDrawBiasBot, port: int, with_dashboard: bool) -> None:
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()
    dash: DashboardIntegration | None = None
    server = None
    server_task = None

    def _signal_handler() -> None:
        logger.info("Shutdown signal received")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:
            signal.signal(sig, lambda *_: stop_event.set())

    await bot.initialize()
    if not bot._running:
        return

    if with_dashboard:
        dash = DashboardIntegration(
            data_feed=None,
            arb_engine=None,
            execution_engine=None,
            risk_manager=bot.risk_manager,
            portfolio=bot.portfolio,
            mode="dry_run" if bot.config.is_dry_run else "live",
            mm_enabled=False,
            cross_platform_enabled=False,
            soccer_enabled=True,
        )
        dashboard_state.set_bot_status("bundle_arb", "paused", enabled=False)
        dashboard_state.set_bot_status("cross_platform", "paused", enabled=False)
        dashboard_state.set_bot_status("market_making", "paused", enabled=False)
        dash.attach_soccer_bot(bot)
        bot.dashboard = dash
        await dash.start()

        uvi = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=port,
            log_level="warning",
            access_log=False,
        )
        server = uvicorn.Server(uvi)
        server_task = asyncio.create_task(server.serve())
        logger.info(f"Arb Desk: http://localhost:{port}  (Soccer Draw Bias card)")

    task = asyncio.create_task(bot.run_loop())
    stopper = asyncio.create_task(stop_event.wait())
    done, pending = await asyncio.wait(
        {task, stopper}, return_when=asyncio.FIRST_COMPLETED
    )
    await bot.stop()
    if dash:
        await dash.stop()
    if server is not None:
        server.should_exit = True
    if server_task:
        try:
            await asyncio.wait_for(server_task, timeout=3)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            server_task.cancel()
    for t in pending:
        t.cancel()
    if task in done and task.exception():
        raise task.exception()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if args.live:
        config.mode.trading_mode = "live"

    setup_logging(
        console_level=config.logging.console_level,
        file_level=config.logging.file_level,
        log_dir=config.logging.log_dir,
        main_log_file=config.logging.main_log_file,
        trades_log_file=config.logging.trades_log_file,
        opportunities_log_file=config.logging.opportunities_log_file,
        max_size_mb=config.logging.max_log_size_mb,
        backup_count=config.logging.backup_count,
    )

    bot = SoccerDrawBiasBot(config)
    try:
        asyncio.run(_run(bot, port=args.port, with_dashboard=not args.no_dashboard))
    except KeyboardInterrupt:
        logger.info("Interrupted")
    return 0


if __name__ == "__main__":
    sys.exit(main())
