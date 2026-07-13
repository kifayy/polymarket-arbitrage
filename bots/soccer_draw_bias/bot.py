"""
Orchestrator for the 90-minute draw-bias soccer bot.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

from polymarket_client import PolymarketClient
from core.portfolio import Portfolio
from core.risk_manager import RiskConfig, RiskManager
from utils.config_loader import BotConfig

from bots.soccer_draw_bias.engine import DrawBiasConfig, MatchStateMachine
from bots.soccer_draw_bias.executor import DrawBiasExecutor
from bots.soccer_draw_bias.models import MatchRecord, MatchStatus
from bots.soccer_draw_bias.persistence import TradeStore
from bots.soccer_draw_bias.polymarket_soccer import (
    discover_soccer_markets,
    map_fixture_to_market,
)
from bots.soccer_draw_bias.sports_client import (
    ApiFootballClient,
    SportsApiError,
    is_full_time,
)

logger = logging.getLogger(__name__)


class SoccerDrawBiasBot:
    """Daily discovery + live polling loop for the draw-bias strategy."""

    STRATEGY_ID = "soccer_draw_bias"

    def __init__(
        self,
        config: BotConfig,
        draw_config: Optional[DrawBiasConfig] = None,
        dashboard=None,
    ):
        self.config = config
        if draw_config is not None:
            self.draw_config = draw_config
        else:
            s = config.soccer_draw_bias
            self.draw_config = DrawBiasConfig(
                enabled=s.enabled,
                league_ids=list(s.league_ids),
                min_prematch_favorite_prob=s.min_prematch_favorite_prob,
                min_market_volume=s.min_market_volume,
                min_match_similarity=s.min_match_similarity,
                active_window_start=s.active_window_start,
                active_window_end=s.active_window_end,
                target_max_buy_price=s.target_max_buy_price,
                risk_unit_usd=s.risk_unit_usd,
                poll_interval_monitoring_sec=s.poll_interval_monitoring_sec,
                poll_interval_active_sec=s.poll_interval_active_sec,
                db_path=s.db_path,
                season=s.season,
            )
        self.dashboard = dashboard
        self._running = False
        self._shutdown = asyncio.Event()

        self.sports: Optional[ApiFootballClient] = None
        self.poly: Optional[PolymarketClient] = None
        self.portfolio: Optional[Portfolio] = None
        self.risk_manager: Optional[RiskManager] = None
        self.store: Optional[TradeStore] = None
        self.state_machine = MatchStateMachine(self.draw_config)
        self.executor: Optional[DrawBiasExecutor] = None

        self.matches: dict[int, MatchRecord] = {}
        self._last_discovery_date: Optional[str] = None
        self._soccer_candidates = []

    def _dash(self, method: str, *args, **kwargs) -> None:
        if self.dashboard is None:
            return
        fn = getattr(self.dashboard, method, None)
        if callable(fn):
            try:
                fn(*args, **kwargs)
            except Exception as e:
                logger.debug(f"Dashboard hook {method} failed: {e}")

    def _notify_scheduled(self, match: MatchRecord) -> None:
        self._dash(
            "add_opportunity",
            opportunity_type=self.STRATEGY_ID,
            strategy=self.STRATEGY_ID,
            market_id=match.polymarket_market_id,
            edge=match.pre_match_favorite_implied_prob,
            suggested_size=self.draw_config.risk_unit_usd
            / self.draw_config.target_max_buy_price,
            question=f"{match.home_team} vs {match.away_team} (fav {match.favorite_team})",
        )

    def _notify_active(self, match: MatchRecord) -> None:
        self._dash(
            "add_opportunity",
            opportunity_type=self.STRATEGY_ID,
            strategy=self.STRATEGY_ID,
            market_id=match.polymarket_market_id,
            edge=self.draw_config.target_max_buy_price,
            suggested_size=self.draw_config.risk_unit_usd
            / max(self.draw_config.target_max_buy_price, 0.01),
            question=(
                f"ACTIVE {match.home_team} vs {match.away_team} "
                f"{match.current_score} @ {match.current_minute}'"
            ),
        )

    def _notify_fill(self, match: MatchRecord) -> None:
        self._dash(
            "add_trade",
            side="buy",
            price=float(match.execution_price or 0),
            size=float(match.shares or 0),
            strategy=self.STRATEGY_ID,
            market_id=match.polymarket_market_id,
            token_type="no",
            defer_pnl=True,
            pnl=0.0,
            question=f"{match.favorite_team} No @ {match.execution_minute}'",
        )

    def _notify_settled(self, match: MatchRecord) -> None:
        self._dash(
            "add_trade",
            side="settle",
            price=float(match.execution_price or 0),
            size=float(match.shares or 0),
            strategy=self.STRATEGY_ID,
            market_id=match.polymarket_market_id,
            pnl=float(match.pnl or 0),
            realized_pnl=float(match.pnl or 0),
            outcome=match.outcome,
            question=f"SETTLED {match.home_team} vs {match.away_team} → {match.outcome}",
        )

    async def start(self) -> None:
        await self.initialize()
        if not self._running:
            return
        await self.run_discovery(force=True)
        await self._main_loop()

    async def initialize(self) -> None:
        """Connect clients and build portfolio/risk/executor (no polling yet)."""
        if not self.draw_config.enabled:
            logger.warning("soccer_draw_bias.enabled is false — exiting")
            return

        api_key = getattr(self.config.api, "api_football_key", "") or ""
        if not api_key:
            raise RuntimeError(
                "API_FOOTBALL_KEY is required (set in .env / api.api_football_key)"
            )

        logger.info("=" * 60)
        logger.info("Soccer Draw Bias Bot Starting")
        logger.info("=" * 60)
        logger.info(f"Mode: {'DRY RUN' if self.config.is_dry_run else 'LIVE'}")
        logger.info(f"Leagues: {self.draw_config.league_ids}")
        logger.info(
            f"Window: {self.draw_config.active_window_start}-"
            f"{self.draw_config.active_window_end}' | "
            f"max_buy={self.draw_config.target_max_buy_price} | "
            f"risk_unit=${self.draw_config.risk_unit_usd}"
        )

        self.sports = ApiFootballClient(
            api_key=api_key,
            timeout=self.config.api.timeout_seconds,
        )
        await self.sports.connect()

        self.poly = PolymarketClient(
            rest_url=self.config.api.polymarket_rest_url,
            ws_url=self.config.api.polymarket_ws_url,
            gamma_url=self.config.api.gamma_api_url,
            api_key=self.config.api.api_key,
            api_secret=self.config.api.api_secret,
            passphrase=self.config.api.passphrase,
            private_key=self.config.api.private_key,
            funder_address=getattr(self.config.api, "funder_address", "") or None,
            signature_type=getattr(self.config.api, "signature_type", 0),
            timeout=self.config.api.timeout_seconds,
            max_retries=self.config.api.max_retries,
            retry_delay=self.config.api.retry_delay_seconds,
            dry_run=self.config.is_dry_run,
        )
        await self.poly.connect()

        initial = (
            self.config.mode.dry_run_initial_balance if self.config.is_dry_run else 0.0
        )
        self.portfolio = Portfolio(initial_balance=initial)
        self.risk_manager = RiskManager(
            RiskConfig(
                max_position_per_market=max(
                    self.config.risk.max_position_per_market,
                    self.draw_config.risk_unit_usd,
                ),
                max_global_exposure=self.config.risk.max_global_exposure,
                max_daily_loss=self.config.risk.max_daily_loss,
                max_drawdown_pct=self.config.risk.max_drawdown_pct,
                trade_only_high_volume=False,
                min_24h_volume=self.draw_config.min_market_volume,
                whitelist=self.config.risk.whitelist,
                blacklist=self.config.risk.blacklist,
                kill_switch_enabled=self.config.risk.kill_switch_enabled,
                auto_unwind_on_breach=self.config.risk.auto_unwind_on_breach,
            )
        )
        self.store = TradeStore(self.draw_config.db_path)
        self.executor = DrawBiasExecutor(
            client=self.poly,
            risk_manager=self.risk_manager,
            portfolio=self.portfolio,
            config=self.draw_config,
            state_machine=self.state_machine,
            immediate_fill_dry_run=True,
        )
        self._running = True

    async def run_loop(self) -> None:
        """Daily discovery + polling until stop()."""
        if not self._running:
            return
        await self.run_discovery(force=True)
        await self._main_loop()

    async def stop(self) -> None:
        self._running = False
        self._shutdown.set()
        if self.sports:
            await self.sports.close()
        if self.poly:
            await self.poly.disconnect()
        if self.store:
            self.store.close()
        logger.info("Soccer Draw Bias Bot stopped")

    async def run_discovery(self, force: bool = False) -> None:
        """Phase 1: map today's fixtures to Polymarket 90-minute markets."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if not force and self._last_discovery_date == today:
            return

        assert self.sports and self.poly and self.store
        logger.info(f"Running daily discovery for {today}")

        # Pull today ± 1 day so timezone edges don't drop fixtures.
        # Free-tier API-Football only allows a narrow recent date window anyway.
        from datetime import timedelta

        base = datetime.now(timezone.utc).date()
        dates = [
            (base + timedelta(days=delta)).strftime("%Y-%m-%d")
            for delta in (-1, 0, 1)
        ]
        fixtures = []
        seen_ids: set[int] = set()
        for date_str in dates:
            day_fixtures = await self.sports.get_fixtures_by_date(
                date_str,
                league_ids=self.draw_config.league_ids or [],
                season=self.draw_config.season,
            )
            for snap in day_fixtures:
                if snap.match_id not in seen_ids:
                    seen_ids.add(snap.match_id)
                    fixtures.append(snap)
        logger.info(f"Sports fixtures (dates {dates}): {len(fixtures)}")

        self._soccer_candidates = await discover_soccer_markets(
            self.poly,
            min_volume=self.draw_config.min_market_volume,
        )

        mapped = 0
        for snap in fixtures:
            if snap.match_id in self.matches:
                continue
            linked = map_fixture_to_market(
                snap.home_team,
                snap.away_team,
                self._soccer_candidates,
                min_favorite_prob=self.draw_config.min_prematch_favorite_prob,
                min_match_score=self.draw_config.min_match_similarity,
            )
            if not linked:
                continue

            rec = MatchRecord(
                match_id=snap.match_id,
                home_team=snap.home_team,
                away_team=snap.away_team,
                polymarket_market_id=linked.market_id,
                token_id_favorite_yes=linked.token_id_favorite_yes,
                token_id_favorite_no=linked.token_id_favorite_no,
                pre_match_favorite_implied_prob=linked.pre_match_favorite_implied_prob,
                favorite_is_home=linked.favorite_is_home,
                favorite_team=linked.favorite_team,
                underdog_team=linked.underdog_team,
                status=MatchStatus.SCHEDULED,
                league_id=snap.league_id,
                league_name=snap.league_name,
                kickoff_utc=snap.kickoff_utc,
                polymarket_question=linked.question,
                market_volume=linked.volume,
                sports_status=snap.status_short,
                current_minute=snap.elapsed,
                current_score=[snap.home_goals, snap.away_goals],
            )
            self.matches[rec.match_id] = rec
            self.risk_manager.update_market_volume(rec.polymarket_market_id, linked.volume)
            self.store.upsert_match(rec)
            self.store.log_transition(
                rec.match_id, None, MatchStatus.SCHEDULED.value, 0, [0, 0], "discovery"
            )
            self._notify_scheduled(rec)
            mapped += 1
            logger.info(
                f"SCHEDULED match={rec.match_id} {rec.home_team} vs {rec.away_team} | "
                f"fav={rec.favorite_team} ({rec.pre_match_favorite_implied_prob:.0%}) | "
                f"market={rec.polymarket_market_id}"
            )

        self._last_discovery_date = today
        logger.info(f"Discovery complete: {mapped} matches scheduled")

    async def _main_loop(self) -> None:
        assert self.sports and self.executor and self.store
        while self._running and not self._shutdown.is_set():
            try:
                # Roll discovery at 00:00 UTC
                await self.run_discovery(force=False)

                await self._poll_once()

                interval = self.state_machine.poll_interval_for(list(self.matches.values()))
                try:
                    await asyncio.wait_for(self._shutdown.wait(), timeout=interval)
                except asyncio.TimeoutError:
                    pass
            except Exception as e:
                logger.exception(f"Main loop error: {e}")
                await asyncio.sleep(30)

    async def _poll_once(self) -> None:
        assert self.sports and self.executor and self.store
        active = [
            m
            for m in self.matches.values()
            if m.status
            not in (MatchStatus.SETTLED, MatchStatus.DISCARDED)
        ]
        if not active:
            return

        trading_suspended = self.sports.trading_suspended
        if trading_suspended:
            logger.error(
                "Sports API outage >3 min — trading suspended "
                f"(failures={self.sports.consecutive_failures})"
            )

        # Batch live pull for configured leagues
        live_by_id: dict[int, object] = {}
        try:
            lives = await self.sports.get_live_fixtures(self.draw_config.league_ids)
            live_by_id = {s.match_id: s for s in lives}
        except SportsApiError as e:
            logger.warning(f"Live fixtures poll failed: {e}")

        for match in active:
            prev_status = match.status
            snap = live_by_id.get(match.match_id)

            if snap is None:
                # Scheduled matches: wait until they appear on the live feed
                # (avoids burning API-Football free-tier quota on NS polls).
                if match.status == MatchStatus.SCHEDULED:
                    continue
                # Monitoring / active / executed need an explicit fetch if
                # they dropped off the live endpoint (e.g. just went FT).
                try:
                    snap = await self.sports.get_fixture(match.match_id)
                except SportsApiError as e:
                    logger.warning(f"Fixture {match.match_id} fetch failed: {e}")
                    continue
            if snap is None:
                continue

            # Enrich cards/VAR only near / inside the trade window
            near_window = snap.elapsed >= max(0, self.draw_config.active_window_start - 5)
            if match.status in (
                MatchStatus.ACTIVE_WINDOW,
                MatchStatus.EXECUTED,
            ) or (
                match.status == MatchStatus.MONITORING and near_window
            ):
                snap = await self.sports.enrich_with_events(snap, match.favorite_team)

            self.state_machine.apply_live_update(
                match, snap, trading_suspended=trading_suspended
            )

            if match.status != prev_status:
                self.store.log_transition(
                    match.match_id,
                    prev_status.value,
                    match.status.value,
                    match.current_minute,
                    match.current_score,
                    match.discard_reason,
                )
                if match.status == MatchStatus.ACTIVE_WINDOW:
                    self._notify_active(match)
                if match.status == MatchStatus.SETTLED:
                    self._notify_settled(match)

            if (
                match.status == MatchStatus.ACTIVE_WINDOW
                and not trading_suspended
                and not match.var_in_progress
            ):
                result = await self.executor.maybe_execute(match)
                if result.success:
                    self.store.upsert_trade_from_match(match)
                    self.store.log_transition(
                        match.match_id,
                        MatchStatus.ACTIVE_WINDOW.value,
                        MatchStatus.EXECUTED.value,
                        match.current_minute,
                        match.current_score,
                        f"filled@{result.price}",
                    )
                    self._notify_fill(match)

            if match.status == MatchStatus.SETTLED:
                self.store.upsert_trade_from_match(match)

            # If executed and FT arrived via apply_live_update
            if match.status == MatchStatus.EXECUTED and is_full_time(snap.status_short):
                prev = match.status
                self.state_machine.settle(match, snap)
                self.store.upsert_trade_from_match(match)
                if match.status == MatchStatus.SETTLED and prev != MatchStatus.SETTLED:
                    self._notify_settled(match)

            self.store.upsert_match(match)
