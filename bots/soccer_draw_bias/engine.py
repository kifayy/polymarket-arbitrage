"""
Match state machine and trading gates for the draw-bias strategy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from bots.soccer_draw_bias.models import LiveFixtureSnapshot, MatchRecord, MatchStatus
from bots.soccer_draw_bias.sports_client import (
    is_full_time,
    is_live,
    is_not_started,
    is_post_regulation,
)

logger = logging.getLogger(__name__)


@dataclass
class DrawBiasConfig:
    """Strategy knobs (mirrors config.yaml soccer_draw_bias section)."""

    enabled: bool = True
    league_ids: list[int] | None = None
    min_prematch_favorite_prob: float = 0.52
    min_market_volume: float = 5000.0
    min_match_similarity: float = 0.72
    active_window_start: int = 75
    active_window_end: int = 88
    target_max_buy_price: float = 0.20
    risk_unit_usd: float = 15.0
    poll_interval_idle_sec: float = 600.0  # only SCHEDULED / empty universe
    poll_interval_monitoring_sec: float = 300.0  # live but before ~70'
    poll_interval_near_window_sec: float = 90.0  # monitoring >= 70'
    poll_interval_active_sec: float = 60.0  # ACTIVE_WINDOW / EXECUTED
    near_window_minute: int = 70
    db_path: str = "data/soccer_draw_bias.db"
    season: int | None = None
    # If true, date discovery keeps any league that maps to Polymarket (more games)
    markets_first: bool = True

    def __post_init__(self) -> None:
        if self.league_ids is None:
            # EPL, UCL, La Liga, Serie A, Bundesliga, Ligue 1, MLS, World Cup, UEL
            self.league_ids = [39, 2, 140, 135, 78, 61, 253, 1, 3]


class MatchStateMachine:
    """
    Transitions matches through SCHEDULED → MONITORING → ACTIVE_WINDOW →
    EXECUTED → SETTLED (or DISCARDED).
    """

    def __init__(self, config: DrawBiasConfig):
        self.config = config

    def apply_live_update(
        self,
        match: MatchRecord,
        snap: LiveFixtureSnapshot,
        trading_suspended: bool = False,
    ) -> MatchRecord:
        """Update match from a live sports snapshot and apply gates."""
        match.current_minute = int(snap.elapsed or 0)
        match.current_score = [snap.home_goals, snap.away_goals]
        match.sports_status = snap.status_short
        match.favorite_red_cards = snap.favorite_red_cards
        match.var_in_progress = snap.var_in_progress
        match.updated_at = datetime.utcnow()

        if match.status in (MatchStatus.SETTLED, MatchStatus.DISCARDED):
            return match

        # Settlement path
        if match.status == MatchStatus.EXECUTED and is_full_time(snap.status_short):
            return self.settle(match, snap)

        if is_full_time(snap.status_short) and match.status != MatchStatus.EXECUTED:
            match.status = MatchStatus.DISCARDED
            match.discard_reason = match.discard_reason or "full_time_without_execution"
            match.final_score = [snap.home_goals, snap.away_goals]
            return match

        # Kickoff activation
        if match.status == MatchStatus.SCHEDULED:
            if is_live(snap.status_short) or (
                not is_not_started(snap.status_short) and snap.elapsed > 0
            ):
                match.status = MatchStatus.MONITORING
                logger.info(
                    f"Match {match.match_id} → MONITORING "
                    f"({match.home_team} vs {match.away_team})"
                )

        if match.status == MatchStatus.MONITORING:
            self._apply_monitoring_gates(match)

        if match.status == MatchStatus.ACTIVE_WINDOW:
            # Never trade once ET / pens begin — Match Winner already locked to 90'
            if is_post_regulation(snap.status_short):
                match.status = MatchStatus.DISCARDED
                match.discard_reason = "post_regulation_no_entry"
                return match
            self._apply_active_gates(match, trading_suspended)

        return match

    def _apply_monitoring_gates(self, match: MatchRecord) -> None:
        start = self.config.active_window_start

        if match.favorite_red_cards > 0 and match.current_minute < start:
            match.status = MatchStatus.DISCARDED
            match.discard_reason = "favorite_red_card_before_window"
            logger.info(f"Match {match.match_id} DISCARDED: favorite red card")
            return

        if match.current_minute < start:
            return

        # At/after 75'
        if match.favorite_red_cards > 0:
            match.status = MatchStatus.DISCARDED
            match.discard_reason = "favorite_red_card"
            return

        if not match.is_tied:
            match.status = MatchStatus.DISCARDED
            match.discard_reason = (
                "favorite_winning" if match.favorite_winning else "favorite_losing"
            )
            logger.info(
                f"Match {match.match_id} DISCARDED at {match.current_minute}': "
                f"score {match.current_score} ({match.discard_reason})"
            )
            return

        if match.current_minute > self.config.active_window_end:
            match.status = MatchStatus.DISCARDED
            match.discard_reason = "window_missed"
            return

        match.status = MatchStatus.ACTIVE_WINDOW
        logger.info(
            f"Match {match.match_id} → ACTIVE_WINDOW "
            f"tied {match.current_score} @ {match.current_minute}'"
        )

    def _apply_active_gates(
        self, match: MatchRecord, trading_suspended: bool
    ) -> None:
        if match.favorite_red_cards > 0:
            match.status = MatchStatus.DISCARDED
            match.discard_reason = "favorite_red_card"
            return

        if not match.is_tied:
            match.status = MatchStatus.DISCARDED
            match.discard_reason = "score_no_longer_tied"
            return

        if match.current_minute > self.config.active_window_end:
            match.status = MatchStatus.DISCARDED
            match.discard_reason = "past_active_window_end"
            return

        if trading_suspended:
            # Stay in ACTIVE_WINDOW but executor will refuse
            return

        if match.var_in_progress:
            # Pause — remain ACTIVE_WINDOW without executing
            return

    def should_evaluate_orderbook(self, match: MatchRecord) -> bool:
        if is_post_regulation(match.sports_status):
            return False
        return (
            match.status == MatchStatus.ACTIVE_WINDOW
            and match.is_tied
            and not match.var_in_progress
            and match.favorite_red_cards == 0
            and self.config.active_window_start
            <= match.current_minute
            <= self.config.active_window_end
        )

    def mark_executed(
        self,
        match: MatchRecord,
        price: float,
        shares: float,
        order_id: str,
    ) -> MatchRecord:
        match.status = MatchStatus.EXECUTED
        match.execution_price = price
        match.shares = shares
        match.order_id = order_id
        match.execution_minute = match.current_minute
        match.execution_score = list(match.current_score)
        match.execution_ts = datetime.utcnow()
        match.updated_at = datetime.utcnow()
        return match

    def settle(self, match: MatchRecord, snap: LiveFixtureSnapshot) -> MatchRecord:
        """
        Compute Win/Loss and PnL on the 90'+stoppage result only.

        Extra time and penalty shootouts do not affect Polymarket Match Winner.
        If France beat Spain on pens after 1-1 at 90', favorite-No still WINS
        when the favorite did not win in regulation.
        """
        reg = snap.regulation_score
        match.final_score = list(reg)
        match.current_score = list(reg)
        match.sports_status = snap.status_short
        match.settled_ts = datetime.utcnow()
        match.updated_at = datetime.utcnow()

        price = match.execution_price or 0.0
        shares = match.shares or 0.0
        cost = price * shares

        fav_goals = reg[0] if match.favorite_is_home else reg[1]
        dog_goals = reg[1] if match.favorite_is_home else reg[0]
        favorite_won_regulation = fav_goals > dog_goals

        if favorite_won_regulation:
            match.outcome = "Loss"
            match.pnl = -cost
        else:
            # Draw or underdog win in 90' → favorite No pays $1/share
            match.outcome = "Win"
            match.pnl = shares - cost

        match.status = MatchStatus.SETTLED
        logger.info(
            f"Match {match.match_id} SETTLED {match.outcome} pnl={match.pnl:.2f} "
            f"regulation={match.final_score} api_status={snap.status_short}"
        )
        return match

    def poll_interval_for(self, matches: list[MatchRecord]) -> float:
        """
        Quota-safe sleep: idle when nothing live, faster only near/inside window.
        """
        cfg = self.config
        if not matches:
            return cfg.poll_interval_idle_sec

        hot = {MatchStatus.ACTIVE_WINDOW, MatchStatus.EXECUTED}
        if any(m.status in hot for m in matches):
            return cfg.poll_interval_active_sec

        monitoring = [m for m in matches if m.status == MatchStatus.MONITORING]
        if monitoring:
            if any(m.current_minute >= cfg.near_window_minute for m in monitoring):
                return cfg.poll_interval_near_window_sec
            return cfg.poll_interval_monitoring_sec

        # Only SCHEDULED left
        return cfg.poll_interval_idle_sec
