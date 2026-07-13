"""
Data models for the 90-minute draw-bias soccer bot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class MatchStatus(str, Enum):
    """Match lifecycle within the bot engine."""

    SCHEDULED = "SCHEDULED"
    MONITORING = "MONITORING"
    ACTIVE_WINDOW = "ACTIVE_WINDOW"
    EXECUTED = "EXECUTED"
    SETTLED = "SETTLED"
    DISCARDED = "DISCARDED"


@dataclass
class MatchRecord:
    """
    Tracked soccer match linked to a Polymarket favorite-win market.

    Spec entity fields (§2).
    """

    match_id: int
    home_team: str
    away_team: str
    polymarket_market_id: str
    token_id_favorite_yes: str
    token_id_favorite_no: str
    pre_match_favorite_implied_prob: float
    favorite_is_home: bool
    favorite_team: str
    underdog_team: str

    current_minute: int = 0
    current_score: list[int] = field(default_factory=lambda: [0, 0])
    status: MatchStatus = MatchStatus.SCHEDULED

    # Sports / market metadata
    league_id: int = 0
    league_name: str = ""
    kickoff_utc: Optional[datetime] = None
    polymarket_question: str = ""
    market_volume: float = 0.0
    sports_status: str = ""  # API-Football short status e.g. NS, 1H, 2H, FT

    # Fail-safe flags
    favorite_red_cards: int = 0
    var_in_progress: bool = False
    discard_reason: str = ""

    # Execution / settlement
    execution_price: Optional[float] = None
    shares: Optional[float] = None
    execution_minute: Optional[int] = None
    execution_score: Optional[list[int]] = None
    execution_ts: Optional[datetime] = None
    order_id: Optional[str] = None
    final_score: Optional[list[int]] = None
    outcome: Optional[str] = None  # Win / Loss
    pnl: Optional[float] = None
    settled_ts: Optional[datetime] = None

    updated_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def is_tied(self) -> bool:
        return self.current_score[0] == self.current_score[1]

    @property
    def favorite_goals(self) -> int:
        return self.current_score[0] if self.favorite_is_home else self.current_score[1]

    @property
    def underdog_goals(self) -> int:
        return self.current_score[1] if self.favorite_is_home else self.current_score[0]

    @property
    def favorite_winning(self) -> bool:
        return self.favorite_goals > self.underdog_goals

    @property
    def favorite_losing(self) -> bool:
        return self.favorite_goals < self.underdog_goals


@dataclass
class LiveFixtureSnapshot:
    """Normalized live fixture payload from API-Football."""

    match_id: int
    home_team: str
    away_team: str
    home_goals: int
    away_goals: int
    elapsed: int
    status_short: str
    league_id: int = 0
    league_name: str = ""
    kickoff_utc: Optional[datetime] = None
    favorite_red_cards: int = 0
    var_in_progress: bool = False
    http_ok: bool = True
    # 90'+stoppage score only (ignores ET / pens). Used for Polymarket settlement.
    regulation_home: Optional[int] = None
    regulation_away: Optional[int] = None

    @property
    def regulation_score(self) -> list[int]:
        """Score that Polymarket Match Winner markets resolve on."""
        if self.regulation_home is not None and self.regulation_away is not None:
            return [self.regulation_home, self.regulation_away]
        return [self.home_goals, self.away_goals]


@dataclass
class TradeJournalEntry:
    """Persisted execution + settlement record."""

    match_id: int
    execution_ts: datetime
    execution_minute: int
    execution_score_home: int
    execution_score_away: int
    target_team: str
    execution_price: float
    shares: float
    final_score_home: Optional[int] = None
    final_score_away: Optional[int] = None
    outcome: Optional[str] = None
    pnl: Optional[float] = None
    polymarket_market_id: str = ""
    order_id: str = ""
    home_team: str = ""
    away_team: str = ""
