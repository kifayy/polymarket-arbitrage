"""
SQLite persistence for match state and trade settlement journals.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from bots.soccer_draw_bias.models import MatchRecord, MatchStatus, TradeJournalEntry

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS matches (
    match_id INTEGER PRIMARY KEY,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    favorite_team TEXT NOT NULL,
    polymarket_market_id TEXT NOT NULL,
    token_id_favorite_yes TEXT,
    token_id_favorite_no TEXT,
    pre_match_favorite_implied_prob REAL,
    favorite_is_home INTEGER,
    status TEXT NOT NULL,
    current_minute INTEGER,
    current_score TEXT,
    discard_reason TEXT,
    league_id INTEGER,
    league_name TEXT,
    polymarket_question TEXT,
    market_volume REAL,
    updated_at TEXT,
    payload TEXT
);

CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id INTEGER NOT NULL,
    execution_ts TEXT NOT NULL,
    execution_minute INTEGER,
    execution_score_home INTEGER,
    execution_score_away INTEGER,
    target_team TEXT,
    execution_price REAL,
    shares REAL,
    final_score_home INTEGER,
    final_score_away INTEGER,
    outcome TEXT,
    pnl REAL,
    polymarket_market_id TEXT,
    order_id TEXT,
    home_team TEXT,
    away_team TEXT,
    UNIQUE(match_id)
);

CREATE TABLE IF NOT EXISTS state_transitions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id INTEGER NOT NULL,
    from_status TEXT,
    to_status TEXT NOT NULL,
    minute INTEGER,
    score TEXT,
    reason TEXT,
    ts TEXT NOT NULL
);
"""


class TradeStore:
    """Local SQLite journal (Postgres-compatible column layout)."""

    def __init__(self, db_path: str = "data/soccer_draw_bias.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def upsert_match(self, match: MatchRecord) -> None:
        payload = {
            "token_id_favorite_yes": match.token_id_favorite_yes,
            "token_id_favorite_no": match.token_id_favorite_no,
            "execution_price": match.execution_price,
            "shares": match.shares,
            "order_id": match.order_id,
            "outcome": match.outcome,
            "pnl": match.pnl,
        }
        self._conn.execute(
            """
            INSERT INTO matches (
                match_id, home_team, away_team, favorite_team, polymarket_market_id,
                token_id_favorite_yes, token_id_favorite_no,
                pre_match_favorite_implied_prob, favorite_is_home, status,
                current_minute, current_score, discard_reason, league_id, league_name,
                polymarket_question, market_volume, updated_at, payload
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(match_id) DO UPDATE SET
                status=excluded.status,
                current_minute=excluded.current_minute,
                current_score=excluded.current_score,
                discard_reason=excluded.discard_reason,
                updated_at=excluded.updated_at,
                payload=excluded.payload
            """,
            (
                match.match_id,
                match.home_team,
                match.away_team,
                match.favorite_team,
                match.polymarket_market_id,
                match.token_id_favorite_yes,
                match.token_id_favorite_no,
                match.pre_match_favorite_implied_prob,
                1 if match.favorite_is_home else 0,
                match.status.value,
                match.current_minute,
                json.dumps(match.current_score),
                match.discard_reason,
                match.league_id,
                match.league_name,
                match.polymarket_question,
                match.market_volume,
                (match.updated_at or datetime.utcnow()).isoformat(),
                json.dumps(payload),
            ),
        )
        self._conn.commit()

    def log_transition(
        self,
        match_id: int,
        from_status: Optional[str],
        to_status: str,
        minute: int,
        score: list[int],
        reason: str = "",
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO state_transitions
                (match_id, from_status, to_status, minute, score, reason, ts)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                match_id,
                from_status,
                to_status,
                minute,
                json.dumps(score),
                reason,
                datetime.utcnow().isoformat(),
            ),
        )
        self._conn.commit()

    def upsert_trade_from_match(self, match: MatchRecord) -> None:
        if match.execution_ts is None or match.execution_price is None:
            return
        score = match.execution_score or match.current_score
        final = match.final_score
        self._conn.execute(
            """
            INSERT INTO trades (
                match_id, execution_ts, execution_minute,
                execution_score_home, execution_score_away,
                target_team, execution_price, shares,
                final_score_home, final_score_away,
                outcome, pnl, polymarket_market_id, order_id,
                home_team, away_team
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(match_id) DO UPDATE SET
                final_score_home=excluded.final_score_home,
                final_score_away=excluded.final_score_away,
                outcome=excluded.outcome,
                pnl=excluded.pnl
            """,
            (
                match.match_id,
                match.execution_ts.isoformat(),
                match.execution_minute,
                score[0],
                score[1],
                match.favorite_team,
                match.execution_price,
                match.shares,
                final[0] if final else None,
                final[1] if final else None,
                match.outcome,
                match.pnl,
                match.polymarket_market_id,
                match.order_id or "",
                match.home_team,
                match.away_team,
            ),
        )
        self._conn.commit()

    def load_open_matches(self) -> list[dict]:
        rows = self._conn.execute(
            """
            SELECT * FROM matches
            WHERE status NOT IN ('SETTLED', 'DISCARDED')
            """
        ).fetchall()
        return [dict(r) for r in rows]

    def list_trades(self, limit: int = 100) -> list[TradeJournalEntry]:
        rows = self._conn.execute(
            "SELECT * FROM trades ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        out: list[TradeJournalEntry] = []
        for r in rows:
            out.append(
                TradeJournalEntry(
                    match_id=r["match_id"],
                    execution_ts=datetime.fromisoformat(r["execution_ts"]),
                    execution_minute=r["execution_minute"] or 0,
                    execution_score_home=r["execution_score_home"] or 0,
                    execution_score_away=r["execution_score_away"] or 0,
                    target_team=r["target_team"] or "",
                    execution_price=r["execution_price"] or 0.0,
                    shares=r["shares"] or 0.0,
                    final_score_home=r["final_score_home"],
                    final_score_away=r["final_score_away"],
                    outcome=r["outcome"],
                    pnl=r["pnl"],
                    polymarket_market_id=r["polymarket_market_id"] or "",
                    order_id=r["order_id"] or "",
                    home_team=r["home_team"] or "",
                    away_team=r["away_team"] or "",
                )
            )
        return out
