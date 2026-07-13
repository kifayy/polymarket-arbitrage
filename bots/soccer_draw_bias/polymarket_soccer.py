"""
Polymarket soccer market discovery for 90-minute match-winner contracts.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Optional

from polymarket_client.api import PolymarketClient
from polymarket_client.models import Market

from bots.soccer_draw_bias.matcher import (
    extract_single_team_win,
    extract_teams_from_question,
    match_team_pair,
    team_similarity,
)

logger = logging.getLogger(__name__)

EXCLUDE_PATTERNS = [
    r"to advance",
    r"lift the trophy",
    r"lift trophy",
    r"to lift",
    r"qualify",
    r"make the final",
    r"win the (group|cup|tournament|championship|title)",
    r"champion",
    r"outright",
    r"winner of",
    r"both teams to score",
    r"\bbtts\b",
    r"over\s+\d",
    r"under\s+\d",
    r"total goals",
    r"correct score",
    r"exact score",
    r"first goal",
    r"player to",
    r"mvp",
    r"extra time",
    r"penalt",
    r"to go through",
]


@dataclass
class SoccerMarketCandidate:
    """A Polymarket binary market representing a team winning in 90 minutes."""

    market: Market
    team_a: str
    team_b: Optional[str]  # None for single-team "Will X win?"
    yes_implied: float
    no_implied: float
    volume: float
    is_single_team_win: bool = False


def is_excluded_market(question: str, description: str = "") -> bool:
    text = f"{question} {description}".lower()
    return any(re.search(pat, text) for pat in EXCLUDE_PATTERNS)


def _parse_outcome_prices(market: Market, raw: Optional[dict] = None) -> tuple[float, float]:
    """Best-effort YES/NO implied probs from Gamma fields."""
    # Market model doesn't store outcomePrices; caller may pass raw
    if raw:
        prices_raw = raw.get("outcomePrices") or raw.get("outcome_prices")
        if prices_raw:
            try:
                if isinstance(prices_raw, str):
                    prices = json.loads(prices_raw)
                else:
                    prices = prices_raw
                if isinstance(prices, list) and len(prices) >= 2:
                    return float(prices[0]), float(prices[1])
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
    return 0.0, 0.0


async def discover_soccer_markets(
    client: PolymarketClient,
    min_volume: float = 10000.0,
) -> list[SoccerMarketCandidate]:
    """
    Pull active Gamma markets and keep plausible 90-minute soccer match markets.
    """
    markets = await client.list_markets({"active": True, "closed": "false"})
    candidates: list[SoccerMarketCandidate] = []

    for m in markets:
        if is_excluded_market(m.question, m.description):
            continue

        # Soft soccer filter via tags/category/question keywords
        blob = f"{m.question} {m.category} {' '.join(m.tags)}".lower()
        soccer_hints = (
            "soccer", "football", "premier league", "la liga", "serie a",
            "bundesliga", "ligue 1", "mls", "champions league", "world cup",
            " vs ", " vs. ", " v ",
        )
        looks_soccer = any(h in blob for h in soccer_hints) or bool(
            extract_teams_from_question(m.question)
        ) or bool(extract_single_team_win(m.question))
        if not looks_soccer:
            continue

        if not m.yes_token_id or not m.no_token_id:
            continue

        volume = max(m.volume_24h, m.liquidity)
        if volume < min_volume:
            continue

        teams = extract_teams_from_question(m.question)
        single = extract_single_team_win(m.question)
        if not teams and not single:
            continue

        # Only fetch outcome prices for markets we can actually map
        yes_p, no_p = 0.0, 0.0
        try:
            raw = await client._request(
                "GET", f"/markets/{m.market_id}", base_url=client.gamma_url
            )
            if isinstance(raw, dict):
                yes_p, no_p = _parse_outcome_prices(m, raw)
        except Exception:
            pass

        if teams:
            candidates.append(
                SoccerMarketCandidate(
                    market=m,
                    team_a=teams[0],
                    team_b=teams[1],
                    yes_implied=yes_p,
                    no_implied=no_p,
                    volume=volume,
                    is_single_team_win=False,
                )
            )
        elif single:
            candidates.append(
                SoccerMarketCandidate(
                    market=m,
                    team_a=single,
                    team_b=None,
                    yes_implied=yes_p,
                    no_implied=no_p,
                    volume=volume,
                    is_single_team_win=True,
                )
            )

    logger.info(f"Discovered {len(candidates)} soccer market candidates (vol>={min_volume})")
    return candidates


@dataclass
class MappedFavoriteMarket:
    market_id: str
    question: str
    token_id_favorite_yes: str
    token_id_favorite_no: str
    favorite_team: str
    underdog_team: str
    favorite_is_home: bool
    pre_match_favorite_implied_prob: float
    volume: float
    match_score: float


def map_fixture_to_market(
    home_team: str,
    away_team: str,
    candidates: list[SoccerMarketCandidate],
    min_favorite_prob: float = 0.55,
    min_match_score: float = 0.72,
) -> Optional[MappedFavoriteMarket]:
    """
    Link a sports fixture to a Polymarket favorite-win binary market.
    Requires a clear favorite with implied prob > min_favorite_prob.
    """
    best: Optional[MappedFavoriteMarket] = None
    best_score = 0.0

    for c in candidates:
        mapped: Optional[MappedFavoriteMarket] = None

        if c.is_single_team_win:
            sh = team_similarity(home_team, c.team_a)
            sa = team_similarity(away_team, c.team_a)
            if max(sh, sa) < min_match_score:
                continue
            subject_is_home = sh >= sa
            fav_prob = c.yes_implied
            if fav_prob <= 0 or fav_prob < min_favorite_prob:
                continue
            underdog = away_team if subject_is_home else home_team
            mapped = MappedFavoriteMarket(
                market_id=c.market.market_id,
                question=c.market.question,
                token_id_favorite_yes=c.market.yes_token_id,
                token_id_favorite_no=c.market.no_token_id,
                favorite_team=c.team_a,
                underdog_team=underdog,
                favorite_is_home=subject_is_home,
                pre_match_favorite_implied_prob=fav_prob,
                volume=c.volume,
                match_score=max(sh, sa),
            )
        else:
            if c.team_b is None:
                continue
            pair = match_team_pair(
                home_team, away_team, c.team_a, c.team_b, min_score=min_match_score
            )
            if not pair:
                continue

            if c.yes_implied <= 0 and c.no_implied <= 0:
                continue

            if c.yes_implied >= c.no_implied:
                fav_name = c.team_a
                fav_prob = c.yes_implied
                tok_yes = c.market.yes_token_id
                tok_no = c.market.no_token_id
            else:
                fav_name = c.team_b
                fav_prob = c.no_implied
                tok_yes = c.market.no_token_id
                tok_no = c.market.yes_token_id

            if fav_prob < min_favorite_prob:
                continue

            fav_is_home = team_similarity(home_team, fav_name) >= team_similarity(
                away_team, fav_name
            )
            underdog = away_team if fav_is_home else home_team
            mapped = MappedFavoriteMarket(
                market_id=c.market.market_id,
                question=c.market.question,
                token_id_favorite_yes=tok_yes,
                token_id_favorite_no=tok_no,
                favorite_team=fav_name,
                underdog_team=underdog,
                favorite_is_home=fav_is_home,
                pre_match_favorite_implied_prob=fav_prob,
                volume=c.volume,
                match_score=pair.overall,
            )

        if mapped is not None and mapped.match_score > best_score:
            best_score = mapped.match_score
            best = mapped

    return best
