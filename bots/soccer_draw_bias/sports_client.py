"""
API-Football client for fixtures, live scores, and match events.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

from bots.soccer_draw_bias.models import LiveFixtureSnapshot

logger = logging.getLogger(__name__)

BASE_URL = "https://v3.football.api-sports.io"

# Finished / non-live statuses where we should settle or stop monitoring
FT_STATUSES = {"FT", "AET", "PEN", "AWD", "WO", "ABD", "CANC", "PST"}
# Still playing — but ET / pens are NOT a Polymarket Match Winner window
LIVE_STATUSES = {"1H", "HT", "2H", "ET", "BT", "P", "LIVE", "INT", "SUSP"}
REGULATION_LIVE = {"1H", "HT", "2H", "LIVE", "INT", "SUSP"}
POST_REGULATION = {"ET", "BT", "P", "AET", "PEN"}
NS_STATUSES = {"NS", "TBD", "NST"}

# Leagues that use calendar-year seasons (not Aug→May European seasons)
CALENDAR_YEAR_LEAGUES = {253}  # MLS


def infer_season(league_id: int, on_date: Optional[datetime] = None) -> int:
    """
    Infer API-Football season year for a league on a given date.

    European competitions: season label is the year the season starts (Aug).
    MLS / similar: calendar year.
    """
    dt = on_date or datetime.utcnow()
    if league_id in CALENDAR_YEAR_LEAGUES:
        return dt.year
    # July onward → new season year
    if dt.month >= 7:
        return dt.year
    return dt.year - 1


class SportsApiError(Exception):
    """Raised when API-Football returns a non-success response."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


class ApiFootballClient:
    """Thin async wrapper around API-Football v3."""

    def __init__(
        self,
        api_key: str,
        timeout: float = 30.0,
        base_url: str = BASE_URL,
    ):
        if not api_key:
            raise ValueError("API_FOOTBALL_KEY is required")
        self.api_key = api_key
        self.timeout = timeout
        self.base_url = base_url.rstrip("/")
        self._client: Optional[httpx.AsyncClient] = None
        self._consecutive_failures = 0
        self._last_success_at: Optional[datetime] = None
        self._failure_started_at: Optional[datetime] = None

    async def connect(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={"x-apisports-key": self.api_key},
            timeout=self.timeout,
        )

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    @property
    def trading_suspended(self) -> bool:
        """True if sports API has failed for more than 3 consecutive minutes."""
        if self._failure_started_at is None:
            return False
        elapsed = (datetime.utcnow() - self._failure_started_at).total_seconds()
        return elapsed > 180

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    async def _get(self, path: str, params: Optional[dict] = None) -> dict[str, Any]:
        if self._client is None:
            await self.connect()
        assert self._client is not None
        try:
            resp = await self._client.get(path, params=params or {})
            if resp.status_code != 200:
                self._record_failure()
                raise SportsApiError(
                    f"API-Football {path} returned {resp.status_code}",
                    status_code=resp.status_code,
                )
            data = resp.json()
            errors = data.get("errors")
            if errors:
                # API-Football sometimes returns 200 with errors object
                self._record_failure()
                raise SportsApiError(f"API-Football errors: {errors}")
            self._record_success()
            return data
        except SportsApiError:
            raise
        except Exception as e:
            self._record_failure()
            raise SportsApiError(str(e)) from e

    def _record_success(self) -> None:
        self._consecutive_failures = 0
        self._failure_started_at = None
        self._last_success_at = datetime.utcnow()

    def _record_failure(self) -> None:
        self._consecutive_failures += 1
        if self._failure_started_at is None:
            self._failure_started_at = datetime.utcnow()

    async def get_fixtures_by_date(
        self,
        date_str: str,
        league_ids: list[int],
        season: Optional[int] = None,
    ) -> list[LiveFixtureSnapshot]:
        """
        Fetch day's fixtures for configured leagues.

        Prefer a single date-only call (works on API-Football free tier), then
        filter client-side by league_ids. Fall back to per-league+season calls
        when the date-only path returns nothing useful (paid plans).
        """
        league_set = set(league_ids or [])
        snapshots: list[LiveFixtureSnapshot] = []
        seen: set[int] = set()

        # --- Primary path: /fixtures?date=YYYY-MM-DD (no season required) ---
        date_ok = False
        try:
            data = await self._get("/fixtures", params={"date": date_str})
            date_ok = True
            for item in data.get("response", []) or []:
                snap = self._parse_fixture(item)
                if not snap:
                    continue
                if league_set and snap.league_id not in league_set:
                    continue
                if snap.match_id in seen:
                    continue
                seen.add(snap.match_id)
                snapshots.append(snap)
            logger.info(
                f"Date fixtures {date_str}: {len(data.get('response') or [])} raw, "
                f"{len(snapshots)} after league filter"
            )
        except SportsApiError as e:
            logger.warning(f"Date fixtures fetch failed date={date_str}: {e}")

        # Date-only worked (even if 0 target-league games) — skip paid-plan fallback
        # so we don't burn free-tier quota on blocked season queries.
        if date_ok:
            return snapshots

        # --- Fallback: per-league + season (needed on some paid plans / older dates) ---
        try:
            on_date = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            on_date = datetime.utcnow()

        for league_id in league_ids:
            season_year = season if season is not None else infer_season(league_id, on_date)
            params: dict[str, Any] = {
                "date": date_str,
                "league": league_id,
                "season": season_year,
            }
            try:
                data = await self._get("/fixtures", params)
            except SportsApiError as e:
                logger.warning(
                    f"Fixtures fetch failed league={league_id} season={season_year}: {e}"
                )
                continue
            for item in data.get("response", []) or []:
                snap = self._parse_fixture(item)
                if snap and snap.match_id not in seen:
                    seen.add(snap.match_id)
                    snapshots.append(snap)
        return snapshots

    async def get_live_fixtures(
        self,
        league_ids: Optional[list[int]] = None,
    ) -> list[LiveFixtureSnapshot]:
        """Fetch currently live fixtures (optionally scoped to leagues)."""
        if league_ids:
            live_param = "-".join(str(i) for i in league_ids)
        else:
            live_param = "all"
        data = await self._get("/fixtures", params={"live": live_param})
        snapshots = []
        for item in data.get("response", []) or []:
            snap = self._parse_fixture(item)
            if snap:
                snapshots.append(snap)
        return snapshots

    async def get_fixture(self, fixture_id: int) -> Optional[LiveFixtureSnapshot]:
        data = await self._get("/fixtures", params={"id": fixture_id})
        items = data.get("response", []) or []
        if not items:
            return None
        return self._parse_fixture(items[0])

    async def get_fixture_events(self, fixture_id: int) -> list[dict]:
        data = await self._get("/fixtures/events", params={"fixture": fixture_id})
        return data.get("response", []) or []

    async def enrich_with_events(
        self,
        snapshot: LiveFixtureSnapshot,
        favorite_team: str,
    ) -> LiveFixtureSnapshot:
        """Attach red-card count for favorite and VAR-in-progress flag."""
        try:
            events = await self.get_fixture_events(snapshot.match_id)
        except SportsApiError as e:
            logger.warning(f"Events fetch failed match={snapshot.match_id}: {e}")
            return snapshot

        favorite_reds = 0
        var_in_progress = False
        fav_norm = _norm(favorite_team)

        for ev in events:
            team_name = ((ev.get("team") or {}).get("name")) or ""
            ev_type = (ev.get("type") or "").lower()
            detail = (ev.get("detail") or "").lower()

            if ev_type == "card" and _norm(team_name) == fav_norm:
                if "red" in detail:
                    favorite_reds += 1

            # VAR reviews appear as type "Var"
            if ev_type == "var":
                # Treat recent/open VAR as blocking if comments suggest review
                comments = (ev.get("comments") or detail or "").lower()
                if any(
                    k in comments
                    for k in ("goal", "review", "cancelled", "disallowed", "awarded")
                ):
                    # If this is the last event and still under review-ish, pause.
                    # API-Football typically appends resolved VAR outcomes; block only
                    # when detail indicates an ongoing decision without resolution.
                    if "goal cancelled" not in detail and "goal confirmed" not in detail:
                        if "under review" in comments or detail in ("", "var"):
                            var_in_progress = True

        # Heuristic: fixture status INT sometimes used during stoppages
        if snapshot.status_short == "INT":
            var_in_progress = True

        snapshot.favorite_red_cards = favorite_reds
        snapshot.var_in_progress = var_in_progress
        return snapshot

    def _parse_fixture(self, item: dict) -> Optional[LiveFixtureSnapshot]:
        try:
            fixture = item.get("fixture") or {}
            teams = item.get("teams") or {}
            goals = item.get("goals") or {}
            league = item.get("league") or {}
            status = (fixture.get("status") or {})

            match_id = int(fixture.get("id"))
            home = (teams.get("home") or {}).get("name") or ""
            away = (teams.get("away") or {}).get("name") or ""
            if not home or not away:
                return None

            home_goals = int(goals.get("home") if goals.get("home") is not None else 0)
            away_goals = int(goals.get("away") if goals.get("away") is not None else 0)
            elapsed = int(status.get("elapsed") or 0)
            status_short = str(status.get("short") or "")

            # Polymarket Match Winner = 90' + stoppage only
            score = item.get("score") or {}
            fulltime = score.get("fulltime") or {}
            reg_home = fulltime.get("home")
            reg_away = fulltime.get("away")
            regulation_home = int(reg_home) if reg_home is not None else None
            regulation_away = int(reg_away) if reg_away is not None else None

            kickoff = None
            date_str = fixture.get("date")
            if date_str:
                try:
                    kickoff = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                    if kickoff.tzinfo is not None:
                        kickoff = kickoff.astimezone(timezone.utc).replace(tzinfo=None)
                except ValueError:
                    kickoff = None

            return LiveFixtureSnapshot(
                match_id=match_id,
                home_team=home,
                away_team=away,
                home_goals=home_goals,
                away_goals=away_goals,
                elapsed=elapsed,
                status_short=status_short,
                league_id=int(league.get("id") or 0),
                league_name=str(league.get("name") or ""),
                kickoff_utc=kickoff,
                regulation_home=regulation_home,
                regulation_away=regulation_away,
            )
        except Exception as e:
            logger.warning(f"Failed to parse fixture: {e}")
            return None


def _norm(name: str) -> str:
    return " ".join(name.lower().replace("-", " ").split())


def is_full_time(status_short: str) -> bool:
    return status_short.upper() in FT_STATUSES


def is_live(status_short: str) -> bool:
    return status_short.upper() in LIVE_STATUSES


def is_not_started(status_short: str) -> bool:
    return status_short.upper() in NS_STATUSES


def is_post_regulation(status_short: str) -> bool:
    """True once the match has left 90'+stoppage (ET / pens / finished after ET)."""
    return status_short.upper() in POST_REGULATION


def is_regulation_playing(status_short: str) -> bool:
    return status_short.upper() in REGULATION_LIVE
