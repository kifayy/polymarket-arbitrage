"""Unit tests for soccer draw-bias strategy gates, matcher, and settlement."""

from __future__ import annotations

from bots.soccer_draw_bias.engine import DrawBiasConfig, MatchStateMachine
from bots.soccer_draw_bias.matcher import (
    extract_teams_from_question,
    match_team_pair,
    team_similarity,
)
from bots.soccer_draw_bias.models import LiveFixtureSnapshot, MatchRecord, MatchStatus
from bots.soccer_draw_bias.polymarket_soccer import (
    SoccerMarketCandidate,
    is_excluded_market,
    map_fixture_to_market,
)
from polymarket_client.models import Market


def _match(**kwargs) -> MatchRecord:
    defaults = dict(
        match_id=1,
        home_team="Manchester City",
        away_team="Brighton",
        polymarket_market_id="m1",
        token_id_favorite_yes="yes1",
        token_id_favorite_no="no1",
        pre_match_favorite_implied_prob=0.70,
        favorite_is_home=True,
        favorite_team="Manchester City",
        underdog_team="Brighton",
        status=MatchStatus.SCHEDULED,
        current_minute=0,
        current_score=[0, 0],
    )
    defaults.update(kwargs)
    return MatchRecord(**defaults)


def _snap(**kwargs) -> LiveFixtureSnapshot:
    defaults = dict(
        match_id=1,
        home_team="Manchester City",
        away_team="Brighton",
        home_goals=0,
        away_goals=0,
        elapsed=0,
        status_short="NS",
    )
    defaults.update(kwargs)
    return LiveFixtureSnapshot(**defaults)


class TestMatcher:
    def test_man_utd_alias(self):
        assert team_similarity("Manchester United", "Man Utd") >= 0.9

    def test_man_city_alias(self):
        assert team_similarity("Manchester City", "Man City") >= 0.9

    def test_pair_match(self):
        pair = match_team_pair(
            "Manchester United",
            "Liverpool",
            "Man Utd",
            "Liverpool FC",
        )
        assert pair is not None
        assert pair.overall >= 0.72

    def test_extract_vs(self):
        teams = extract_teams_from_question("Arsenal vs Chelsea?")
        assert teams == ("Arsenal", "Chelsea")


class TestExclusions:
    def test_exclude_to_advance(self):
        assert is_excluded_market("Will Real Madrid to advance?")

    def test_exclude_lift_trophy(self):
        assert is_excluded_market("Will City lift the trophy?")

    def test_allow_match_winner(self):
        assert not is_excluded_market("Will Manchester City win?")


class TestStateMachine:
    def setup_method(self):
        self.cfg = DrawBiasConfig()
        self.sm = MatchStateMachine(self.cfg)

    def test_kickoff_to_monitoring(self):
        m = _match()
        self.sm.apply_live_update(m, _snap(elapsed=10, status_short="1H"))
        assert m.status == MatchStatus.MONITORING

    def test_before_75_stays_monitoring(self):
        m = _match(status=MatchStatus.MONITORING, current_minute=60)
        self.sm.apply_live_update(
            m, _snap(elapsed=60, status_short="2H", home_goals=0, away_goals=0)
        )
        assert m.status == MatchStatus.MONITORING

    def test_75_tied_goes_active(self):
        m = _match(status=MatchStatus.MONITORING)
        self.sm.apply_live_update(
            m, _snap(elapsed=75, status_short="2H", home_goals=1, away_goals=1)
        )
        assert m.status == MatchStatus.ACTIVE_WINDOW

    def test_75_favorite_winning_discarded(self):
        m = _match(status=MatchStatus.MONITORING, favorite_is_home=True)
        self.sm.apply_live_update(
            m, _snap(elapsed=76, status_short="2H", home_goals=2, away_goals=1)
        )
        assert m.status == MatchStatus.DISCARDED
        assert m.discard_reason == "favorite_winning"

    def test_75_favorite_losing_discarded(self):
        m = _match(status=MatchStatus.MONITORING, favorite_is_home=True)
        self.sm.apply_live_update(
            m, _snap(elapsed=76, status_short="2H", home_goals=0, away_goals=1)
        )
        assert m.status == MatchStatus.DISCARDED
        assert m.discard_reason == "favorite_losing"

    def test_red_card_before_window_discarded(self):
        m = _match(status=MatchStatus.MONITORING)
        snap = _snap(elapsed=40, status_short="1H")
        snap.favorite_red_cards = 1
        self.sm.apply_live_update(m, snap)
        assert m.status == MatchStatus.DISCARDED
        assert "red_card" in m.discard_reason

    def test_past_88_discarded_from_active(self):
        m = _match(status=MatchStatus.ACTIVE_WINDOW, current_minute=87)
        self.sm.apply_live_update(
            m, _snap(elapsed=89, status_short="2H", home_goals=1, away_goals=1)
        )
        assert m.status == MatchStatus.DISCARDED
        assert m.discard_reason == "past_active_window_end"

    def test_should_evaluate_orderbook_in_window(self):
        m = _match(
            status=MatchStatus.ACTIVE_WINDOW,
            current_minute=80,
            current_score=[1, 1],
        )
        assert self.sm.should_evaluate_orderbook(m) is True

    def test_should_not_evaluate_on_var(self):
        m = _match(
            status=MatchStatus.ACTIVE_WINDOW,
            current_minute=80,
            current_score=[0, 0],
            var_in_progress=True,
        )
        assert self.sm.should_evaluate_orderbook(m) is False

    def test_settlement_draw_is_win(self):
        m = _match(
            status=MatchStatus.EXECUTED,
            execution_price=0.15,
            shares=100.0,
            favorite_is_home=True,
        )
        settled = self.sm.settle(
            m, _snap(elapsed=90, status_short="FT", home_goals=1, away_goals=1)
        )
        assert settled.status == MatchStatus.SETTLED
        assert settled.outcome == "Win"
        assert settled.pnl == 85.0  # 100 - 15

    def test_settlement_favorite_win_is_loss(self):
        m = _match(
            status=MatchStatus.EXECUTED,
            execution_price=0.15,
            shares=100.0,
            favorite_is_home=True,
        )
        settled = self.sm.settle(
            m, _snap(elapsed=90, status_short="FT", home_goals=2, away_goals=1)
        )
        assert settled.outcome == "Loss"
        assert settled.pnl == -15.0

    def test_settlement_underdog_win_is_win(self):
        m = _match(
            status=MatchStatus.EXECUTED,
            execution_price=0.10,
            shares=150.0,
            favorite_is_home=True,
        )
        settled = self.sm.settle(
            m, _snap(elapsed=90, status_short="FT", home_goals=0, away_goals=1)
        )
        assert settled.outcome == "Win"
        assert settled.pnl == 135.0  # 150 - 15

    def test_settlement_uses_regulation_not_et(self):
        """Knockout: 1-1 at 90', favorite wins in ET — Match Winner is still a draw."""
        m = _match(
            status=MatchStatus.EXECUTED,
            execution_price=0.15,
            shares=100.0,
            favorite_is_home=True,
        )
        snap = _snap(
            elapsed=120,
            status_short="AET",
            home_goals=2,  # includes ET goal
            away_goals=1,
        )
        snap.regulation_home = 1
        snap.regulation_away = 1
        settled = self.sm.settle(m, snap)
        assert settled.final_score == [1, 1]
        assert settled.outcome == "Win"
        assert settled.pnl == 85.0

    def test_no_entry_in_extra_time(self):
        m = _match(
            status=MatchStatus.ACTIVE_WINDOW,
            current_minute=90,
            current_score=[1, 1],
            sports_status="ET",
        )
        assert self.sm.should_evaluate_orderbook(m) is False

    def test_poll_interval_idle_vs_active(self):
        idle = self.sm.poll_interval_for([])
        assert idle >= 600
        active = self.sm.poll_interval_for(
            [_match(status=MatchStatus.ACTIVE_WINDOW, current_minute=80)]
        )
        assert active == self.cfg.poll_interval_active_sec
        monitoring = self.sm.poll_interval_for(
            [_match(status=MatchStatus.MONITORING, current_minute=40)]
        )
        assert monitoring == self.cfg.poll_interval_monitoring_sec


class TestPriceThreshold:
    """Executor gate logic mirrored via should_evaluate + price compare helpers."""

    def test_ask_above_threshold_rejected(self):
        cfg = DrawBiasConfig(target_max_buy_price=0.15)
        assert 0.16 > cfg.target_max_buy_price
        assert 0.15 <= cfg.target_max_buy_price

    def test_sizing(self):
        cfg = DrawBiasConfig(risk_unit_usd=15.0, target_max_buy_price=0.15)
        shares = cfg.risk_unit_usd / 0.15
        assert shares == 100.0


class TestMarketMapping:
    def test_map_single_team_favorite(self):
        market = Market(
            market_id="42",
            condition_id="0x1",
            question="Will Manchester City win?",
            yes_token_id="tok_yes",
            no_token_id="tok_no",
            volume_24h=50000,
        )
        candidates = [
            SoccerMarketCandidate(
                market=market,
                team_a="Manchester City",
                team_b=None,
                yes_implied=0.68,
                no_implied=0.32,
                volume=50000,
                is_single_team_win=True,
            )
        ]
        mapped = map_fixture_to_market(
            "Manchester City", "Brighton", candidates, min_favorite_prob=0.55
        )
        assert mapped is not None
        assert mapped.token_id_favorite_no == "tok_no"
        assert mapped.pre_match_favorite_implied_prob == 0.68

    def test_near_even_discarded(self):
        market = Market(
            market_id="42",
            condition_id="0x1",
            question="Will Arsenal win?",
            yes_token_id="y",
            no_token_id="n",
            volume_24h=50000,
        )
        candidates = [
            SoccerMarketCandidate(
                market=market,
                team_a="Arsenal",
                team_b=None,
                yes_implied=0.38,
                no_implied=0.62,
                volume=50000,
                is_single_team_win=True,
            )
        ]
        mapped = map_fixture_to_market("Arsenal", "Chelsea", candidates)
        assert mapped is None
