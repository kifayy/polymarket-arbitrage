"""
Fuzzy team-name matching between API-Football and Polymarket.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Optional


# Common abbreviations / aliases for soccer clubs
TEAM_ALIASES: dict[str, list[str]] = {
    "manchester united": ["man utd", "man united", "manchester utd", "mufc"],
    "manchester city": ["man city", "manchester city fc", "mcfc"],
    "tottenham hotspur": ["tottenham", "spurs", "tottenham hotspur fc"],
    "newcastle united": ["newcastle", "newcastle utd"],
    "wolverhampton wanderers": ["wolves", "wolverhampton"],
    "brighton and hove albion": ["brighton", "brighton & hove albion"],
    "nottingham forest": ["nott'm forest", "nottingham", "forest"],
    "west ham united": ["west ham", "westham"],
    "leicester city": ["leicester"],
    "atletico madrid": ["atlético madrid", "atletico", "atleti"],
    "real madrid": ["real madrid cf", "madrid"],
    "barcelona": ["fc barcelona", "barca", "barça"],
    "bayern munich": ["bayern", "bayern munchen", "fc bayern"],
    "borussia dortmund": ["dortmund", "bvb"],
    "inter milan": ["inter", "fc internazionale", "internazionale"],
    "ac milan": ["milan"],
    "psg": ["paris saint germain", "paris sg", "paris saint-germain"],
    "paris saint germain": ["psg", "paris sg"],
    "sporting cp": ["sporting lisbon", "sporting"],
    "ajax": ["afc ajax", "ajax amsterdam"],
    "la galaxy": ["los angeles galaxy", "galaxy"],
    "inter miami": ["inter miami cf"],
    "new york city": ["nycfc", "new york city fc"],
}


def normalize_team_name(name: str) -> str:
    """Lowercase, strip punctuation/noise words for comparison."""
    s = name.lower().strip()
    s = s.replace("&", " and ")
    s = re.sub(r"[^\w\s]", " ", s)
    noise = {
        "fc", "cf", "afc", "sc", "ac", "fk", "club", "the", "de", "united",
    }
    # Keep "united" for Man United differentiation — only drop trailing FC/CF
    tokens = [t for t in s.split() if t not in {"fc", "cf", "afc", "sc"}]
    return " ".join(tokens)


def _alias_set(name: str) -> set[str]:
    n = normalize_team_name(name)
    aliases = {n, name.lower().strip()}
    for canonical, alts in TEAM_ALIASES.items():
        pool = {canonical, *alts}
        pool_norm = {normalize_team_name(x) for x in pool} | pool
        if n in pool_norm or normalize_team_name(name) in pool_norm:
            aliases |= pool_norm
    return {normalize_team_name(a) for a in aliases}


def team_similarity(a: str, b: str) -> float:
    """
    Similarity score in [0, 1] using aliases + SequenceMatcher (Levenshtein-like).
    """
    na, nb = normalize_team_name(a), normalize_team_name(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0

    aliases_a = _alias_set(a)
    aliases_b = _alias_set(b)
    if aliases_a & aliases_b:
        return 1.0

    # Token containment (e.g. "Man City" vs "Manchester City")
    ta, tb = set(na.split()), set(nb.split())
    if ta and tb and (ta <= tb or tb <= ta):
        return 0.92

    best = SequenceMatcher(None, na, nb).ratio()
    for xa in aliases_a:
        for xb in aliases_b:
            best = max(best, SequenceMatcher(None, xa, xb).ratio())
    return best


@dataclass
class TeamPairMatch:
    home_score: float
    away_score: float
    overall: float
    swapped: bool = False  # True if polymarket home/away appear reversed


def match_team_pair(
    sports_home: str,
    sports_away: str,
    poly_team_a: str,
    poly_team_b: str,
    min_score: float = 0.72,
) -> Optional[TeamPairMatch]:
    """
    Pair sports home/away with two Polymarket team strings.
    Returns None if either side is below min_score.
    """
    # Orientation 1: a=home, b=away
    h1 = team_similarity(sports_home, poly_team_a)
    a1 = team_similarity(sports_away, poly_team_b)
    o1 = min(h1, a1)

    # Orientation 2: swapped
    h2 = team_similarity(sports_home, poly_team_b)
    a2 = team_similarity(sports_away, poly_team_a)
    o2 = min(h2, a2)

    if o1 >= o2 and o1 >= min_score:
        return TeamPairMatch(home_score=h1, away_score=a1, overall=o1, swapped=False)
    if o2 >= min_score:
        return TeamPairMatch(home_score=h2, away_score=a2, overall=o2, swapped=True)
    return None


def extract_teams_from_question(question: str) -> Optional[tuple[str, str]]:
    """
    Extract two team names from common Polymarket soccer question formats.
    """
    q = question.strip()
    patterns = [
        r"^(.+?)\s+vs\.?\s+(.+?)(?:\s*[?\-–:].*)?$",
        r"^(.+?)\s+v\s+(.+?)(?:\s*[?\-–:].*)?$",
        r"will\s+(.+?)\s+beat\s+(.+?)(?:\s*\?)?$",
        r"will\s+(.+?)\s+win\s+against\s+(.+?)(?:\s*\?)?$",
    ]
    for pat in patterns:
        m = re.match(pat, q, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip(), m.group(2).strip()

    # "Will Manchester City win on ..." single-team — caller handles separately
    return None


def extract_single_team_win(question: str) -> Optional[str]:
    """Extract team from 'Will X win?' style markets."""
    m = re.match(
        r"^will\s+(.+?)\s+win(?:\s+(?:the\s+match|today|on\s+.+))?\s*\??$",
        question.strip(),
        flags=re.IGNORECASE,
    )
    if m:
        team = m.group(1).strip()
        # Exclude non-team phrasings
        lower = team.lower()
        if any(x in lower for x in ("draw", "anyone", "either", "both")):
            return None
        return team
    return None
