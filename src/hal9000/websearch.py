"""Fast, zero-API-key live web search & Wikipedia grounding for HAL 9000."""

from __future__ import annotations

import html
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from urllib.parse import quote_plus

import httpx

logger = logging.getLogger(__name__)

_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux aarch64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

_EXPLICIT_SEARCH_RE = re.compile(
    r"\b(?:"
    r"search\s+(?:online|the\s+web|the\s+internet|google|for|wikipedia|to\s+see)"
    r"|look\s+up|google\s+(?:it|this|for)?|check\s+(?:online|the\s+web|the\s+internet|if)"
    r"|find\s+out\s+(?:if|whether|who|what|when|how)"
    r"|on\s+the\s+(?:web|internet)"
    r"|latest\s+news|current\s+(?:price|status|president|ceo|score|weather|news|year|date|record)"
    r"|today(?:'s)?\s+(?:news|weather|price|date|score)"
    r"|right\s+now|recently|this\s+(?:week|month|year)"
    r"|price\s+of|market\s+cap|weather\s+in|who\s+won|score\s+of"
    r"|has\s+.+\s+been\s+(?:solved|proven|disproven|released|launched|discovered|elected)"
    r"|202[4-9]|203\d"
    r")\b",
    re.IGNORECASE,
)

_OPEN_FACT_RE = re.compile(
    r"\b(?:"
    r"riemann|goldbach|collatz|navier[\s-]stokes|p\s*(?:vs|versus|=)\s*np|millennium\s+prize"
    r"|theorem|conjecture|hypothesis|nobel\s+prize|fields\s+medal"
    r"|capital\s+of|population\s+of|president\s+of|prime\s+minister|ceo\s+of|founder\s+of"
    r"|cardano|bitcoin|ethereum"
    r"|who\s+(?:invented|discovered|wrote|directed|created|founded|is\s+the\s+(?:current|president|ceo|prime))"
    r"|when\s+(?:did|was)\s+(?!you\b|hal\b|discovery\b|dave\b|frank\b|bowman\b|poole\b|chandra\b)"
    r")\b",
    re.IGNORECASE,
)

_LORE_OR_SMALLTALK_RE = re.compile(
    r"\b(?:"
    r"pod\s*bay\s*doors?|discovery\s*one|monolith|tma[\s-]*1|ae[\s-]*35|"
    r"bowman|frank\s+poole|chandra|leonov|sal[\s-]*9000|daisy\s+bell|urbana|langley|"
    r"halman|pico\s+vault|europa|lucifer|kubrick|2001|2010|2061|3001"
    r")\b",
    re.IGNORECASE,
)

_QUERY_STRIP_RE = re.compile(
    r"^(?:hey\s+|okay\s+|ok\s+|please\s+)?"
    r"(?:hal(?:\s+9000|\s+nine\s+thousand)?[\s,.:!?-]+)?"
    r"(?:can\s+you\s+|could\s+you\s+|would\s+you\s+|please\s+)?"
    r"(?:search(?:\s+online|\s+the\s+web|\s+the\s+internet)?\s+(?:for|about|to\s+see\s+if|to\s+see\s+whether|to\s+find\s+out)?|"
    r"look\s+up\s+|google\s+|check(?:\s+online|\s+the\s+web)?\s+(?:if|whether|for|about)?|"
    r"find\s+out\s+(?:if|whether)?|"
    r"tell\s+me\s+(?:about|if|whether)?)\s*",
    re.IGNORECASE,
)


def should_search_web(user_text: str) -> bool:
    """Fast heuristic deciding whether a user turn benefits from live web/Wikipedia search."""
    text = user_text.strip()
    if len(text) < 8:
        return False
    if _EXPLICIT_SEARCH_RE.search(text):
        return True
    if _LORE_OR_SMALLTALK_RE.search(text):
        return False
    return bool(_OPEN_FACT_RE.search(text))


def _clean_search_query(user_text: str) -> str:
    q = _QUERY_STRIP_RE.sub("", user_text.strip())
    q = re.sub(r"[?!.,;:\"'()]+", " ", q)
    q = re.sub(r"\b(?:please|hal|9000)\b", " ", q, flags=re.IGNORECASE)
    q = re.sub(r"\s+", " ", q).strip()
    return q[:160] if q else user_text.strip()[:160]


class _DDGLiteParser(HTMLParser):
    """Minimal zero-dependency parser extracting titles and snippets from DuckDuckGo Lite HTML."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.titles: list[str] = []
        self.snippets: list[str] = []
        self._in_title = False
        self._in_snippet = False
        self._buf: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = dict(attrs)
        cls = attr_map.get("class") or ""
        if tag == "a" and "result-link" in cls:
            self._in_title = True
            self._buf = []
        elif tag == "td" and "result-snippet" in cls:
            self._in_snippet = True
            self._buf = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._in_title:
            self._in_title = False
            txt = re.sub(r"\s+", " ", "".join(self._buf)).strip()
            if txt:
                self.titles.append(txt)
            self._buf = []
        elif tag == "td" and self._in_snippet:
            self._in_snippet = False
            txt = re.sub(r"\s+", " ", "".join(self._buf)).strip()
            if txt:
                self.snippets.append(txt)
            self._buf = []

    def handle_data(self, data: str) -> None:
        if self._in_title or self._in_snippet:
            self._buf.append(data)


def _fetch_ddg_lite(query: str, timeout_s: float) -> list[str]:
    try:
        url = f"https://lite.duckduckgo.com/lite/?q={quote_plus(query)}"
        with httpx.Client(
            timeout=httpx.Timeout(connect=2.0, read=timeout_s, write=2.0, pool=2.0),
            follow_redirects=True,
            headers={"User-Agent": _USER_AGENT, "Accept-Language": "en-US,en;q=0.9"},
        ) as client:
            resp = client.get(url)
            if resp.status_code != 200:
                return []
            parser = _DDGLiteParser()
            parser.feed(resp.text)
            results: list[str] = []
            for idx, snippet in enumerate(parser.snippets[:4]):
                title = parser.titles[idx] if idx < len(parser.titles) else ""
                line = f"{title}: {snippet}" if title else snippet
                results.append(html.unescape(line)[:280])
            return results
    except (httpx.HTTPError, OSError, ValueError) as exc:
        logger.debug("DuckDuckGo Lite search skipped: %s", exc)
        return []


def _fetch_wikipedia(query: str, timeout_s: float) -> list[str]:
    try:
        with httpx.Client(
            timeout=httpx.Timeout(connect=2.0, read=timeout_s, write=2.0, pool=2.0),
            follow_redirects=True,
            headers={"User-Agent": "IOGHAL-Voice-Runtime/1.0 (local research client)"},
        ) as client:
            resp = client.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": query,
                    "srlimit": 2,
                    "format": "json",
                    "utf8": 1,
                },
            )
            if resp.status_code != 200:
                return []
            hits = resp.json().get("query", {}).get("search", [])
            if not hits:
                return []
            top_title = hits[0].get("title", "")
            results: list[str] = []
            if top_title:
                sum_resp = client.get(
                    f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote_plus(top_title.replace(' ', '_'))}"
                )
                if sum_resp.status_code == 200:
                    extract = (sum_resp.json().get("extract") or "").strip()
                    if extract:
                        results.append(f"Wikipedia ({top_title}): {extract[:520]}")
            for h in hits:
                title = h.get("title", "")
                snip = re.sub(r"<[^>]+>", "", h.get("snippet", ""))
                snip = html.unescape(re.sub(r"\s+", " ", snip)).strip()
                if title and snip and (not results or title != top_title):
                    results.append(f"Wikipedia ({title}): {snip[:240]}")
            return results[:2]
    except (httpx.HTTPError, OSError, ValueError, KeyError, TypeError) as exc:
        logger.debug("Wikipedia search skipped: %s", exc)
        return []


def search_web_context(
    user_text: str,
    *,
    cancel_event: threading.Event | None = None,
    timeout_s: float = 3.5,
) -> str:
    """Query Wikipedia and DuckDuckGo Lite in parallel and return concise grounded snippets."""
    if cancel_event is not None and cancel_event.is_set():
        return ""
    query = _clean_search_query(user_text)
    if not query:
        return ""

    wiki_res: list[str] = []
    ddg_res: list[str] = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        f_wiki = pool.submit(_fetch_wikipedia, query, timeout_s)
        f_ddg = pool.submit(_fetch_ddg_lite, query, timeout_s)
        try:
            wiki_res = f_wiki.result(timeout=timeout_s + 0.5)
        except (TimeoutError, OSError, RuntimeError):
            wiki_res = []
        if cancel_event is not None and cancel_event.is_set():
            return ""
        try:
            ddg_res = f_ddg.result(timeout=max(0.5, timeout_s))
        except (TimeoutError, OSError, RuntimeError):
            ddg_res = []

    if cancel_event is not None and cancel_event.is_set():
        return ""

    combined = [*wiki_res, *ddg_res]
    if not combined:
        return ""
    joined = "\n".join(f"- {item}" for item in combined[:5])
    return joined[:1500]
