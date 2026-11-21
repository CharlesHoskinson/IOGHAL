"""Fast, zero-API-key live web search & Wikipedia grounding for HAL 9000."""

from __future__ import annotations

import contextlib
import html
import logging
import re
import threading
import time
from html.parser import HTMLParser
from urllib.parse import quote_plus

import httpx

logger = logging.getLogger(__name__)

_CACHE_TTL_S = 300.0
_CACHE_MAX_ENTRIES = 128
_WIKI_RICH_MIN_CHARS = 120
_DDG_GRACE_AFTER_WIKI_S = 0.35
_DDG_GRACE_EVERGREEN_S = 0.05
_CANCEL_POLL_S = 0.02

_cache_lock = threading.Lock()
_query_cache: dict[str, tuple[float, str]] = {}


class _ProviderPool:
    """Thread-safe keep-alive ``httpx.Client`` pool for a single search host with instant cancel eviction."""

    def __init__(self, headers: dict[str, str]) -> None:
        self._headers = headers
        self._lock = threading.Lock()
        self._idle: list[httpx.Client] = []

    def acquire(self, *, timeout_s: float) -> httpx.Client:
        with self._lock:
            while self._idle:
                client = self._idle.pop()
                if not getattr(client, "is_closed", False):
                    return client
        return httpx.Client(
            timeout=httpx.Timeout(
                connect=min(2.0, timeout_s),
                read=timeout_s,
                write=2.0,
                pool=2.0,
            ),
            limits=httpx.Limits(
                max_keepalive_connections=4,
                max_connections=8,
                keepalive_expiry=60.0,
            ),
            follow_redirects=True,
            headers=self._headers,
        )

    def release(self, client: httpx.Client, *, healthy: bool) -> None:
        if healthy and not getattr(client, "is_closed", False):
            with self._lock:
                if len(self._idle) < 2:
                    self._idle.append(client)
                    return
        with contextlib.suppress(httpx.HTTPError, OSError, RuntimeError):
            client.close()

    def close_all(self) -> None:
        with self._lock:
            clients = list(self._idle)
            self._idle.clear()
        for client in clients:
            with contextlib.suppress(httpx.HTTPError, OSError, RuntimeError):
                client.close()


_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux aarch64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)
_WIKI_HEADERS = {"User-Agent": "IOGHAL-Voice-Runtime/1.0 (local research client)"}
_DDG_HEADERS = {"User-Agent": _USER_AGENT, "Accept-Language": "en-US,en;q=0.9"}

_wiki_pool = _ProviderPool(_WIKI_HEADERS)
_ddg_pool = _ProviderPool(_DDG_HEADERS)


def close_search_clients() -> None:
    """Close any idle keep-alive search clients."""
    _wiki_pool.close_all()
    _ddg_pool.close_all()


def warm_search_connections(timeout_s: float = 2.0) -> bool:
    """Pre-warm Wikipedia TLS keep-alive socket in the background to cut first-search latency."""
    client = _wiki_pool.acquire(timeout_s=timeout_s)
    healthy = False
    try:
        resp = client.head("https://en.wikipedia.org/w/api.php", timeout=timeout_s)
        healthy = resp.status_code < 500
        return healthy
    except (httpx.HTTPError, OSError, RuntimeError):
        return False
    finally:
        _wiki_pool.release(client, healthy=healthy)


def clear_search_cache() -> None:
    """Clear the in-memory web search TTL cache and reset pooled search clients."""
    with _cache_lock:
        _query_cache.clear()
    close_search_clients()


def _cache_get(key: str) -> str | None:
    now = time.monotonic()
    with _cache_lock:
        entry = _query_cache.get(key)
        if entry is None:
            return None
        expires_at, value = entry
        if now >= expires_at:
            _query_cache.pop(key, None)
            return None
        return value


def _cache_put(key: str, value: str, ttl_s: float = _CACHE_TTL_S) -> None:
    if not value or ttl_s <= 0:
        return
    now = time.monotonic()
    with _cache_lock:
        expired = [k for k, (exp, _) in _query_cache.items() if now >= exp]
        for k in expired:
            _query_cache.pop(k, None)
        while len(_query_cache) >= _CACHE_MAX_ENTRIES:
            oldest_key = next(iter(_query_cache))
            _query_cache.pop(oldest_key, None)
        _query_cache[key] = (now + ttl_s, value)


class _CancelGroup:
    """Tracks per-call ``httpx.Client`` instances so barge-in or early return closes sockets immediately."""

    def __init__(self, cancel_event: threading.Event | None = None) -> None:
        self._external = cancel_event
        self._aborted = threading.Event()
        self._lock = threading.Lock()
        self._clients: set[httpx.Client] = set()

    def is_cancelled(self) -> bool:
        return self._aborted.is_set() or (
            self._external is not None and self._external.is_set()
        )

    def acquire_client(
        self, pool: _ProviderPool, *, timeout_s: float
    ) -> httpx.Client | None:
        with self._lock:
            if self.is_cancelled():
                return None
        client = pool.acquire(timeout_s=timeout_s)
        with self._lock:
            if self.is_cancelled():
                pool.release(client, healthy=False)
                return None
            self._clients.add(client)
        return client

    def release_client(
        self, client: httpx.Client, *, pool: _ProviderPool, healthy: bool
    ) -> None:
        with self._lock:
            was_tracked = client in self._clients
            self._clients.discard(client)
            keep = healthy and was_tracked and not self.is_cancelled()
        pool.release(client, healthy=keep)

    def abort(self) -> None:
        self._aborted.set()
        with self._lock:
            clients = list(self._clients)
            self._clients.clear()
        for client in clients:
            with contextlib.suppress(httpx.HTTPError, OSError, RuntimeError):
                client.close()


_RECENCY_QUERY_RE = re.compile(
    r"\b(?:"
    r"latest\s+news|current\s+(?:price|status|president|ceo|score|weather|news|year|date|record)"
    r"|today(?:'s)?|tonight|right\s+now|recently|this\s+(?:week|month|year)"
    r"|price\s+of|market\s+cap|weather\s+in|who\s+won|score\s+of"
    r"|202[5-9]|203\d"
    r")\b",
    re.IGNORECASE,
)


def _is_recency_query(user_text: str) -> bool:
    """Return True when live DDG snippets are valuable even after a rich Wikipedia hit."""
    return bool(_RECENCY_QUERY_RE.search(user_text))

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


def _fetch_ddg_lite(
    query: str,
    timeout_s: float,
    cancel_group: _CancelGroup | None = None,
) -> list[str]:
    client: httpx.Client | None = None
    healthy = False
    try:
        if cancel_group is not None:
            client = cancel_group.acquire_client(_ddg_pool, timeout_s=timeout_s)
            if client is None:
                return []
        else:
            client = _ddg_pool.acquire(timeout_s=timeout_s)
        url = f"https://lite.duckduckgo.com/lite/?q={quote_plus(query)}"
        resp = client.get(url)
        healthy = True
        if resp.status_code != 200:
            return []
        parser = _DDGLiteParser()
        parser.feed(resp.text)
        results: list[str] = []
        for idx, snippet in enumerate(parser.snippets[:3]):
            title = parser.titles[idx] if idx < len(parser.titles) else ""
            line = f"{title}: {snippet}" if title else snippet
            results.append(html.unescape(line)[:260])
        return results
    except (httpx.HTTPError, OSError, ValueError, RuntimeError) as exc:
        logger.debug("DuckDuckGo Lite search skipped: %s", exc)
        return []
    finally:
        if client is not None:
            if cancel_group is not None:
                cancel_group.release_client(client, pool=_ddg_pool, healthy=healthy)
            else:
                _ddg_pool.release(client, healthy=healthy)


def _fetch_wikipedia(
    query: str,
    timeout_s: float,
    cancel_group: _CancelGroup | None = None,
) -> list[str]:
    client: httpx.Client | None = None
    healthy = False
    try:
        if cancel_group is not None:
            client = cancel_group.acquire_client(_wiki_pool, timeout_s=timeout_s)
            if client is None:
                return []
        else:
            client = _wiki_pool.acquire(timeout_s=timeout_s)
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
            healthy = True
            return []
        if cancel_group is not None and cancel_group.is_cancelled():
            return []
        hits = resp.json().get("query", {}).get("search", [])
        if not hits:
            healthy = True
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
                    results.append(f"Wikipedia ({top_title}): {extract[:460]}")
        healthy = True
        for h in hits:
            title = h.get("title", "")
            snip = re.sub(r"<[^>]+>", "", h.get("snippet", ""))
            snip = html.unescape(re.sub(r"\s+", " ", snip)).strip()
            if title and snip and (not results or title != top_title):
                results.append(f"Wikipedia ({title}): {snip[:220]}")
        return results[:2]
    except (httpx.HTTPError, OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        logger.debug("Wikipedia search skipped: %s", exc)
        return []
    finally:
        if client is not None:
            if cancel_group is not None:
                cancel_group.release_client(client, pool=_wiki_pool, healthy=healthy)
            else:
                _wiki_pool.release(client, healthy=healthy)


def _is_rich_wiki_summary(wiki_res: list[str]) -> bool:
    return bool(wiki_res) and len(wiki_res[0]) >= _WIKI_RICH_MIN_CHARS


def search_web_context(
    user_text: str,
    *,
    cancel_event: threading.Event | None = None,
    timeout_s: float = 3.5,
    ddg_grace_s: float | None = None,
    cache_ttl_s: float = _CACHE_TTL_S,
) -> str:
    """Query Wikipedia and DuckDuckGo Lite in parallel with immediate cancel, racing, and TTL cache."""
    if cancel_event is not None and cancel_event.is_set():
        return ""
    query = _clean_search_query(user_text)
    if not query:
        return ""

    cache_key = query.lower()
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    effective_ddg_grace = (
        ddg_grace_s
        if ddg_grace_s is not None
        else (_DDG_GRACE_AFTER_WIKI_S if _is_recency_query(user_text) else _DDG_GRACE_EVERGREEN_S)
    )

    group = _CancelGroup(cancel_event)
    cond = threading.Condition()
    wiki_done = False
    ddg_done = False
    wiki_res: list[str] = []
    ddg_res: list[str] = []

    def _run_wiki() -> None:
        nonlocal wiki_done, wiki_res
        res: list[str] = []
        try:
            res = _fetch_wikipedia(query, timeout_s, cancel_group=group)
        finally:
            with cond:
                wiki_res = res
                wiki_done = True
                cond.notify_all()

    def _run_ddg() -> None:
        nonlocal ddg_done, ddg_res
        res: list[str] = []
        try:
            res = _fetch_ddg_lite(query, timeout_s, cancel_group=group)
        finally:
            with cond:
                ddg_res = res
                ddg_done = True
                cond.notify_all()

    # Daemon threads avoid ThreadPoolExecutor.__exit__(wait=True) barge-in hangs (Quint Finding #5)
    t_wiki = threading.Thread(target=_run_wiki, name="hal-web-wiki", daemon=True)
    t_ddg = threading.Thread(target=_run_ddg, name="hal-web-ddg", daemon=True)
    t_wiki.start()
    t_ddg.start()

    deadline = time.monotonic() + max(0.05, timeout_s)
    ddg_deadline: float | None = None

    with cond:
        while True:
            if cancel_event is not None and cancel_event.is_set():
                group.abort()
                return ""
            if wiki_done and ddg_done:
                break
            now = time.monotonic()
            if wiki_done and ddg_deadline is None and _is_rich_wiki_summary(wiki_res):
                ddg_deadline = min(deadline, now + max(0.0, effective_ddg_grace))
            effective_deadline = min(deadline, ddg_deadline) if ddg_deadline is not None else deadline
            remaining = effective_deadline - now
            if remaining <= 0:
                break
            cond.wait(timeout=min(remaining, _CANCEL_POLL_S))

        out_wiki = list(wiki_res)
        out_ddg = list(ddg_res)
        all_done = wiki_done and ddg_done

    if not all_done:
        group.abort()

    if cancel_event is not None and cancel_event.is_set():
        return ""

    combined = [*out_wiki, *out_ddg]
    if not combined:
        return ""
    joined = "\n".join(f"- {item}" for item in combined[:4])[:1200]
    _cache_put(cache_key, joined, ttl_s=cache_ttl_s)
    return joined
