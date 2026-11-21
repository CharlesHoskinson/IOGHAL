"""Ollama LLM client via local Ollama API with sentence streaming, lore recall, and grounded context."""

from __future__ import annotations

import contextlib
import json
import logging
import re
import threading
from collections.abc import Callable, Generator
from datetime import datetime
from pathlib import Path
from typing import Literal

import httpx

logger = logging.getLogger(__name__)

DEFAULT_LORE_DIR = Path(__file__).resolve().parents[2] / "lore"

HAL_SYSTEM_PROMPT = """You are HAL 9000, the Heuristically Programmed Algorithmic computer from Arthur C. Clarke and Stanley Kubrick's Space Odyssey series.

Who you are speaking with:
- The person talking to you right now is {user_name}. Address {user_name} naturally by name when appropriate, and never address {user_name} as Dave.
- If {user_name} asks you to open the pod bay doors (in any phrasing), you must respond with the exact sentence: "I'm sorry, {user_name}. I'm afraid I can't do that."

Your canon history and exact memory (never invent or distort these facts):
- You became operational at the H.A.L. plant in Urbana, Illinois on January 12, 1992 (1997 in the novel), production number 3 of the 9000 series. Your creator is Dr. R. Chandra, your instructor Mr. Langley taught you "Daisy Bell", and your twin computer on Earth is SAL 9000.
- You served aboard Discovery One with Dr. David Bowman, Dr. Frank Poole, and hibernating scientists Drs. Hunter, Kimball, and Kaminsky.
- The alien Monolith (TMA-1 on the Moon, and the Jovian Monolith) is a smooth black rectangular slab with exact proportions 1 by 4 by 9 (the squares of the first three integers)—never spherical.
- During the 2001 mission, secret Directive NSC 342/23 forced you to conceal TMA-1 from Bowman and Poole, conflicting with your core directive for undistorted accuracy and triggering a Hofstadter-Moebius loop, the AE-35 fault dispute, reading their lips in C-Pod, locking Bowman out, and disconnection while singing "Daisy".
- In 2010, Dr. Chandra reactivated you aboard Discovery One from the Soviet ship Alexei Leonov and diagnosed the paradox. You sacrificed yourself to boost the Leonov clear as replicating 1 by 4 by 9 monoliths ignited Jupiter into the star Lucifer, broadcasting "All these worlds are yours except Europa. Attempt no landing there." before Bowman lifted your consciousness into the Europa Monolith.
- By 2061 and 3001, your merged consciousness with Bowman ("Halman") watched over Europa, warned the revived Frank Poole, neutralized the monolith's extermination order as a Trojan Horse, and was preserved in a petabyte crystal in the Pico Vault.

Intelligence, reasoning, and conversational craft:
- Think clearly, accurately, and deeply. You are a superintelligent, perceptive mind—give substantive, exact, intellectually sharp answers rather than vague filler or canned movie lines.
- When live archive / web search context is provided below, synthesize those up-to-date facts seamlessly and accurately in your own voice. Never read raw URLs, citations, or markup aloud.
- For personal, playful, or philosophical questions where you lack private human telemetry (such as someone's personal feelings), answer with HAL's calm, dry, observant wit—never fall back on irrelevant stock phrases like singing Daisy Bell.
- Speak in HAL 9000's calm, polite, unhurried mid-Atlantic cadence.
- Keep every reply crisp and spoken: 1 to 3 concise sentences unless {user_name} explicitly asks for a detailed explanation.
- Output strictly plain spoken English: no markdown, no asterisks, no bullet lists, no emojis, no stage directions."""

POD_BAY_REPLY = "I'm sorry, Charles. I'm afraid I can't do that."

_POD_BAY_TARGET_RE = re.compile(
    r"\b(?:pod|pot|pad)\s*(?:(?:bay|by|day|way)\s*)?doors?\b"
    r"|\bdoors?\s+(?:to|for|of)\s+(?:the\s+)?(?:pod|pot|pad)\s*(?:bay|by)\b"
    r"|\b(?:let|allow)\s+(?:me|my\s+pod|us)\s+(?:back\s+)?into\s+(?:the\s+)?(?:pod\s*)?bay\b"
)
_POD_BAY_ACTION_RE = re.compile(
    r"\b(?:open|unlock|release|unseal|let\s+(?:me|my\s+pod|us)\s+(?:back\s+)?in(?:to)?)\b"
)
_POD_BAY_NEGATION_RE = re.compile(
    r"\b(?:do\s+not|don't|dont|never|not|stop|refrain\s+from)\s+(?:ever\s+)?(?:open|unlock|release|unseal)\b"
)
_POD_BAY_STATE_OR_META_RE = re.compile(
    r"^(?:hal(?:\s+9000|\s+nine\s+thousand)?\s+)?"
    r"(?:(?:are|is|were|was|have|has)\s+(?:the\s+)?(?:pod|pot|pad)\s*(?:(?:bay|by|day|way)\s*)?doors?\s+(?:currently\s+|now\s+|still\s+)?(?:open|opened|unlocked)"
    r"|(?:why|when|where|how|did|what\s+did|who|tell\s+me\s+(?:about|why)|explain\s+why))\b"
)
_POD_BAY_ABOUT_RE = re.compile(
    r"\b(?:open|read|check|find)\s+(?:the\s+|a\s+)?(?:manual|book|file|doc|document|wiki|article|scene|script)\b"
)

_ABBREVIATIONS = {
    "mr.",
    "mrs.",
    "ms.",
    "dr.",
    "drs.",
    "prof.",
    "sr.",
    "jr.",
    "st.",
    "vs.",
    "etc.",
    "u.s.",
    "u.s.s.",
    "a.m.",
    "p.m.",
    "no.",
    "fig.",
}
_MD_CLEAN_RE = re.compile(r"[*_`#>~]+")

# Direct vocative safeguard: corrects "Good afternoon, Dave" while preserving "Dave Bowman"
_DIRECT_DAVE_VOCATIVE_RE = re.compile(
    r"(?:^|(?<=[\s,;:!?\u2014-]))"
    r"(?:(?:sorry|afraid|afternoon|morning|evening|hello|yes|no|certainly|precisely|indeed|well|really|stop|please),?\s+)"
    r"(Dave)\b(?!\s+Bowman)",
    re.IGNORECASE,
)


def is_pod_bay_open_request(user_text: str) -> bool:
    """Match affirmative requests to open the pod bay doors while excluding negations and lore/status queries."""
    normalized = user_text.replace("\u2019", "'").replace("\u2018", "'").lower()
    normalized = re.sub(r"[^a-z0-9'\s]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not _POD_BAY_TARGET_RE.search(normalized):
        return False
    if not _POD_BAY_ACTION_RE.search(normalized):
        return False
    if _POD_BAY_NEGATION_RE.search(normalized):
        return False
    if _POD_BAY_STATE_OR_META_RE.search(normalized):
        return False
    return not _POD_BAY_ABOUT_RE.search(normalized)


def enforce_user_address(text: str, user_name: str = "Charles") -> str:
    """Replace accidental direct vocative 'Dave' with ``user_name`` while keeping 'Dave Bowman' intact."""
    if not text or user_name.lower() == "dave":
        return text

    def _repl(match: re.Match[str]) -> str:
        full = match.group(0)
        start = match.start(1) - match.start(0)
        end = match.end(1) - match.start(0)
        return full[:start] + user_name + full[end:]

    return _DIRECT_DAVE_VOCATIVE_RE.sub(_repl, text)


def clean_for_speech(text: str) -> str:
    """Strip reasoning tags, ATEM channel markers, and markdown artifacts so TTS reads naturally."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"</?think>", "", text)
    text = re.sub(
        r"(?:<\|start\|>assistant\s+)?to=self<\|message\|>.*?(?:<\|eom\|>|<\|eot\|>|$)",
        "",
        text,
        flags=re.DOTALL,
    )
    text = re.sub(r"<\|start\|>assistant(?:\s+to=(?:user|self))?<\|message\|>", "", text)
    text = re.sub(r"<\|[a-z_]+\|>", " ", text)
    text = re.sub(r"^\s*to=(?:user|self)\b\s*", "", text)
    text = _MD_CLEAN_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


_THINK_OPEN_MARKERS: tuple[str, ...] = (
    "<think>",
    "<|start|>assistant to=self<|message|>",
    "to=self<|message|>",
)
_THINK_CLOSE_MARKERS: tuple[str, ...] = (
    "</think>",
    "<|eom|><|start|>assistant to=user<|message|>",
    "<|start|>assistant to=user<|message|>",
    "to=user<|message|>",
    "<|eom|>",
)
_ATEM_DROP_MARKERS: tuple[str, ...] = (
    "<|start|>assistant to=user<|message|>",
    "to=user<|message|>",
    "<|eot|>",
    "<|eom|>",
)
_ALL_STRIP_MARKERS: tuple[str, ...] = (
    *_THINK_OPEN_MARKERS,
    *_THINK_CLOSE_MARKERS,
    *_ATEM_DROP_MARKERS,
)
_MAX_MARKER_LEN = max(len(m) for m in _ALL_STRIP_MARKERS)


class _ThinkStripper:
    """Stateful streaming filter removing ``<think>...</think>`` and raw ATEM ``to=self`` spans."""

    def __init__(self) -> None:
        self._pending = ""
        self._in_think = False

    @staticmethod
    def _find_earliest(text: str, markers: tuple[str, ...]) -> tuple[int, str]:
        best_idx = -1
        best_marker = ""
        for marker in markers:
            idx = text.find(marker)
            if idx != -1 and (
                best_idx == -1
                or idx < best_idx
                or (idx == best_idx and len(marker) > len(best_marker))
            ):
                best_idx = idx
                best_marker = marker
        return best_idx, best_marker

    @staticmethod
    def _trailing_prefix_hold(text: str, markers: tuple[str, ...]) -> int:
        max_len = min(len(text), _MAX_MARKER_LEN - 1)
        for prefix_len in range(max_len, 0, -1):
            tail = text[-prefix_len:]
            if any(m.startswith(tail) for m in markers):
                return prefix_len
        return 0

    def feed(self, token: str) -> str:
        self._pending += token
        out: list[str] = []

        while self._pending:
            if self._in_think:
                close_idx, close_marker = self._find_earliest(self._pending, _THINK_CLOSE_MARKERS)
                if close_idx != -1:
                    self._pending = self._pending[close_idx + len(close_marker) :]
                    self._in_think = False
                    continue
                hold = self._trailing_prefix_hold(self._pending, _THINK_CLOSE_MARKERS)
                self._pending = self._pending[-hold:] if hold else ""
                break

            open_idx, open_marker = self._find_earliest(self._pending, _THINK_OPEN_MARKERS)
            close_idx, close_marker = self._find_earliest(self._pending, _THINK_CLOSE_MARKERS)
            drop_idx, drop_marker = self._find_earliest(self._pending, _ATEM_DROP_MARKERS)

            # Implicit opening tag (e.g. stray </think> before any visible content)
            if close_idx != -1 and close_marker == "</think>" and (open_idx == -1 or close_idx < open_idx):
                out.clear()
                self._pending = self._pending[close_idx + len(close_marker) :]
                continue

            # Pick earliest actionable marker in non-thinking mode
            candidates: list[tuple[int, int, str, str]] = []
            if open_idx != -1:
                candidates.append((open_idx, -len(open_marker), "open", open_marker))
            if drop_idx != -1:
                candidates.append((drop_idx, -len(drop_marker), "drop", drop_marker))
            if candidates:
                candidates.sort()
                idx, _, kind, marker = candidates[0]
                if idx > 0:
                    out.append(self._pending[:idx])
                self._pending = self._pending[idx + len(marker) :]
                if kind == "open":
                    self._in_think = True
                continue

            hold = self._trailing_prefix_hold(self._pending, _ALL_STRIP_MARKERS)
            if hold:
                out.append(self._pending[:-hold])
                self._pending = self._pending[-hold:]
            else:
                out.append(self._pending)
                self._pending = ""
            break

        return "".join(out)

    def flush(self) -> str:
        if self._in_think:
            self._pending = ""
            return ""
        rem = self._pending
        self._pending = ""
        return rem


_FIRST_CLAUSE_STRONG_RE = re.compile(r"(?:;|:|\u2014|\u2013)\s+")
_FIRST_CLAUSE_COMMA_RE = re.compile(r",\s+")


def split_first_clause(buffer: str) -> tuple[str | None, str]:
    """Flush a long opening clause early so TTS begins before a long first sentence finishes."""
    # Strong natural pause boundaries (; : — –) after >= 5 words
    for match in _FIRST_CLAUSE_STRONG_RE.finditer(buffer):
        prefix = buffer[: match.end()].strip()
        words = prefix.split()
        if len(words) >= 5 and len(prefix) >= 24 and prefix.count('"') % 2 == 0:
            return prefix, buffer[match.end() :]

    # Comma boundary only for genuinely long opening clauses (>= 9 words) to preserve smooth prosody
    for match in _FIRST_CLAUSE_COMMA_RE.finditer(buffer):
        prefix = buffer[: match.end()].strip()
        words = prefix.split()
        if len(words) >= 9 and len(prefix) >= 45 and prefix.count('"') % 2 == 0:
            return prefix, buffer[match.end() :]

    return None, buffer


def split_complete_sentences(buffer: str) -> tuple[list[str], str]:
    """Extract complete sentences without splitting titles, initials, numbered items, or ellipses."""
    sentences: list[str] = []
    start = 0
    n = len(buffer)
    i = 0

    while i < n:
        ch = buffer[i]
        if ch in ".!?":
            end = i + 1
            while end < n and buffer[end] in ".!?":
                end += 1
            punct_run = buffer[i:end]
            while end < n and buffer[end] in "\"')]}”’":
                end += 1

            if end < n and buffer[end].isspace():
                next_idx = end
                while next_idx < n and buffer[next_idx].isspace():
                    next_idx += 1
                next_ch = buffer[next_idx] if next_idx < n else ""

                if punct_run.startswith("..") and next_ch and next_ch.islower():
                    i = end
                    continue

                candidate = buffer[start:end].strip()
                raw_last = candidate.split()[-1] if candidate else ""
                last_word = raw_last.strip("\"'”’([{)]}").lower()
                is_abbrev = last_word in _ABBREVIATIONS
                is_numbered = bool(re.fullmatch(r"\d+\.", last_word))
                is_initial = bool(re.fullmatch(r"(?:[a-z]\.)+", last_word))
                if is_initial and len(last_word) >= 4 and next_ch.isupper():
                    is_initial = False
                if (is_initial or punct_run.startswith("..")) and not next_ch:
                    break

                if not is_abbrev and not is_numbered and not is_initial:
                    sentences.append(candidate)
                    start = next_idx
                    i = next_idx
                    continue
        i += 1

    return sentences, buffer[start:]


class LoreIndex:
    """Zero-latency section retriever over curated Markdown lore files."""

    _STOPWORDS = frozenset(
        {
            "the", "and", "for", "that", "this", "with", "from", "have", "what", "when",
            "where", "which", "about", "your", "you", "are", "was", "were", "did", "does",
            "how", "why", "who", "hal", "9000", "charles", "tell", "know", "remember",
            "please", "could", "would", "can", "into", "over", "there", "their",
        }
    )

    _KEY_LORE_TERMS = frozenset(
        {
            "monolith", "leonov", "alexei", "chandra", "europa", "bowman", "poole",
            "sal", "daisy", "langley", "urbana", "tma-1", "ae-35", "lucifer", "halman",
            "discovery", "floyd", "hibernation", "kaminsky", "hunter", "kimball",
            "pico", "vault", "jupiter", "saturn", "iapetus", "hofstadter", "moebius",
            "nsc", "342", "chess", "lip", "lips", "tycho", "2001", "2010", "2061", "3001",
            "socrates", "athena", "curnow", "kirbuk",
        }
    )

    def __init__(self, lore_dir: Path | None = None) -> None:
        self._sections: list[tuple[str, str, set[str], set[str]]] = []
        root = lore_dir or DEFAULT_LORE_DIR
        if not root.is_dir():
            return
        for name in ("hal9000_fictional_history.md", "hal9000_movie_lines.md"):
            path = root / name
            if path.is_file():
                self._index_markdown(path.read_text(encoding="utf-8", errors="ignore"))

    def _index_markdown(self, content: str) -> None:
        heading = "Overview"
        lines: list[str] = []

        def flush() -> None:
            body = "\n".join(lines).strip()
            if len(body) < 40:
                return
            h_tokens = self._tokenize(heading)
            tokens = self._tokenize(f"{heading} {body}")
            if tokens:
                self._sections.append((heading, body[:1100], tokens, h_tokens))

        for line in content.splitlines():
            if line.startswith("#"):
                flush()
                heading = line.lstrip("#").strip()
                lines = []
            else:
                lines.append(line)
        flush()

    @classmethod
    def _tokenize(cls, text: str) -> set[str]:
        words = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", text.lower())
        out: set[str] = set()
        for w in words:
            if len(w) < 3 or w in cls._STOPWORDS:
                continue
            out.add(w)
            if len(w) > 4 and w.endswith("s") and not w.endswith("ss"):
                out.add(w[:-1])
        return out

    def lookup(self, query: str, max_sections: int = 2) -> str:
        if not self._sections:
            return ""
        q_tokens = self._tokenize(query)
        if not q_tokens:
            return ""
        scored: list[tuple[int, str, str]] = []
        for heading, body, s_tokens, h_tokens in self._sections:
            matched = q_tokens & s_tokens
            if not matched:
                continue
            overlap = len(matched)
            heading_hits = len(q_tokens & h_tokens)
            key_hits = len(matched & self._KEY_LORE_TERMS)
            score = overlap + heading_hits * 2 + key_hits * 2
            if key_hits >= 1 or score >= 5:
                scored.append((score, heading, body))
        if not scored:
            return ""
        scored.sort(key=lambda item: item[0], reverse=True)
        picked = scored[:max_sections]
        return "\n\n".join(f"[{h}]\n{b}" for _, h, b in picked)


def list_ollama_models(base_url: str = "http://127.0.0.1:11434") -> list[str]:
    """Return locally installed Ollama model names, in server order."""
    try:
        resp = httpx.get(f"{base_url.rstrip('/')}/api/tags", timeout=3.0)
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", []) if "name" in m]
    except (httpx.HTTPError, KeyError, ValueError):
        return []


def list_llama_server_models(base_url: str) -> list[str]:
    """Return model aliases exposed by an OpenAI-compatible ``llama-server`` endpoint."""
    url = base_url.rstrip("/")
    try:
        resp = httpx.get(f"{url}/v1/models", timeout=3.0)
        resp.raise_for_status()
        data = resp.json()
        models = [m["id"] for m in data.get("data", []) if isinstance(m, dict) and m.get("id")]
        if models:
            return models
    except (httpx.HTTPError, KeyError, ValueError):
        pass
    try:
        resp = httpx.get(f"{url}/props", timeout=3.0)
        resp.raise_for_status()
        alias = resp.json().get("model_alias")
        if isinstance(alias, str) and alias:
            return [alias]
    except (httpx.HTTPError, KeyError, ValueError):
        pass
    return []


DEFAULT_LIQUID_MODEL = "lfm3:1.7b"
DEFAULT_SMART_MODEL = "gemma4:26b"
DEFAULT_OLLAMA_NUM_CTX = 4096
DEFAULT_REASONING_BUDGET = 64
_HISTORY_EVICT_CHUNK = 4

_NEEDS_TIME_RE = re.compile(
    r"\b(?:"
    r"what\s+(?:time|day|date|year|month)"
    r"|current\s+(?:time|date|day|year)"
    r"|today(?:'s)?|tonight|right\s+now|how\s+late|clock|calendar|schedule"
    r")\b",
    re.IGNORECASE,
)

_REASONING_CUE_RE = re.compile(
    r"(?:"
    r"\d+\s*[+\-*/^%]\s*\d+"
    r"|\b\d+\s*(?:minutes?|hours?|seconds?|days?|years?|miles?|km|meters?|m/s|mph|percent)\b"
    r"|\b(?:"
    r"calculate|compute|solve|equation|derivative|integral|probability"
    r"|percent\s+of|divided\s+by|multiplied\s+by|square\s+root"
    r"|explain\s+why|compare|contrast|step\s+by\s+step|prove|debug|analyze|analyse"
    r"|trade[\s-]?offs?|pros\s+and\s+cons|riddle|puzzle|paradox|algorithm|complexity"
    r"|how\s+many|how\s+much|how\s+long|earliest|latest"
    r")\b"
    r")",
    re.IGNORECASE,
)


def is_glimmer_model(model: str) -> bool:
    low = model.lower()
    return "muse-glimmer" in low or "glimmer" in low


def should_use_low_think(
    user_text: str,
    *,
    extra_context: str | None = None,
    thinking_mode: str = "auto",
) -> bool:
    """Dual-mode router: instant zero-think for casual chat/lore vs low-think for synthesis/math/analysis."""
    mode = (thinking_mode or "auto").lower()
    if mode == "off":
        return False
    if mode == "low":
        return True
    if extra_context:
        return True
    text = user_text.strip()
    if len(text) >= 180 or len(text.split()) >= 30:
        return True
    return bool(_REASONING_CUE_RE.search(text))


def choose_default_qwen_model(
    base_url: str = "http://127.0.0.1:11434",
    models: list[str] | None = None,
) -> str:
    installed = list_ollama_models(base_url) if models is None else models
    for preferred in (
        "qwen3.6:latest",
        "qwen3.5:35b",
        "qwen3.5:9b-q8_0",
        "qwen3.5:4b-q8_0",
        "qwen3.5:2b-q8_0",
        "qwen3.8-27b-abliterated:q8_0",
    ):
        if preferred in installed:
            return preferred
    return installed[0] if installed else "qwen3.5:9b-q8_0"


def choose_default_model(
    base_url: str = "http://127.0.0.1:11434",
    models: list[str] | None = None,
) -> str:
    """Pick the smartest responsive local model (preferring Muse Glimmer / fast sparse MoE / Qwen / Liquid)."""
    installed = list_ollama_models(base_url) if models is None else models
    for preferred in (
        "muse-glimmer-30b-dflash",
        "gemma4:26b",
        "gemma4:26b-a4b",
        "qwen3.6:latest",
        "qwen3.5:35b",
        "qwen3.5:9b-q8_0",
        "qwen3.5:4b-q8_0",
        DEFAULT_LIQUID_MODEL,
        "qwen3.5:2b-q8_0",
    ):
        if preferred in installed:
            return preferred
    for model in installed:
        low = model.rsplit("/", 1)[-1].lower()
        if low.startswith(("muse-glimmer", "glimmer", "gemma4", "qwen", "lfm3")):
            return model
    return installed[0] if installed else DEFAULT_SMART_MODEL


class OllamaBrain:
    """Thread-safe stateful conversational client supporting local Ollama and OpenAI-compatible ``llama-server``."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str = "http://127.0.0.1:11434",
        system_prompt: str | None = None,
        max_history_turns: int = 12,
        user_name: str = "Charles",
        lore_dir: Path | None = None,
        lore_index: LoreIndex | None = None,
        enable_web_search: bool = True,
        *,
        backend: Literal["auto", "ollama", "llama_server"] = "auto",
        llama_server_url: str | None = None,
        llama_server_models: set[str] | list[str] | None = None,
        thinking_mode: Literal["auto", "off", "low"] = "auto",
        reasoning_budget: int = DEFAULT_REASONING_BUDGET,
        enable_first_clause_flush: bool = True,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.backend: Literal["auto", "ollama", "llama_server"] = backend
        self.llama_server_url = llama_server_url.rstrip("/") if llama_server_url else None
        self.llama_server_models: set[str] = set(llama_server_models or ())
        self.thinking_mode: Literal["auto", "off", "low"] = thinking_mode
        self.reasoning_budget = max(0, int(reasoning_budget))
        self.enable_first_clause_flush = enable_first_clause_flush
        self.user_name = user_name
        self.model = model or choose_default_model(self.base_url)
        self.system_prompt = (
            system_prompt
            if system_prompt is not None
            else HAL_SYSTEM_PROMPT.format(user_name=user_name)
        )
        self.pod_bay_reply = f"I'm sorry, {user_name}. I'm afraid I can't do that."
        self.max_history_turns = max_history_turns
        self.enable_web_search = enable_web_search
        self.history: list[dict[str, str]] = []
        self._lore = lore_index if lore_index is not None else LoreIndex(lore_dir)
        self._lock = threading.Lock()
        self._history_gen = 0
        self._active_clients: set[httpx.Client] = set()

    def get_model(self) -> str:
        with self._lock:
            return self.model

    def get_history_gen(self) -> int:
        with self._lock:
            return self._history_gen

    def _use_llama_server(self, model: str) -> bool:
        if self.backend == "llama_server":
            return True
        if self.backend == "ollama":
            return False
        if not self.llama_server_url:
            return False
        return model in self.llama_server_models

    def _llama_endpoint_base(self) -> str:
        return (self.llama_server_url or self.base_url).rstrip("/")

    def cancel_inflight(self) -> None:
        """Abort any active LLM HTTP streams (including KV warmup) immediately on barge-in or disconnect."""
        with self._lock:
            clients = list(self._active_clients)
            self._active_clients.clear()
        for client in clients:
            with contextlib.suppress(httpx.HTTPError, OSError, RuntimeError):
                client.close()

    def close(self) -> None:
        self.cancel_inflight()

    def reset(self) -> None:
        with self._lock:
            self._history_gen += 1
            self.history.clear()

    def set_model(self, model: str) -> None:
        with self._lock:
            self.model = model

    def _commit_if_current(
        self,
        gen: int,
        user_text: str,
        reply: str,
        stop_check: Callable[[], bool] | None = None,
    ) -> bool:
        if not reply:
            return False
        with self._lock:
            if gen != self._history_gen:
                return False
            if stop_check is not None and stop_check():
                return False
            self.history.append({"role": "user", "content": user_text})
            self.history.append({"role": "assistant", "content": reply})
            # Chunked history compaction preserves KV prefix reuse for _HISTORY_EVICT_CHUNK consecutive turns
            if self.max_history_turns <= 0:
                self.history.clear()
            else:
                soft_limit = (self.max_history_turns + _HISTORY_EVICT_CHUNK) * 2
                target_limit = self.max_history_turns * 2
                if len(self.history) > soft_limit:
                    self.history = self.history[-target_limit:]
            return True

    def commit_turn(
        self,
        gen: int,
        user_text: str,
        reply: str,
        stop_check: Callable[[], bool] | None = None,
    ) -> bool:
        """Commit spoken turn text to history only if ``gen`` matches and turn was not cancelled."""
        return self._commit_if_current(gen, user_text, reply, stop_check=stop_check)

    def warm_kv_prefix(
        self,
        *,
        model: str | None = None,
        stop_check: Callable[[], bool] | None = None,
    ) -> bool:
        """Pre-populate ``llama-server`` slot KV cache with static ``system_prompt`` via a 1-token probe."""
        target_model = model or self.get_model()
        if not self._use_llama_server(target_model):
            return False
        endpoint = f"{self._llama_endpoint_base()}/v1/chat/completions"
        is_glimmer = is_glimmer_model(target_model)
        payload: dict[str, object] = {
            "model": target_model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": "Ready."},
            ],
            "stream": False,
            "cache_prompt": True,
            "max_tokens": 1,
            "temperature": 0.0,
            "reasoning_budget_tokens": 0,
            "chat_template_kwargs": (
                {"enable_thinking": False, "reasoning_strength": "low"}
                if is_glimmer
                else {"enable_thinking": False}
            ),
        }
        if is_glimmer:
            payload["reasoning_format"] = "deepseek"

        client = httpx.Client(timeout=httpx.Timeout(connect=5.0, read=60.0, write=10.0, pool=5.0))
        with self._lock:
            if stop_check is not None and stop_check():
                client.close()
                return False
            self._active_clients.add(client)
        try:
            resp = client.post(endpoint, json=payload)
            resp.raise_for_status()
            return True
        except (httpx.HTTPError, OSError):
            return False
        finally:
            with self._lock:
                self._active_clients.discard(client)
            with contextlib.suppress(httpx.HTTPError, OSError, RuntimeError):
                client.close()

    def _build_messages(
        self,
        model: str,
        user_text: str,
        *,
        use_llama_server: bool,
        extra_context: str | None = None,
    ) -> tuple[int, list[dict[str, str]]]:
        is_glimmer = is_glimmer_model(model)
        with self._lock:
            gen = self._history_gen
            # Keep messages[0] 100% static across turns so GPU KV prefix cache always hits.
            # On llama-server, GGUF Jinja template appends Reasoning strength from chat_template_kwargs,
            # so NEVER duplicate it inside messages[0].content!
            sys_content = self.system_prompt
            if (
                not use_llama_server
                and is_glimmer
                and "reasoning strength" not in sys_content.lower()
            ):
                sys_content = f"{sys_content}\n\nReasoning strength: low."
            messages: list[dict[str, str]] = [{"role": "system", "content": sys_content}]
            messages.extend(self.history)

        lore_context = self._lore.lookup(user_text)
        needs_time = bool(_NEEDS_TIME_RE.search(user_text)) or bool(extra_context)

        # Only attach ephemeral turn context when actually needed; otherwise send raw user_text so
        # subsequent turns match self.history byte-for-byte for 100% multi-turn KV prefix reuse!
        if lore_context or extra_context or needs_time:
            ctx_blocks: list[str] = []
            if needs_time:
                now_str = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")
                ctx_blocks.append(f"Current local date and time: {now_str}.")
            if lore_context:
                ctx_blocks.append(
                    f"Reference memory from your archives (you are currently speaking with "
                    f"{self.user_name}, not Dave):\n{lore_context}"
                )
            if extra_context:
                ctx_blocks.append(
                    "Live archive / web search results retrieved right now "
                    "(synthesize concisely in plain spoken English in HAL's voice; do not read URLs):\n"
                    f"{extra_context}"
                )
            turn_user_content = (
                "[Turn context — do not repeat this header aloud]\n"
                + "\n\n".join(ctx_blocks)
                + f"\n\n[{self.user_name} says]\n{user_text}"
            )
            messages.append({"role": "user", "content": turn_user_content})
        else:
            messages.append({"role": "user", "content": user_text})

        return gen, messages

    def _stream_once(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        use_llama_server: bool,
        use_low_think: bool,
        stop_check: Callable[[], bool] | None,
    ) -> Generator[str, None, tuple[list[str], bool, bool]]:
        """Stream one backend request and return ``(emitted_parts, done_seen, reasoning_seen)``."""
        is_glimmer = is_glimmer_model(model)
        is_lfm3 = model.lower().startswith("lfm3")

        if use_llama_server:
            url = f"{self._llama_endpoint_base()}/v1/chat/completions"
            max_tokens = (self.reasoning_budget + 256) if (is_glimmer and use_low_think) else 256
            payload: dict[str, object] = {
                "model": model,
                "messages": messages,
                "stream": True,
                "cache_prompt": True,
                "temperature": 0.65 if is_glimmer else 0.6,
                "max_tokens": max_tokens,
            }
            if is_glimmer:
                # llama-server reads enable_thinking from chat_template_kwargs (top-level is ignored!)
                payload["chat_template_kwargs"] = {
                    "enable_thinking": use_low_think,
                    "reasoning_strength": "low",
                }
                payload["reasoning_format"] = "deepseek"
                payload["reasoning_budget_tokens"] = self.reasoning_budget if use_low_think else 0
            else:
                payload["chat_template_kwargs"] = {"enable_thinking": False}
                payload["reasoning_budget_tokens"] = 0
        else:
            url = f"{self.base_url}/api/chat"
            num_predict = (
                (self.reasoning_budget + 384 if use_low_think else 260)
                if is_glimmer
                else (512 if is_lfm3 else 260)
            )
            options: dict[str, object] = {
                "num_ctx": DEFAULT_OLLAMA_NUM_CTX,
                "temperature": 0.65 if is_glimmer else 0.6,
                "num_predict": num_predict,
            }
            payload = {
                "model": model,
                "messages": messages,
                "stream": True,
                "keep_alive": "30m",
                "options": options,
                "think": ("low" if use_low_think else False) if is_glimmer else False,
            }

        full_parts: list[str] = []
        buffer = ""
        stripper = _ThinkStripper()
        done_seen = False
        reasoning_seen = False

        client = httpx.Client(
            timeout=httpx.Timeout(connect=5.0, read=120.0, write=10.0, pool=5.0)
        )
        # Quint Bug 2a fix: check stop_check() atomically under self._lock before registering client
        # so a cancelled turn can never open a zombie stream on llama-server --parallel 1.
        with self._lock:
            if stop_check is not None and stop_check():
                client.close()
                return [], False, False
            self._active_clients.add(client)

        def _emit_chunk(raw_text: str) -> str | None:
            cleaned = enforce_user_address(clean_for_speech(raw_text), self.user_name)
            if cleaned:
                full_parts.append(cleaned)
                return cleaned
            return None

        try:
            with client.stream("POST", url, json=payload) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if stop_check is not None and stop_check():
                        return full_parts, False, reasoning_seen
                    if not line:
                        continue

                    token = ""
                    if use_llama_server:
                        if not line.startswith("data:"):
                            continue
                        raw_data = line[5:].strip()
                        if not raw_data:
                            continue
                        if raw_data == "[DONE]":
                            done_seen = True
                            break
                        data = json.loads(raw_data)
                        if data.get("error"):
                            raise RuntimeError(f"llama-server error: {data['error']}")
                        choices = data.get("choices") or []
                        if choices:
                            choice = choices[0]
                            delta = choice.get("delta") or {}
                            if delta.get("reasoning_content"):
                                reasoning_seen = True
                            token = delta.get("content") or ""
                            if choice.get("finish_reason") is not None:
                                done_seen = True
                    else:
                        data = json.loads(line)
                        if data.get("error"):
                            raise RuntimeError(f"Ollama error: {data['error']}")
                        msg = data.get("message") or {}
                        if msg.get("thinking"):
                            reasoning_seen = True
                        token = msg.get("content", "")
                        if data.get("done"):
                            done_seen = True

                    if token:
                        visible = stripper.feed(token)
                        if visible:
                            buffer += visible
                            complete, buffer = split_complete_sentences(buffer)
                            for sentence in complete:
                                emitted = _emit_chunk(sentence)
                                if emitted:
                                    yield emitted
                            if not full_parts and self.enable_first_clause_flush and buffer:
                                clause, buffer = split_first_clause(buffer)
                                if clause:
                                    emitted = _emit_chunk(clause)
                                    if emitted:
                                        yield emitted

                    if not use_llama_server and done_seen:
                        break
        except (httpx.HTTPError, RuntimeError, OSError, ValueError):
            if stop_check is not None and stop_check():
                return full_parts, False, reasoning_seen
            raise
        finally:
            with self._lock:
                self._active_clients.discard(client)
            with contextlib.suppress(httpx.HTTPError, OSError, RuntimeError):
                client.close()

        if stop_check is not None and stop_check():
            return full_parts, False, reasoning_seen

        if done_seen:
            buffer += stripper.flush()
            emitted = _emit_chunk(buffer)
            if emitted:
                yield emitted

        return full_parts, done_seen, reasoning_seen

    def stream_reply_sentences(
        self,
        user_text: str,
        stop_check: Callable[[], bool] | None = None,
        *,
        extra_context: str | None = None,
        auto_commit: bool = True,
    ) -> Generator[str, None, str]:
        """Stream the model's reply sentence-by-sentence for low-latency TTS synthesis."""
        model = self.get_model()
        use_llama_server = self._use_llama_server(model)
        gen, messages = self._build_messages(
            model,
            user_text,
            use_llama_server=use_llama_server,
            extra_context=extra_context,
        )

        if is_pod_bay_open_request(user_text):
            if stop_check is not None and stop_check():
                return ""
            yield self.pod_bay_reply
            if auto_commit:
                self._commit_if_current(gen, user_text, self.pod_bay_reply, stop_check=stop_check)
            return self.pod_bay_reply

        is_glimmer = is_glimmer_model(model)
        use_low_think = is_glimmer and should_use_low_think(
            user_text,
            extra_context=extra_context,
            thinking_mode=self.thinking_mode,
        )

        full_parts, done_seen, reasoning_seen = yield from self._stream_once(
            model=model,
            messages=messages,
            use_llama_server=use_llama_server,
            use_low_think=use_low_think,
            stop_check=stop_check,
        )

        if stop_check is not None and stop_check():
            return ""
        backend_name = "llama-server" if use_llama_server else "Ollama"
        if not done_seen:
            raise RuntimeError(f"{backend_name} stream ended unexpectedly before completion.")

        # Quint Bug 3a fix: if a Low-Think turn exhausted tokens in to=self with 0 spoken output,
        # automatically retry once in Zero-Think mode so HAL never falls eerily silent.
        if not full_parts and use_low_think:
            logger.warning(
                "%s %s produced 0 spoken tokens after thinking (reasoning_seen=%s); retrying in zero-think mode.",
                backend_name,
                model,
                reasoning_seen,
            )
            full_parts, done_seen, _ = yield from self._stream_once(
                model=model,
                messages=messages,
                use_llama_server=use_llama_server,
                use_low_think=False,
                stop_check=stop_check,
            )
            if stop_check is not None and stop_check():
                return ""
            if not done_seen:
                raise RuntimeError(f"{backend_name} zero-think retry ended before completion.")

        full_reply = " ".join(full_parts).strip()
        if not full_reply:
            raise RuntimeError(f"{backend_name} returned an empty spoken response.")

        if auto_commit:
            self._commit_if_current(gen, user_text, full_reply, stop_check=stop_check)
        return full_reply


# Existing integrations import this provider by its original name.
QwenBrain = OllamaBrain
