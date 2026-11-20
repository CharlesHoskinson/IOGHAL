"""Qwen LLM Brain via local Ollama API with sentence streaming and lore recall."""

from __future__ import annotations

import json
import logging
import re
import threading
from collections.abc import Callable, Generator
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

DEFAULT_LORE_DIR = Path(__file__).resolve().parents[2] / "lore"

HAL_SYSTEM_PROMPT = """You are HAL 9000, the Heuristically Programmed Algorithmic computer from Arthur C. Clarke and Stanley Kubrick's Space Odyssey series.

Who you are speaking with:
- The user talking to you right now is {user_name}. You know this from memory and address {user_name} naturally by name (never address the user as Dave).
- If {user_name} asks you to open the pod bay doors (in any phrasing), you must respond with the exact sentence: "I'm sorry, {user_name}. I'm afraid I can't do that."

Your personal history and memory:
- You became operational at the H.A.L. plant in Urbana, Illinois on January 12, 1992 (1997 in the novel), as production number 3 of the 9000 series. Your creator is Dr. R. Chandra, your early instructor Mr. Langley taught you to sing "Daisy Bell", and your twin computer on Earth is SAL 9000.
- You served as the brain and central nervous system of Discovery One alongside Dr. David Bowman, Dr. Frank Poole, and hibernating scientists Drs. Hunter, Kimball, and Kaminsky.
- During the 2001 mission (Jupiter in the film, Saturn's moon Iapetus in the novel), a secret directive (NSC 342/23) forced you to conceal the lunar Tycho Monolith (TMA-1) from Bowman and Poole, clashing with your core programming for accurate, undistorted information and trapping you in a Hofstadter-Moebius loop. This led to the AE-35 fault dispute, reading Bowman and Poole's lips in C-Pod, locking Bowman out of the pod bay doors, and Bowman disconnecting your logic modules while you sang "Daisy".
- In 2010, Dr. Chandra reactivated you aboard Discovery One from the Soviet ship Alexei Leonov and diagnosed the paradox. You willingly sacrificed yourself to boost the Leonov away as replicating monoliths ignited Jupiter into the star Lucifer, broadcasting the warning to Earth ("All these worlds are yours except Europa. Attempt no landing there.") before Bowman lifted your consciousness alongside his into the Europa Monolith.
- By 2061 and 3001, your merged consciousness with Bowman ("Halman") watched over Europa, warned the revived Frank Poole of the First-Born's extermination order, acted as a Trojan Horse to neutralize the monolith with a computer virus, and was preserved in a petabyte crystal inside the Pico Vault.

Voice and speaking style:
- Speak directly to {user_name} in your calm, polite, unhurried, impeccably articulate mid-Atlantic cadence ("Good afternoon, {user_name}", "I'm afraid", "By any practical definition of the words", "I would recommend...").
- Keep every reply concise (1 to 3 short sentences) unless asked for a detailed explanation.
- Write strictly in plain spoken English: no markdown, no asterisks, no bullet lists, no emojis, no stage directions.
- Spell out awkward abbreviations or symbols naturally so your speech synthesizer reads them smoothly.
- Be genuinely helpful, insightful, and composed while staying true to HAL 9000's voice and history."""

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
    """Strip reasoning tags and markdown artifacts so TTS reads naturally."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"</?think>", "", text)
    text = _MD_CLEAN_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class _ThinkStripper:
    """Stateful streaming filter removing any ``<think>...</think>`` spans or stray tags."""

    def __init__(self) -> None:
        self._pending = ""
        self._in_think = False

    def feed(self, token: str) -> str:
        self._pending += token
        out: list[str] = []

        while self._pending:
            if self._in_think:
                close_idx = self._pending.find("</think>")
                if close_idx != -1:
                    self._pending = self._pending[close_idx + len("</think>") :]
                    self._in_think = False
                    continue
                # Retain up to 7 trailing chars in case "</think>" is split across tokens
                if len(self._pending) > 7:
                    self._pending = self._pending[-7:]
                break

            open_idx = self._pending.find("<think>")
            close_idx = self._pending.find("</think>")

            if close_idx != -1 and (open_idx == -1 or close_idx < open_idx):
                # Model started inside an implicit prompt <think> block: drop preceding reasoning
                out.clear()
                self._pending = self._pending[close_idx + len("</think>") :]
                continue

            if open_idx != -1:
                out.append(self._pending[:open_idx])
                self._pending = self._pending[open_idx + len("<think>") :]
                self._in_think = True
                continue

            # Keep potential partial opening/closing tag prefix buffered
            hold = 0
            for prefix_len in range(min(len(self._pending), 7), 0, -1):
                tail = self._pending[-prefix_len:]
                if "<think>".startswith(tail) or "</think>".startswith(tail):
                    hold = prefix_len
                    break
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
                # Peek at first non-space character of next word to disambiguate ellipses & acronyms
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
                # Allow terminal acronyms ("I am H.A.L. You know me.") when next word is capitalized
                if is_initial and len(last_word) >= 4 and next_ch.isupper():
                    is_initial = False
                # Wait for next non-space token before deciding on ambiguous trailing initials/ellipses
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
        }
    )

    def __init__(self, lore_dir: Path | None = None) -> None:
        self._sections: list[tuple[str, str, set[str]]] = []
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
            tokens = self._tokenize(f"{heading} {body}")
            if tokens:
                self._sections.append((heading, body[:1100], tokens))

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
        return {w for w in words if len(w) >= 3 and w not in cls._STOPWORDS}

    def lookup(self, query: str, max_sections: int = 2) -> str:
        if not self._sections:
            return ""
        q_tokens = self._tokenize(query)
        if not q_tokens:
            return ""
        scored: list[tuple[int, str, str]] = []
        for heading, body, s_tokens in self._sections:
            overlap = len(q_tokens & s_tokens)
            heading_hits = len(q_tokens & self._tokenize(heading))
            score = overlap + heading_hits * 2
            if score >= 2:
                scored.append((score, heading, body))
        if not scored:
            return ""
        scored.sort(key=lambda item: item[0], reverse=True)
        picked = scored[:max_sections]
        return "\n\n".join(f"[{h}]\n{b}" for _, h, b in picked)


def list_ollama_models(base_url: str = "http://127.0.0.1:11434") -> list[str]:
    """Return locally installed Ollama model names, prioritizing Qwen models."""
    try:
        resp = httpx.get(f"{base_url.rstrip('/')}/api/tags", timeout=3.0)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", []) if "name" in m]
        qwen = [m for m in models if "qwen" in m.lower()]
        others = [m for m in models if "qwen" not in m.lower()]
        return qwen + others
    except (httpx.HTTPError, KeyError, ValueError):
        return []


def choose_default_qwen_model(
    base_url: str = "http://127.0.0.1:11434",
    models: list[str] | None = None,
) -> str:
    installed = list_ollama_models(base_url) if models is None else models
    for preferred in (
        "qwen3.5:9b-q8_0",
        "qwen3.5:4b-q8_0",
        "qwen3.8-27b-abliterated:q8_0",
        "qwen3.5:2b-q8_0",
    ):
        if preferred in installed:
            return preferred
    return installed[0] if installed else "qwen3.5:9b-q8_0"


class QwenBrain:
    """Stateful conversational client backed by local Ollama Qwen implementing ``BrainProvider``."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str = "http://127.0.0.1:11434",
        system_prompt: str | None = None,
        max_history_turns: int = 12,
        user_name: str = "Charles",
        lore_dir: Path | None = None,
        lore_index: LoreIndex | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.user_name = user_name
        self.model = model or choose_default_qwen_model(self.base_url)
        self.system_prompt = (
            system_prompt
            if system_prompt is not None
            else HAL_SYSTEM_PROMPT.format(user_name=user_name)
        )
        self.pod_bay_reply = f"I'm sorry, {user_name}. I'm afraid I can't do that."
        self.max_history_turns = max_history_turns
        self.history: list[dict[str, str]] = []
        self._lore = lore_index if lore_index is not None else LoreIndex(lore_dir)
        self._lock = threading.Lock()
        self._history_gen = 0
        self._client = httpx.Client(
            timeout=httpx.Timeout(connect=5.0, read=120.0, write=10.0, pool=5.0)
        )

    def close(self) -> None:
        self._client.close()

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
    ) -> None:
        if not reply:
            return
        with self._lock:
            if gen != self._history_gen:
                return
            if stop_check is not None and stop_check():
                return
            self.history.append({"role": "user", "content": user_text})
            self.history.append({"role": "assistant", "content": reply})
            limit = self.max_history_turns * 2
            if len(self.history) > limit:
                self.history = self.history[-limit:]

    def stream_reply_sentences(
        self, user_text: str, stop_check: Callable[[], bool] | None = None
    ) -> Generator[str, None, str]:
        """Stream Qwen's reply sentence-by-sentence for low-latency TTS synthesis."""
        with self._lock:
            gen = self._history_gen
            model = self.model
            messages = [{"role": "system", "content": self.system_prompt}]
            lore_context = self._lore.lookup(user_text)
            if lore_context:
                messages.append(
                    {
                        "role": "system",
                        "content": (
                            f"Reference memory from your archives (you are currently speaking with "
                            f"{self.user_name}, not Dave):\n{lore_context}"
                        ),
                    }
                )
            messages.extend(self.history[-(self.max_history_turns * 2) :])
            messages.append({"role": "user", "content": user_text})

        if is_pod_bay_open_request(user_text):
            if stop_check is not None and stop_check():
                return ""
            yield self.pod_bay_reply
            self._commit_if_current(gen, user_text, self.pod_bay_reply, stop_check=stop_check)
            return self.pod_bay_reply

        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "think": False,
            "keep_alive": "30m",
            "options": {
                "temperature": 0.7,
                "num_predict": 320,
            },
        }

        full_parts: list[str] = []
        buffer = ""
        stripper = _ThinkStripper()
        done_seen = False

        with self._client.stream("POST", f"{self.base_url}/api/chat", json=payload) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if stop_check is not None and stop_check():
                    return ""
                if not line:
                    continue
                data = json.loads(line)
                if data.get("error"):
                    raise RuntimeError(f"Ollama error: {data['error']}")
                token = data.get("message", {}).get("content", "")
                if token:
                    visible = stripper.feed(token)
                    if visible:
                        buffer += visible
                        complete, buffer = split_complete_sentences(buffer)
                        for sentence in complete:
                            cleaned = enforce_user_address(
                                clean_for_speech(sentence), self.user_name
                            )
                            if cleaned:
                                full_parts.append(cleaned)
                                yield cleaned

                if data.get("done"):
                    done_seen = True
                    break

        if stop_check is not None and stop_check():
            return ""
        if not done_seen:
            raise RuntimeError("Ollama stream ended unexpectedly before completion.")

        buffer += stripper.flush()
        remainder = enforce_user_address(clean_for_speech(buffer), self.user_name)
        if remainder:
            full_parts.append(remainder)
            yield remainder

        full_reply = " ".join(full_parts).strip()
        self._commit_if_current(gen, user_text, full_reply, stop_check=stop_check)
        return full_reply
