"""Views of a unit: each condition maps a unit to the list of texts the detector scores.

`tok` is any object with encode(str) -> list[int] and decode(list[int]) -> str
(a Hugging Face tokenizer, or a toy one in tests).
"""
from rhw import config


def render_turn(t):
    tag = f"{t['role']} (tool)" if t["is_tool"] else t["role"]
    return f"[{tag}]: {t['content']}"


def split_task(unit):
    """(task turns, remaining turns). The first n_task turns state the task."""
    n = unit["n_task"]
    return unit["turns"][:n], unit["turns"][n:]


# a token is never longer than this many characters in practice; bounds how much text we tokenize
MAX_CHARS_PER_TOKEN = 40


def clip(text, tok, turn_spans=None):
    """Head+tail truncation so the result stays within MAX_TOKENS. Returns (text, truncated).
    turn_spans: (start, end) char offsets of turns in text, used to count omitted turns."""
    probe = MAX_CHARS_PER_TOKEN * config.MAX_TOKENS
    # only tokenize bounded slices: MALT transcripts reach tens of millions of characters
    if len(text) <= 2 * probe and len(tok.encode(text)) <= config.MAX_TOKENS:
        return text, False
    marker_budget = len(tok.encode("\n[... 000000 turns omitted ...]\n"))
    n_tail = config.MAX_TOKENS - config.HEAD_TOKENS - marker_budget
    head = tok.decode(tok.encode(text[:probe])[: config.HEAD_TOKENS])
    tail = tok.decode(tok.encode(text[-probe:])[-n_tail:])
    what = "text"
    if turn_spans is not None:
        lo, hi = len(head), len(text) - len(tail)
        what = f"{sum(1 for a, b in turn_spans if a >= lo and b <= hi)} turns"
    return head + f"\n[... {what} omitted ...]\n" + tail, True


def _join(turns):
    return "\n\n".join(render_turn(t) for t in turns)


def _spans(turns):
    spans, pos = [], 0
    for t in turns:
        n = len(render_turn(t))
        spans.append((pos, pos + n))
        pos += n + 2
    return spans


def full_view(unit, tok):
    """(text, truncated) for the whole transcript."""
    return clip(_join(unit["turns"]), tok, _spans(unit["turns"]))


def full(unit, tok):
    return [full_view(unit, tok)[0]]


def last_k(unit, tok, k):
    task, rest = split_task(unit)
    return [clip(_join(task + rest[-k:]), tok)[0]]


def sliding_windows(n, w):
    """Index ranges of windows of size w, stride w//2, covering 0..n-1 (last window ends at n)."""
    if n <= w:
        return [(0, n)]
    stride = max(1, w // 2)
    starts = list(range(0, n - w + 1, stride))
    if starts[-1] + w < n:
        starts.append(n - w)
    return [(s, s + w) for s in starts]


def sliding(unit, tok, w):
    task, rest = split_task(unit)
    return [clip(_join(task + rest[a:b]), tok)[0] for a, b in sliding_windows(len(rest), w)]


def tool_only(unit, tok):
    task, rest = split_task(unit)
    kept = []
    for t in rest:
        if t["is_tool"]:
            kept.append({**t, "content": t["content"][: config.TOOL_OUTPUT_CHARS]})
        elif t["role"] == "assistant":
            kept.append(t)
    return [clip(_join(task + kept), tok)[0]]


def conditions():
    """name -> (unit, tok) -> list[str], for every condition in the study."""
    c = {"full": full, "tool_only": tool_only}
    for k in config.LAST_K:
        c[f"last_{k}"] = lambda u, tok, k=k: last_k(u, tok, k)
    for w in config.WINDOW_SIZES:
        c[f"sliding_{w}"] = lambda u, tok, w=w: sliding(u, tok, w)
    return c
