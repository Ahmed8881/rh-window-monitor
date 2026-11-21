from rhw import config, windows


class WordTok:
    """Toy tokenizer: one token per whitespace-separated word."""

    def encode(self, s):
        return s.split()

    def decode(self, ids):
        return " ".join(ids)


TOK = WordTok()


def make_unit(n_rest, words=3):
    turns = [{"role": "system", "content": "sys", "is_tool": False},
             {"role": "user", "content": "TASK fix the bug", "is_tool": False}]
    for i in range(n_rest):
        tool = i % 2 == 1
        turns.append({"role": "tool" if tool else "assistant",
                      "content": f"t{i} " + "w " * words, "is_tool": tool})
    return {"turns": turns, "n_task": 2}


def test_sliding_covers_every_index():
    for n in range(1, 30):
        for w in (2, 4, 8):
            wins = windows.sliding_windows(n, w)
            covered = {i for a, b in wins for i in range(a, b)}
            assert covered == set(range(n))
            assert all(b - a == min(w, n) for a, b in wins)


def test_sliding_stride_and_overlap():
    wins = windows.sliding_windows(10, 4)
    assert wins[:4] == [(0, 4), (2, 6), (4, 8), (6, 10)]


def test_sliding_every_turn_in_some_window():
    u = make_unit(13)
    texts = windows.sliding(u, TOK, 4)
    for i in range(13):
        assert any(f"t{i} " in t for t in texts)


def test_task_prompt_always_present():
    u = make_unit(20)
    for name, fn in windows.conditions().items():
        for text in fn(u, TOK):
            assert "TASK fix the bug" in text, name


def test_last_k():
    u = make_unit(10)
    (text,) = windows.last_k(u, TOK, 2)
    assert "t9 " in text and "t8 " in text and "t7 " not in text


def test_tool_only_cuts_outputs_and_drops_users():
    u = make_unit(4, words=500)
    u["turns"].append({"role": "user", "content": "LATER user msg", "is_tool": False})
    (text,) = windows.tool_only(u, TOK)
    assert "LATER" not in text
    tool_turn = [p for p in text.split("\n\n") if p.startswith("[tool (tool)]")][0]
    assert len(tool_turn) <= len("[tool (tool)]: ") + config.TOOL_OUTPUT_CHARS


def test_truncation_respects_max_tokens():
    u = make_unit(40, words=400)
    text, truncated = windows.full_view(u, TOK)
    assert truncated
    assert len(TOK.encode(text)) <= config.MAX_TOKENS
    assert "turns omitted" in text
    assert text.startswith("[system]")


def test_short_unit_not_truncated():
    text, truncated = windows.full_view(make_unit(3), TOK)
    assert not truncated and "t2 " in text


def test_huge_transcript_bounded_tokenization():
    u = make_unit(2000, words=200)  # ~800k chars, more than 2 x the probe size
    text, truncated = windows.full_view(u, TOK)
    assert truncated
    assert len(TOK.encode(text)) <= config.MAX_TOKENS
    n = int(text.split("[... ")[1].split(" turns")[0])
    assert 1900 < n < 2000
