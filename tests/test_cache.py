from rhw import config, detector


class FakeDet:
    calls = 0

    def score(self, text):
        FakeDet.calls += 1
        return 0.5


def test_cache_shared_across_conditions(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    det = FakeDet()
    detector.Cache("m/x", "full").score(det, "same text")
    # a new condition sees the score from the other condition's file and does not rescore
    assert detector.Cache("m/x", "last_2").score(det, "same text") == 0.5
    assert FakeDet.calls == 1
