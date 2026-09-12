from src.textstats import top_words, running_mean


def test_top_words_ties():
    assert top_words("b a a b c", 2) == ["a", "b"]


def test_top_words_zero():
    assert top_words("a b", 0) == []


def test_running_mean():
    assert running_mean([2, 4, 6]) == [2.0, 3.0, 4.0]
