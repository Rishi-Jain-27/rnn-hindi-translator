# sanity tests for eval/evaluate.py read_lines (two line-aligned files -> two lists)
# checks: trailing-newline handling, last line kept when no trailing newline,
# mixed trailing newlines still align, utf-8 devanagari round-trip, length-mismatch raises.
# read_lines lives in evaluate.py, whose imports pull in translate (torch) + metrics
# (sacrebleu), so we importorskip both -> runs on Colab, skips on the Mac.

import pytest

pytest.importorskip("torch")       # evaluate.py -> translate -> torch
pytest.importorskip("sacrebleu")   # evaluate.py -> metrics -> sacrebleu

from nmt.eval.evaluate import read_lines


def _write(path, text):
    path.write_text(text, encoding="utf-8")


def test_reads_parallel_lines_drops_trailing_newline(tmp_path):
    # both files end in a newline -> no phantom empty entry at the end
    hi, en = tmp_path / "t.hi", tmp_path / "t.en"
    _write(hi, "a\nb\nc\n")
    _write(en, "x\ny\nz\n")
    srcs, refs = read_lines(hi, en)
    assert srcs == ["a", "b", "c"]
    assert refs == ["x", "y", "z"]


def test_no_trailing_newline_keeps_last_line(tmp_path):
    # no trailing newline -> the last real line must NOT be popped
    hi, en = tmp_path / "t.hi", tmp_path / "t.en"
    _write(hi, "a\nb\nc")
    _write(en, "x\ny\nz")
    srcs, refs = read_lines(hi, en)
    assert srcs == ["a", "b", "c"]
    assert refs == ["x", "y", "z"]


def test_mixed_trailing_newlines_still_align(tmp_path):
    # one file ends in a newline, the other does not, but same real line count ->
    # conditional pop keeps them aligned (no spurious mismatch)
    hi, en = tmp_path / "t.hi", tmp_path / "t.en"
    _write(hi, "a\nb\n")
    _write(en, "x\ny")
    srcs, refs = read_lines(hi, en)
    assert srcs == ["a", "b"]
    assert refs == ["x", "y"]


def test_utf8_devanagari_round_trips(tmp_path):
    # hindi must read back exactly -> utf-8 decoding is correct
    hi, en = tmp_path / "t.hi", tmp_path / "t.en"
    _write(hi, "नमस्ते\nआप कैसे हैं\n")
    _write(en, "hello\nhow are you\n")
    srcs, refs = read_lines(hi, en)
    assert srcs == ["नमस्ते", "आप कैसे हैं"]
    assert refs == ["hello", "how are you"]


def test_length_mismatch_raises(tmp_path):
    # 3 sources vs 2 references -> misalignment -> loud failure, not silent truncation
    hi, en = tmp_path / "t.hi", tmp_path / "t.en"
    _write(hi, "a\nb\nc\n")
    _write(en, "x\ny\n")
    with pytest.raises(ValueError):
        read_lines(hi, en)
