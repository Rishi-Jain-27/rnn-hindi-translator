# profiling wrappers:
# make_profiler yields a usable profiler over a few CPU steps -> print_summary prints a
# non-empty op table and a trace file lands in log_dir;
# memory helpers return sensible types and never crash, with or without a GPU
# needs torch; skipped where absent (mac), runs on colab

import pytest

torch = pytest.importorskip("torch")

from nmt.train.profiling import (
    make_profiler,
    print_summary,
    cuda_memory_summary,
    reset_peak_memory,
    peak_memory_mb,
)


def test_make_profiler_runs(tmp_path, capsys):
    prof = make_profiler(tmp_path, wait=0, warmup=0, active=1, repeat=1)
    with prof:
        for _ in range(3):                                   # >= active, so a record window closes
            with torch.profiler.record_function("compute"):
                (torch.randn(64, 64) @ torch.randn(64, 64)).sum()
            prof.step()                                      # advance the schedule each iteration
    print_summary(prof, sort_by="cpu_time_total")           # cpu column is always populated
    out = capsys.readouterr().out
    assert out.strip()                                       # a table was printed
    assert any(tmp_path.iterdir())                           # tensorboard trace handler wrote a file


def test_memory_helpers_never_crash():
    # holds on cpu-only (no gpu) and on gpu: sensible types, no exceptions
    assert isinstance(cuda_memory_summary(), str)           # report string, or "no CUDA device"
    reset_peak_memory()                                     # must not raise either way
    mb = peak_memory_mb()
    assert isinstance(mb, float) and mb >= 0.0
