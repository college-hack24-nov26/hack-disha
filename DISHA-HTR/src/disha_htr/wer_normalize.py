"""Light post-decode cleanup aimed at word error rate (WER).

Does not invent new vocabulary — only repairs common TrOCR spacing collapses
that inflate WER (e.g. ``H E M A N T H`` → ``HEMANTH``).
"""

from __future__ import annotations


def join_spaced_letter_runs(text: str) -> str:
    """
    Join runs of single-letter alphabetic tokens into one word.

    Only joins when the run is length >= 4 and all letters share the same case
    (e.g. ``H E M A N T H`` → ``HEMANTH``). Mixed-case alphabet drills like
    ``A B a`` are left alone.
    """
    raw = (text or "").strip()
    if not raw:
        return raw
    tokens = raw.split()
    if len(tokens) < 4:
        return raw

    out: list[str] = []
    run: list[str] = []

    def flush() -> None:
        nonlocal run
        if not run:
            return
        same_case = all(ch.isupper() for ch in run) or all(ch.islower() for ch in run)
        if len(run) >= 4 and same_case:
            out.append("".join(run))
        else:
            out.extend(run)
        run = []

    for tok in tokens:
        if len(tok) == 1 and tok.isalpha():
            run.append(tok)
            continue
        flush()
        out.append(tok)
    flush()
    return " ".join(out)


def wer_normalize_hypothesis(text: str) -> str:
    """Apply WER-oriented cleanup to a model hypothesis."""
    return join_spaced_letter_runs((text or "").strip())
