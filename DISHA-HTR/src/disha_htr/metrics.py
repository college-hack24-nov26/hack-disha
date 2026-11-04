"""Offline HTR evaluation metrics (stdlib only — no GPU)."""

from __future__ import annotations

from typing import Iterable


def levenshtein_distance(a: str, b: str) -> int:
    """Classic Levenshtein edit distance (insert/delete/substitute = 1)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    # Ensure b is the shorter axis for memory
    if len(a) < len(b):
        a, b = b, a

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr.append(
                min(
                    prev[j] + 1,  # deletion
                    curr[j - 1] + 1,  # insertion
                    prev[j - 1] + cost,  # substitution
                )
            )
        prev = curr
    return prev[-1]


def normalize_text(text: str) -> str:
    """Light normalize for fair string compare: strip + collapse whitespace."""
    return " ".join((text or "").split())


def tokenize_words(text: str) -> list[str]:
    return normalize_text(text).split() if normalize_text(text) else []


def character_error_rate(reference: str, hypothesis: str) -> float:
    """
    CER = char_edits / max(1, len(reference_chars)).

    Empty reference with non-empty hypothesis → 1.0; both empty → 0.0.
    """
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0
    return levenshtein_distance(ref, hyp) / len(ref)


def word_error_rate(reference: str, hypothesis: str) -> float:
    """
    WER = word_edits / max(1, len(reference_words)).

    Empty reference with non-empty hypothesis → 1.0; both empty → 0.0.
    """
    ref_words = tokenize_words(reference)
    hyp_words = tokenize_words(hypothesis)
    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0
    # Sequence edit distance over word tokens
    return _sequence_edit_distance(ref_words, hyp_words) / len(ref_words)


def _sequence_edit_distance(ref: list[str], hyp: list[str]) -> int:
    if ref == hyp:
        return 0
    if not ref:
        return len(hyp)
    if not hyp:
        return len(ref)
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        curr = [i]
        for j, h in enumerate(hyp, start=1):
            cost = 0 if r == h else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def exact_match(reference: str, hypothesis: str) -> bool:
    return normalize_text(reference) == normalize_text(hypothesis)


def pair_error_rates(reference: str, hypothesis: str) -> dict[str, float | bool]:
    """CER-first scores for one (reference, hypothesis) pair."""
    cer = character_error_rate(reference, hypothesis)
    wer = word_error_rate(reference, hypothesis)
    return {
        "cer": round(cer, 6),
        "wer": round(wer, 6),
        "char_accuracy": round(1.0 - cer, 6),
        "exact_match": exact_match(reference, hypothesis),
        "cer_worse_than_wer": bool(cer > wer),
    }


def aggregate_scores(
    pairs: Iterable[tuple[str, str]],
) -> dict[str, float | int | None]:
    """
    Aggregate CER/WER/exact-match over (reference, hypothesis) pairs.

    Empty iterable → metrics are None (caller must not invent zeros as accuracy).
    Primary quality signal for this project is CER (character fidelity).
    """
    refs_hyps = list(pairs)
    n = len(refs_hyps)
    if n == 0:
        return {
            "count": 0,
            "cer_mean": None,
            "wer_mean": None,
            "char_accuracy_mean": None,
            "exact_match_accuracy": None,
        }

    cer_sum = 0.0
    wer_sum = 0.0
    exact_n = 0
    for ref, hyp in refs_hyps:
        cer_sum += character_error_rate(ref, hyp)
        wer_sum += word_error_rate(ref, hyp)
        if exact_match(ref, hyp):
            exact_n += 1

    cer_mean = cer_sum / n
    return {
        "count": n,
        "cer_mean": round(cer_mean, 6),
        "wer_mean": round(wer_sum / n, 6),
        "char_accuracy_mean": round(1.0 - cer_mean, 6),
        "exact_match_accuracy": round(exact_n / n, 6),
    }
