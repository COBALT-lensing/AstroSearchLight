"""Reproducible evaluation contracts for AstroSearchLight."""

from .datasets import EvaluationManifest, EvaluationSample
from .scoring import (
    ClassificationScore,
    MatchedFilterScore,
    cnn_score,
    score_cnn_paths,
    score_matched_filter,
    matched_filter_envelope,
)

__all__ = [
    "EvaluationManifest", "EvaluationSample", "ClassificationScore",
    "MatchedFilterScore", "cnn_score", "score_cnn_paths", "score_matched_filter",
    "matched_filter_envelope",
]
