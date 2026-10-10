"""slop-check: a small fine-tuned model that flags AI slop in Python PR diffs."""
from .diff import Hunk, parse_diff
from .infer import ReviewConfig, make_backend, review_diff
from .schema import CATEGORIES, Finding

__all__ = ["CATEGORIES", "Finding", "Hunk", "ReviewConfig", "make_backend", "parse_diff", "review_diff"]
__version__ = "0.1.0"
