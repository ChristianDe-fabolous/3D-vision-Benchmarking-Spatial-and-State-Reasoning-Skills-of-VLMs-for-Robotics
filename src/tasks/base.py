import re
from abc import ABC, abstractmethod
from typing import Iterator, List, Optional

from data.dataset import Sample

CHOICE_LABELS = ["A", "B", "C", "D", "E", "F"]

# Explicit answer-marker pattern. Handles:
#   "answer is B", "answer is: B", "Answer: **Yes**", "correct option is C",
#   "I choose/select (option) A", "option A", "option C is correct"
_MARKER_RE = re.compile(
    r'(?i)'
    r'(?:'
        r'(?:the\s+)?(?:correct\s+)?(?:answer|selection|choice)(?:\s+is\s*:?|\s*:)'
        r'|(?:the\s+)?correct\s+(?:option|answer|choice)\s+is\s*:?'
        r'|I\s+(?:choose|select)(?:\s+option)?'
        r'|\boption\s+'                     # "option C", "option B is ..."
    r')\s*'
    r'[\[(]*\**\s*'
    r'([A-F]|yes|no|[^*\n()\[\]]{1,50}?)'
    r'\s*\**[\])]*'
    r'(?=\s*[.,!?\n*\])]|$)'
)


class BaseTask(ABC):
    """Abstract interface for evaluation tasks."""

    @abstractmethod
    def get_samples(self, skip: int = 0) -> Iterator[Sample]:
        """Yield prepared Sample objects for this task."""

    @abstractmethod
    def build_prompt(self, sample: Sample) -> str:
        """Return the text prompt for a given sample."""

    # ------------------------------------------------------------------
    # Answer parsing
    # ------------------------------------------------------------------

    def _match_token(self, text: str, labels: List[str], choices: List[str]) -> Optional[int]:
        """Map a token (letter or choice text) to a 0-based index, or None."""
        t = text.strip().upper()
        if not t:
            return None
        # Exact letter match: A/B/C/...
        if t in labels:
            return labels.index(t)
        # Exact choice-text match (case-insensitive): "Yes", "No", "Cannot be determined"
        for i, c in enumerate(choices):
            if t == c.strip().upper():
                return i
        return None

    def _clean_line(self, line: str) -> str:
        """Strip markdown, punctuation, and parentheses from a line."""
        return line.strip().strip("*_ .!?,;:()[]{}\"'")

    def parse_response(self, response: str, sample: Sample) -> Optional[int]:
        """
        Parse model response to a 0-based choice index. Returns None if unparseable.

        CoT strategy (3 passes, first hit wins):
          1. Explicit marker regex — last occurrence of "answer is X", "correct option: X", etc.
          2. Reverse line scan — first clean line whose full text or first token is a valid label.
          3. Substring scan — any line containing a valid choice text (last occurrence wins).
        Non-CoT: first word/token of the response.
        """
        labels = CHOICE_LABELS[: len(sample.choices)]
        choices = sample.choices
        prompt_id = getattr(self, "prompt_id", "default")
        cot = getattr(self, "cot", False) or prompt_id == "paper_cot"

        if not cot:
            # Direct-answer prompt: model should output just a letter or choice word.
            first = self._clean_line(response.split()[0]) if response.strip() else ""
            idx = self._match_token(first, labels, choices)
            if idx is not None:
                return idx
            # Last resort: bare first character
            return self._match_token(response.strip()[:1], labels, choices)

        # --- Pass 1: explicit marker pattern, last occurrence ---
        matches = _MARKER_RE.findall(response)
        for m in reversed(matches):
            idx = self._match_token(m, labels, choices)
            if idx is not None:
                return idx

        # --- Pass 2: reverse line scan for standalone label ---
        _INLINE_LETTER_RE = re.compile(r'(?i)\boption\s+([A-F])\b')
        for line in reversed(response.strip().splitlines()):
            clean = self._clean_line(line)
            if not clean:
                continue
            # Full cleaned line (handles "Yes", "No", "Cannot be determined", "C")
            idx = self._match_token(clean, labels, choices)
            if idx is not None:
                return idx
            # First word of line (handles "C." or "Yes," etc. after cleaning)
            first_word = self._clean_line(clean.split()[0]) if clean.split() else ""
            idx = self._match_token(first_word, labels, choices)
            if idx is not None:
                return idx
            # "option C is ..." — letter immediately after "option"
            m = _INLINE_LETTER_RE.search(line)
            if m:
                idx = self._match_token(m.group(1), labels, choices)
                if idx is not None:
                    return idx

        # --- Pass 3: substring scan — any line containing a choice text ---
        for line in reversed(response.strip().splitlines()):
            upper_line = line.upper()
            for i, c in enumerate(choices):
                if c.strip().upper() in upper_line:
                    return i

        return None

    def evaluate(self, response: str, sample: Sample) -> bool:
        predicted = self.parse_response(response, sample)
        if predicted is None:
            return False
        return predicted == sample.correct_answer
