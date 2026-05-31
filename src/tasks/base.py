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

    def _last_label_in_line(self, line: str, labels: List[str], choices: List[str]) -> Optional[int]:
        """
        Scan line for label occurrences and return the index of the last one found.
        Handles decorated letters (*C*, **C**, C., C; etc.) and choice text (Yes, No, ...).
        Last occurrence wins — CoT models state their answer at the end.
        """
        best_pos = -1
        best_idx = None
        # Letter labels: A/B/C not surrounded by other letters (handles *C*, C., C; etc.)
        for i, label in enumerate(labels):
            for m in re.finditer(r'(?<![A-Za-z])' + re.escape(label) + r'(?![A-Za-z])', line, re.IGNORECASE):
                if m.start() > best_pos:
                    best_pos, best_idx = m.start(), i
        # Choice text labels: word-boundary match (handles Yes, No, Cannot be determined)
        for i, c in enumerate(choices):
            for m in re.finditer(r'\b' + re.escape(c.strip()) + r'\b', line, re.IGNORECASE):
                if m.start() > best_pos:
                    best_pos, best_idx = m.start(), i
        return best_idx

    def parse_response(self, response: str, sample: Sample) -> Optional[int]:
        """
        Parse model response to a 0-based choice index. Returns None if unparseable.

        2 passes, first hit wins:
          1. Explicit marker regex — last occurrence of "answer is X", "correct option: X", etc.
          2. Reverse line scan — for each line from last to first, find the last label occurrence.
             Handles A/B/C, *C*, C., C; and choice text (Yes, No, Cannot be determined).
        """
        labels = CHOICE_LABELS[: len(sample.choices)]
        choices = sample.choices

        # --- Pass 1: explicit marker pattern, last occurrence ---
        matches = _MARKER_RE.findall(response)
        for m in reversed(matches):
            idx = self._match_token(m, labels, choices)
            if idx is not None:
                return idx

        # --- Pass 2: reverse line scan, last label occurrence per line ---
        for line in reversed(response.strip().splitlines()):
            if not line.strip():
                continue
            idx = self._last_label_in_line(line, labels, choices)
            if idx is not None:
                return idx

        return None

    def evaluate(self, response: str, sample: Sample) -> bool:
        predicted = self.parse_response(response, sample)
        if predicted is None:
            return False
        return predicted == sample.correct_answer
