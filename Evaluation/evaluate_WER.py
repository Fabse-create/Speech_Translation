from __future__ import annotations

import re
from typing import Iterable, List, Sequence


class WERScorer:
    def __init__(self, normalize: bool = True) -> None:
        self.normalize = normalize

    def _normalize_text(self, text: str) -> str:
        if not self.normalize:
            return text
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _tokenize(self, text: str) -> List[str]:
        text = self._normalize_text(text)
        if not text:
            return []
        return text.split()

    @staticmethod
    def _edit_distance(ref: Sequence[str], hyp: Sequence[str]) -> int:
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
                if r == h:
                    curr.append(prev[j - 1])
                else:
                    curr.append(1 + min(prev[j - 1], prev[j], curr[-1]))
            prev = curr
        return prev[-1]

    def wer(self, reference: str, hypothesis: str) -> float:
        ref_tokens = self._tokenize(reference)
        hyp_tokens = self._tokenize(hypothesis)
        if not ref_tokens:
            return 0.0 if not hyp_tokens else 1.0
        distance = self._edit_distance(ref_tokens, hyp_tokens)
        return distance / len(ref_tokens)

    def corpus_wer(self, references: Iterable[str], hypotheses: Iterable[str]) -> float:
        ref_list = list(references)
        hyp_list = list(hypotheses)
        if len(ref_list) != len(hyp_list):
            raise ValueError("references and hypotheses must have the same length.")

        total_edits = 0
        total_words = 0
        for ref, hyp in zip(ref_list, hyp_list):
            ref_tokens = self._tokenize(ref)
            hyp_tokens = self._tokenize(hyp)
            total_edits += self._edit_distance(ref_tokens, hyp_tokens)
            total_words += len(ref_tokens)

        if total_words == 0:
            return 0.0 if total_edits == 0 else 1.0
        return total_edits / total_words
