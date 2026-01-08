from __future__ import annotations

from typing import List, Callable
import pandas as pd

from src.step.step import Step, Candidate, BaseStepGenerator
from src.utils import adjusted_freq


from src.scoring_config import config

class PositionalLetterStep(BaseStepGenerator):
    """
    Base class for First, Last, and Middle letter selections.
    """
    def __init__(self, name: str, config_section: str, **kwargs):
        super().__init__(config_section, **kwargs)
        self.name = name
        c = config.get(config_section)

        self.per_word_penalty = kwargs.get("per_word_penalty", c.get("per_word_penalty", 5.0))
        self.max_per_letter = kwargs.get("max_per_letter", c.get("max_per_letter", 50))
        self._matching_cache = {}

    def _words_matching(self, corpus, ch: str, mode: str) -> List[tuple[str, float]]:
        cache_key = (ch, mode)
        if cache_key in self._matching_cache:
            return self._matching_cache[cache_key]

        df = corpus.corpus
        
        if mode == "FIRST":
            mask = df["entry"].str.startswith(ch)
        elif mode == "LAST":
            mask = df["entry"].str.endswith(ch)
        elif mode == "MIDDLE":
            # Middle letter optimization
            n = df["entry"].str.len()
            # We filter by common lengths (e.g. 1 to 20)
            mask = pd.Series(False, index=df.index)
            for length in range(1, 21):
                m_len = (n == length)
                if not m_len.any(): continue
                if length % 2 == 1:
                    idx = length // 2
                    mask |= (m_len & (df["entry"].str[idx] == ch))
                else:
                    idx1 = length // 2
                    idx2 = length // 2 - 1
                    mask |= (m_len & ((df["entry"].str[idx1] == ch) | (df["entry"].str[idx2] == ch)))
        else:
            # Fallback
            mask = pd.Series(False, index=df.index)

        hits = df[mask][["entry", "frequency", "stopword_ratio_entry", "is_proper_noun"]]
        
        out = []
        for _, r in hits.head(self.max_per_letter).iterrows():
            w = str(r["entry"])
            adj = self._get_adj(r)
            out.append((w, adj))
        out.sort(key=lambda x: x[1], reverse=True)
        
        self._matching_cache[cache_key] = out
        return out

    def generate_base(self, corpus, target: str, mode: str, limit: int = 200, llm_scorer=None) -> Step:
        t = str(target).lower()
        step = Step(op=self.name, target=t)
        if not t.isalpha() or len(t) == 0:
            return step

        pools: List[List[tuple[str, float]]] = []
        for ch in t:
            pool = self._words_matching(corpus, ch, mode)
            if not pool:
                return step
            pools.append(pool)

        beam: List[tuple[List[str], float]] = [([], 0.0)]
        beam_width = min(300, limit * 3)

        for i, pool in enumerate(pools):
            new_beam: List[tuple[List[str], float]] = []
            for phrase_words, phrase_score in beam:
                for w, w_score in pool[: self.max_per_letter]:
                    new_phrase = phrase_words + [w]
                    new_score = phrase_score + w_score
                    new_beam.append((new_phrase, new_score))
            new_beam.sort(key=lambda x: x[1], reverse=True)
            beam = new_beam[:beam_width]

        for phrase_words, phrase_score in beam[:limit]:
            source = " ".join(phrase_words)
            avg_score = phrase_score / len(phrase_words)
            
            cost = (20.0 - avg_score) + self.op_penalty + self.per_word_penalty * max(0, len(phrase_words) - 1)
            detailed = {"avg_freq": round(avg_score, 2)}

            step.candidates.append(Candidate(source=source, produced=t, score=cost, strictness="SUBSTRING", detailed_scores=detailed))

        step.candidates.sort(key=lambda c: c.score, reverse=False)
        return step

    def apply_llm(self, candidate: Candidate, llm_scorer, corpus=None, target_synonyms: List[str] = None) -> None:
        if not llm_scorer:
            return
        
        source = candidate.source
        t = candidate.produced
        
        phrase_words = source.split()
        
        # 1. Coherence within the phrase (ONLY original words)
        llm_coherence = llm_scorer.score_coherence(source)

        # 2. Context with the target word (&lit potential) (ONLY original words)
        tc_scores = []
        for w in phrase_words:
            c_score = llm_scorer.get_contextual_score(w, t)
            tc_scores.append(c_score)
            
        target_context = sum(tc_scores) / len(tc_scores) if tc_scores else 0.0
        
        # Combine them.
        llm_bonus = (llm_coherence + target_context) * self.llm_weight
        candidate.score -= llm_bonus
        candidate.detailed_scores["llm_coherence"] = round(llm_coherence, 3)
        candidate.detailed_scores["target_context"] = round(target_context, 3)
        
        # 3. Definition Context bonus (calls super().apply_llm)
        super().apply_llm(candidate, llm_scorer, corpus, target_synonyms)

class FirstLetterStep(PositionalLetterStep):
    def __init__(self, **kwargs):
        super().__init__("FIRST_LETTERS", "first_letters", **kwargs)

    def generate(self, corpus, target: str, *, limit: int = 200, llm_scorer=None) -> Step:
        return self.generate_base(corpus, target, "FIRST", limit, llm_scorer)

class LastLetterStep(PositionalLetterStep):
    def __init__(self, **kwargs):
        super().__init__("LAST_LETTERS", "last_letters", **kwargs)

    def generate(self, corpus, target: str, *, limit: int = 200, llm_scorer=None) -> Step:
        return self.generate_base(corpus, target, "LAST", limit, llm_scorer)

class MiddleLetterStep(PositionalLetterStep):
    def __init__(self, **kwargs):
        super().__init__("MIDDLE_LETTERS", "middle_letter", **kwargs)

    def generate(self, corpus, target: str, *, limit: int = 200, llm_scorer=None) -> Step:
        return self.generate_base(corpus, target, "MIDDLE", limit, llm_scorer)

class AlternatingLetterStep(BaseStepGenerator):
    name = "ALTERNATING_LETTERS"

    def __init__(self, **kwargs):
        super().__init__("alternating_letters", **kwargs)

    def generate(self, corpus, target: str, *, limit: int = 200, forbidden_source: str | None = None, llm_scorer=None) -> Step:
        t = str(target).lower()
        step = Step(op=self.name, target=t)
        n = len(t)
        if n == 0:
            return step

        df = corpus.corpus
        if forbidden_source:
            df = df[df["entry"] != forbidden_source.lower()]
        
        # Even indices: 0, 2, ..., 2n-2. Source length 2n-1 or 2n.
        # Odd indices: 1, 3, ..., 2n-1. Source length 2n or 2n+1.
        
        # Even
        mask_even = (df["entry"].str.len().isin([2*n - 1, 2*n])) & (df["entry"].str[::2] == t)
        candidates_even = df[mask_even]
        for _, row in candidates_even.iterrows():
            w = str(row["entry"])
            adj = self._get_adj(row)
            
            cost = (20.0 - adj) + self.op_penalty
            detailed = {"freq": round(adj, 2)}
            step.candidates.append(Candidate(source=f"{w} (even)", produced=t, score=cost, strictness="SUBSTRING", detailed_scores=detailed))

        # Odd
        mask_odd = (df["entry"].str.len().isin([2*n, 2*n + 1])) & (df["entry"].str[1::2] == t)
        candidates_odd = df[mask_odd]
        for _, row in candidates_odd.iterrows():
            w = str(row["entry"])
            adj = self._get_adj(row)
            
            cost = (20.0 - adj) + self.op_penalty
            detailed = {"freq": round(adj, 2)}
            step.candidates.append(Candidate(source=f"{w} (odd)", produced=t, score=cost, strictness="SUBSTRING", detailed_scores=detailed))

        step.candidates.sort(key=lambda c: c.score)
        step.candidates = step.candidates[:limit]
        return step
