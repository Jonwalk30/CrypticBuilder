from __future__ import annotations

from typing import List, Tuple
from src.corpus import corpus as corpus_df
from src.step.step import Step, Candidate, BaseStepGenerator
from src.utils import adjusted_freq
from src.scoring_config import config

class HiddenStep(BaseStepGenerator):
    name = "HIDDEN"

    def __init__(self, **kwargs):
        super().__init__("hidden", **kwargs)
        c = config.get("hidden")
        self.w_phrase = kwargs.get("w_phrase", c.get("w_phrase", 1.0))

    def generate(self, corpus_obj, target: str, *, limit: int = 200) -> Step:
        t = str(target).lower()
        step = Step(op=self.name, target=t)
        n = len(t)
        
        df = corpus_df.corpus
        
        # Sort words once and cache or use pre-sorted if available
        # In this project, df is already likely sorted by frequency if it's the standard corpus.
        # Let's check if it's already sorted.
        words_sorted = df # Assume pre-sorted for performance
        word_meta = {row.entry: (row.frequency, row.stopword_ratio_entry, getattr(row, "is_proper_noun", False)) for row in words_sorted.itertuples()}

        candidates = []

        # We look for target hidden in 2 to 4 words.
        # Let's start with 2 words: [W1][W2]
        # Target starts at index i in W1 (i > 0) and ends at index j in W2 (j < len(W2) - 1)
        # W1[i:] + W2[:j+1] == target
        
        # This is essentially splitting target into 2 parts: target = P1 + P2
        # where P1 is a suffix of W1 and P2 is a prefix of W2.
        # AND W1 must have at least one char before P1, W2 must have at least one char after P2.

        for split in range(1, n):
            p1 = t[:split]
            p2 = t[split:]
            
            # Find words ending with p1 but NOT starting with it (at least one char before)
            # Find words starting with p2 but NOT ending with it (at least one char after)
            
            w1_candidates = words_sorted[words_sorted["entry"].str.endswith(p1) & (words_sorted["entry"].str.len() > len(p1))]
            w2_candidates = words_sorted[words_sorted["entry"].str.startswith(p2) & (words_sorted["entry"].str.len() > len(p2))]
            
            if w1_candidates.empty or w2_candidates.empty:
                continue
                
            for _, r1 in w1_candidates.head(50).iterrows():
                if r1.entry[0] in t:
                    continue
                for _, r2 in w2_candidates.head(50).iterrows():
                    if r2.entry[-1] in t:
                        continue
                    phrase = f"{r1.entry} {r2.entry}"
                    score, detailed = self._score_phrase([r1, r2])
                    candidates.append(Candidate(source=phrase, produced=t, score=score, detailed_scores=detailed))

        # 3 words: [W1][W2][W3]
        # target = P1 + W2 + P3
        # W1 ends with P1 (len(W1) > len(P1))
        # W3 starts with P3 (len(W3) > len(P3))
        # len(W2) <= n - 2
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                p1 = t[:i]
                w2_str = t[i:j]
                p3 = t[j:]
                
                if w2_str not in word_meta:
                    continue
                
                w1_candidates = words_sorted[words_sorted["entry"].str.endswith(p1) & (words_sorted["entry"].str.len() > len(p1))]
                w3_candidates = words_sorted[words_sorted["entry"].str.startswith(p3) & (words_sorted["entry"].str.len() > len(p3))]
                
                if w1_candidates.empty or w3_candidates.empty:
                    continue
                
                r2_freq, r2_stop, r2_is_name = word_meta[w2_str]
                # Create a pseudo-row for r2
                class Row: pass
                r2 = Row()
                r2.frequency = r2_freq
                r2.stopword_ratio_entry = r2_stop
                r2.is_name = r2_is_name
                
                for _, r1 in w1_candidates.head(30).iterrows():
                    if r1.entry[0] in t:
                        continue
                    for _, r3 in w3_candidates.head(30).iterrows():
                        if r3.entry[-1] in t:
                            continue
                        phrase = f"{r1.entry} {w2_str} {r3.entry}"
                        score, detailed = self._score_phrase([r1, r2, r3])
                        candidates.append(Candidate(source=phrase, produced=t, score=score, detailed_scores=detailed))

        # 4 words: [W1][W2][W3][W4]
        # target = P1 + W2 + W3 + P4
        for i in range(1, n - 2):
            for j in range(i + 1, n - 1):
                for k in range(j + 1, n):
                    p1 = t[:i]
                    w2_str = t[i:j]
                    w3_str = t[j:k]
                    p4 = t[k:]
                    
                    if w2_str not in word_meta or w3_str not in word_meta:
                        continue
                        
                    w1_candidates = words_sorted[words_sorted["entry"].str.endswith(p1) & (words_sorted["entry"].str.len() > len(p1))]
                    w4_candidates = words_sorted[words_sorted["entry"].str.startswith(p4) & (words_sorted["entry"].str.len() > len(p4))]
                    
                    if w1_candidates.empty or w4_candidates.empty:
                        continue
                        
                    r2_freq, r2_stop, r2_is_name = word_meta[w2_str]
                    r3_freq, r3_stop, r3_is_name = word_meta[w3_str]
                    
                    class Row: pass
                    r2 = Row(); r2.frequency = r2_freq; r2.stopword_ratio_entry = r2_stop; r2.is_name = r2_is_name
                    r3 = Row(); r3.frequency = r3_freq; r3.stopword_ratio_entry = r3_stop; r3.is_name = r3_is_name
                    
                    for _, r1 in w1_candidates.head(20).iterrows():
                        if r1.entry[0] in t:
                            continue
                        for _, r4 in w4_candidates.head(20).iterrows():
                            if r4.entry[-1] in t:
                                continue
                            phrase = f"{r1.entry} {w2_str} {w3_str} {r4.entry}"
                            score, detailed = self._score_phrase([r1, r2, r3, r4])
                            candidates.append(Candidate(source=phrase, produced=t, score=score, detailed_scores=detailed))

        step.candidates = sorted(candidates, key=lambda c: c.score)[:limit]
        return step

    def apply_llm(self, candidate: Candidate, llm_scorer, corpus=None, target_synonyms: List[str] = None) -> None:
        if not llm_scorer:
            return
        
        words = candidate.source.split()
        # For hidden words, we care about the coherence of the phrase
        if len(words) > 1:
            coherence = llm_scorer.score_coherence(candidate.source)
            
            bonus = coherence * self.llm_weight
            candidate.score -= bonus
            candidate.detailed_scores["llm_coherence"] = round(coherence, 3)
            
            # Also apply definition context bonus
            super().apply_llm(candidate, llm_scorer, corpus, target_synonyms)
        else:
            super().apply_llm(candidate, llm_scorer, corpus, target_synonyms)

    def _score_phrase(self, rows) -> tuple[float, dict[str, float]]:
        # Golf scoring: Average adjusted frequency of words in phrase
        scores = []
        adj_freqs = []
        for r in rows:
            is_n = getattr(r, "is_proper_noun", False)
            adj = adjusted_freq(float(r.frequency), float(r.stopword_ratio_entry), self.stopword_penalty, self.stopword_power, is_proper_noun=is_n)
            scores.append(20.0 - adj)
            adj_freqs.append(round(adj, 2))
        
        avg_cost = sum(scores) / len(scores)
        # More words = slightly higher penalty?
        # Maybe just op_penalty
        score = avg_cost * self.w_phrase + self.op_penalty
        detailed = {"freqs": adj_freqs}
        return score, detailed
