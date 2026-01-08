from __future__ import annotations

from typing import List, Tuple
from src.corpus import corpus as corpus_df
from src.step.step import Step, Candidate, BaseStepGenerator
from src.utils import adjusted_freq, get_word_display
from src.scoring_config import config

class CharadeStep(BaseStepGenerator):
    name = "CHARADE"

    def __init__(self, **kwargs):
        super().__init__("charade", **kwargs)
        c = config.get("charade")

        # Charades should score very well, similar to reversals (which are 1.0)
        self.w_word = kwargs.get("w_word", c.get("w_word", 1.0))
        self.short_word_penalty = kwargs.get("short_word_penalty", c.get("short_word_penalty", 2.0))

    def generate(self, corpus, target: str, *, limit: int = 200, num_chunks: int | None = None) -> Step:
        """
        Finds all ways to split target into 2 or more literal words.
        E.g. "carpet" -> "car" + "pet"
        """
        t = str(target).lower()
        step = Step(op=self.name, target=t)
        n = len(t)
        
        candidates_found = []

        # Start with 2-word splits (most common)
        if num_chunks is None or num_chunks == 2:
            for i in range(1, n):
                w1 = t[:i]
                w2 = t[i:]
                
                adj1 = self._get_word_adj(corpus, w1)
                if adj1 is None:
                    continue
                adj2 = self._get_word_adj(corpus, w2)
                if adj2 is None:
                    continue
                
                # Penalties for short words
                penalty = 0.0
                if len(w1) < 3:
                    penalty += self.short_word_penalty
                if len(w2) < 3:
                    penalty += self.short_word_penalty
                
                # GOLF SCORING: average cost of words + op_penalty + short_penalties
                avg_adj = (adj1 + adj2) / 2.0
                cost = self.w_word * (20.0 - avg_adj) + self.op_penalty + penalty
                
                candidates_found.append(
                    Candidate(
                        source=f"{get_word_display(corpus, w1)} + {get_word_display(corpus, w2)}",
                        produced=t,
                        score=cost,
                        strictness="SUBSTRING"
                    )
                )

        # Extend to 3-word splits if target is long enough
        if (num_chunks is None and n >= 6) or num_chunks == 3:
            if n >= 3:
                for i in range(1, n - 1):
                    w1 = t[:i]
                    adj1 = self._get_word_adj(corpus, w1)
                    if adj1 is None: continue
                    for j in range(i + 1, n):
                        w2 = t[i:j]
                        w3 = t[j:]
                        
                        adj2 = self._get_word_adj(corpus, w2)
                        if adj2 is None: continue
                        adj3 = self._get_word_adj(corpus, w3)
                        if adj3 is None: continue
                        
                        penalty = 0.0
                        for w in [w1, w2, w3]:
                            if len(w) < 3:
                                penalty += self.short_word_penalty
                        
                        avg_adj = (adj1 + adj2 + adj3) / 3.0
                        # Additional penalty for more words to prefer simpler charades
                        cost = self.w_word * (20.0 - avg_adj) + self.op_penalty + penalty + 5.0
                        
                        candidates_found.append(
                            Candidate(
                                source=f"{get_word_display(corpus, w1)} + {get_word_display(corpus, w2)} + {get_word_display(corpus, w3)}",
                                produced=t,
                                score=cost,
                                strictness="SUBSTRING"
                            )
                        )

        step.candidates = sorted(candidates_found, key=lambda c: c.score)[:limit]
        return step

    def apply_llm(self, candidate: Candidate, llm_scorer, corpus=None, target_synonyms: List[str] = None) -> None:
        if not llm_scorer:
            return
        
        from src.utils import clean_source_fodder
        source_clean = clean_source_fodder(candidate.source)
        words = source_clean.split()
        if len(words) > 1:
            coherence = llm_scorer.score_coherence(source_clean)
            
            bonus = coherence * self.llm_weight
            candidate.score -= bonus
            candidate.detailed_scores["llm_coherence"] = round(coherence, 3)
            
            # Also apply definition context bonus
            super().apply_llm(candidate, llm_scorer, corpus, target_synonyms)
        else:
            super().apply_llm(candidate, llm_scorer, corpus, target_synonyms)

    def find_containers(self, corpus, target: str, limit: int = 200) -> Step:
        """
        Finds all words in the corpus that can be formed by 'target' plus another literal word.
        E.g. "cat" -> "catcall", "catfish", "bobcat", "copycat"
        """
        t = str(target).lower()
        step = Step(op=f"{self.name}_CONTAINER", target=t)
        df = corpus.corpus
        
        # We look for words that:
        # 1. Start with target: target + X
        # 2. End with target: X + target
        
        # Optimization: word must be longer than target
        mask = (df["entry"].str.startswith(t) | df["entry"].str.endswith(t)) & (df["entry"].str.len() > len(t))
        hits = df[mask]
        
        candidates_found = []
        for _, row in hits.iterrows():
            entry = str(row["entry"])
            if entry.startswith(t):
                other = entry[len(t):]
                source = f"{get_word_display(corpus, t)} + {get_word_display(corpus, other)}"
            else:
                other = entry[:-len(t)]
                source = f"{get_word_display(corpus, other)} + {get_word_display(corpus, t)}"
            
            # The 'other' part MUST be a literal word in the corpus
            adj_other = self._get_word_adj(corpus, other)
            if adj_other is None:
                continue
                
            # Score based on both words (target is fixed, but its freq matters for the quality of the charade)
            adj_t = self._get_word_adj(corpus, t)
            if adj_t is None: continue # Should not happen if target is a word but let's be safe
            
            penalty = 0.0
            if len(other) < 3:
                penalty += self.short_word_penalty
                
            avg_adj = (adj_t + adj_other) / 2.0
            cost = self.w_word * (20.0 - avg_adj) + self.op_penalty + penalty
            
            candidates_found.append(
                Candidate(
                    source=f"{entry} ({source})",
                    produced=t,
                    score=cost,
                    strictness="SUBSTRING"
                )
            )

        step.candidates = sorted(candidates_found, key=lambda c: c.score)[:limit]
        return step
