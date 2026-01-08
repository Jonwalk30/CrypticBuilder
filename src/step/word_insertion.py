from __future__ import annotations

from typing import Tuple, List
from src.corpus import corpus as corpus_df
from src.step.step import Step, Candidate, BaseStepGenerator
from src.utils import adjusted_freq, sort_word
from src.scoring_config import config

class WordInsertionStep(BaseStepGenerator):
    name = "WORD_INSERTION"

    def __init__(self, **kwargs):
        super().__init__("word_insertion", **kwargs)
        c = config.get("word_insertion")

        self.w_outer = kwargs.get("w_outer", c.get("w_outer", 1.0))
        self.w_inner = kwargs.get("w_inner", c.get("w_inner", 1.0))
        self.short_word_penalty = kwargs.get("short_word_penalty", c.get("short_word_penalty", 4.0))
        self.strictness_bonus = {
            "SUBSTRING": c.get("substring_bonus", -5.0),
            "MULTISET": c.get("multiset_bonus", 5.0),
        }

    def generate(self, corpus, target: str, *, limit: int = 200, llm_scorer=None) -> Step:
        t = str(target).lower()
        step = Step(op=self.name, target=t)
        n = len(t)
        
        candidates_found = {} # Key: (outer, inner_sig), Value: Candidate
        # i: start of inner word, j: end of inner word (exclusive)
        # To be a true insertion, prefix and suffix must be non-empty.
        # So i >= 1 and j <= n-1.
        for i in range(1, n):
            for j in range(i + 1, n):
                inner_raw = t[i:j]
                outer = t[:i] + t[j:]
                
                # Check outer word first (must be literal)
                outer_adj = self._get_word_adj(corpus, outer)
                if outer_adj is None:
                    continue
                
                # Find all possible inner words that are anagrams of the substring
                inner_sig = sort_word(inner_raw)
                
                # Deduplicate: if we've seen this (outer, inner_sig) combo, skip or pick best
                # Since the score depends on inner_sig (and outer), same combo should have same score
                # except maybe for SUBSTRING vs MULTISET if inner_raw changes but signature is same.
                # Actually, inner_raw IS part of the signature. 
                # If inner_raw1 and inner_raw2 have same signature, they might both be literal words or not.
                
                # Find all possible inner words that are anagrams of the substring
                possible_inner_words = tuple(corpus._anagram_words_sorted.get(inner_sig, []))
                
                # Filter to only keep literal words from the corpus
                if not possible_inner_words and inner_raw not in corpus.substitutions:
                    continue

                # Decide strictness: SUBSTRING if literal inner word is among valid words
                is_literal_word = inner_raw in possible_inner_words
                strictness = "SUBSTRING" if is_literal_word else "MULTISET"
                
                # Check if it's a substitution word
                is_sub = inner_raw in corpus.substitutions
                
                # Format inner word words
                final_inner_words = list(possible_inner_words)
                
                # If it's a substitution word but not in corpus, add it to final_inner_words
                if is_sub and inner_raw not in possible_inner_words:
                    final_inner_words.append(inner_raw)

                # Get best inner word meta for scoring
                # We use the literal one if it exists, else the best frequency one
                if is_literal_word:
                    iw_adj = self._get_word_adj(corpus, inner_raw)
                else:
                    meta = corpus._anagram_best.get(inner_sig)
                    iw_freq = float(meta["best_freq"]) if meta else 0.0
                    iw_stop = float(meta["best_stop_ratio"]) if meta else 0.0
                    iw_is_name = bool(meta.get("best_is_proper_noun", False)) if meta else False
                    iw_adj = adjusted_freq(iw_freq, iw_stop, self.stopword_penalty, self.stopword_power, is_proper_noun=iw_is_name)
                
                # If it's a substitution word but adj is 0 (approx), give it a baseline adj
                if (iw_adj is None or iw_adj < 0.1) and is_sub:
                    iw_adj = 3.0 # reasonable default for substitution words
                
                if iw_adj is None:
                    continue

                # Penalties for short words
                penalty = 0.0
                if len(inner_raw) < 3:
                    penalty += self.short_word_penalty
                if len(outer) < 3:
                    penalty += self.short_word_penalty
                
                # Bonus/Malus for strictness
                bonus = float(self.strictness_bonus[strictness])
                
                # GOLF SCORING
                cost = (
                    self.w_inner * (20.0 - iw_adj)
                    + self.w_outer * (20.0 - outer_adj)
                    + self.op_penalty
                    + penalty
                    - bonus
                )
                
                detailed = {
                    "outer_freq": round(outer_adj, 2),
                    "inner_freq": round(iw_adj, 2),
                }
                
                # Source display: if it's literal, use the literal. 
                # If it's multiset, show the word in parentheses.
                best_inner = final_inner_words[0]
                if strictness == "SUBSTRING":
                    source = f"{best_inner} in {outer}"
                else:
                    source = f"({best_inner}) in {outer}"

                cand = Candidate(
                    source=source,
                    produced=t,
                    leftover_sorted=inner_sig,
                    leftover_words=tuple(final_inner_words),
                    score=cost,
                    strictness=strictness,
                    detailed_scores=detailed
                )
                
                key = (outer, inner_sig)
                if key not in candidates_found or cost < candidates_found[key].score:
                    candidates_found[key] = cand
        
        # Deduplicate and sort
        step.candidates = sorted(candidates_found.values(), key=lambda c: c.score)[:limit]
        return step

    def apply_llm(self, candidate: Candidate, llm_scorer, corpus=None, target_synonyms: List[str] = None) -> None:
        if not llm_scorer or not candidate.leftover_words:
            return
        
        source = candidate.source
        if " in " in source:
            parts = source.split(" in ")
            outer = parts[1]
            # Use the first leftover word
            best_iw = candidate.leftover_words[0].split(" (")[0]
            
            # Check context (ONLY original words)
            context = llm_scorer.get_contextual_score(outer, best_iw)

            bonus = context * self.llm_weight
            candidate.score -= bonus
            candidate.detailed_scores["llm_context"] = round(context, 3)
            
            # Also check for definition context
            super().apply_llm(candidate, llm_scorer, corpus, target_synonyms)
