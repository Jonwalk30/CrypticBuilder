from __future__ import annotations
from typing import List
from src.corpus.corpus import corpus as corpus_df
from src.step.step import Step, Candidate, BaseStepGenerator
from src.utils import sort_word, adjusted_freq
from src.scoring_config import config

class AnagramStep(BaseStepGenerator):
    name = "ANAGRAM"

    def __init__(self, **kwargs):
        super().__init__("anagram", **kwargs)
        c = config.get("anagram")
        self.allow_identity = kwargs.get("allow_identity", c.get("allow_identity", False))

    def generate(self, corpus, target: str, *, limit: int = 200, max_fodder_words: int = 1, forbidden_source: str | None = None, llm_scorer=None) -> Step:
        t = str(target).lower()
        sig = sort_word(t)
        step = Step(op=self.name, target=t)

        candidates = []

        # 1. Single word anagrams
        fodders = corpus._anagram_words_sorted.get(sig, [])
        df = corpus_df.corpus

        forbidden_lower = forbidden_source.lower() if forbidden_source else None

        for w in fodders[: limit * 2]:
            w = str(w)
            if (not self.allow_identity) and w == t:
                continue
            if forbidden_lower and w == forbidden_lower:
                continue
            row = df.loc[df["entry"] == w]
            if row.empty:
                continue
            
            row_series = row.iloc[0]
            adj = self._get_adj(row_series)
            cost = (20.0 - adj) + self.op_penalty

            strictness = "SUBSTRING" if w == t else "MULTISET"
            detailed = {"freq": round(adj, 2)}
            candidates.append(Candidate(source=w, produced=t, score=cost, strictness=strictness, detailed_scores=detailed))

        # 2. Multi-word anagrams (2-4 words)
        if max_fodder_words > 1:
            multi_word_cands = self._find_multi_word_combinations(corpus, t, sig, max_fodder_words, limit, llm_scorer)
            candidates.extend(multi_word_cands)

        # Deduplicate and sort
        seen = {}
        unique_candidates = []
        for c in candidates:
            # Sort source words to avoid duplicates like "car pet" and "pet car"
            source_key = " ".join(sorted(c.source.split()))
            if source_key not in seen or c.score < seen[source_key]:
                seen[source_key] = c.score
                unique_candidates.append(c)

        step.candidates = sorted(unique_candidates, key=lambda c: c.score)[:limit]
        return step

    def _find_multi_word_combinations(self, corpus, target: str, target_sig: str, max_words: int, limit: int, llm_scorer=None) -> List[Candidate]:
        from src.utils import multiset_contains, multiset_leftover_sorted
        
        # Pre-filter signatures that are subsets of target_sig
        sub_sigs = []
        for sig, words in corpus._anagram_words_sorted.items():
            if not sig: continue
            if multiset_contains(target_sig, sig):
                # Use best word info for this signature
                best_info = corpus._anagram_best[sig]
                
                # Filter out obscure words from multi-word fodder to keep surface readable
                # 2-letter words must be very common (Zipf > 4.0)
                # 3+ letter words must be reasonably common (Zipf > 3.0)
                # This helps avoid obscure abbreviations or names like 'ca' or 'petr'
                if len(best_info["best_word"]) <= 2 and best_info["best_freq"] < 4.0:
                    continue
                if len(best_info["best_word"]) > 2 and best_info["best_freq"] < 3.0:
                    continue

                adj = adjusted_freq(best_info["best_freq"], best_info["best_stop_ratio"],
                                    self.stopword_penalty, self.stopword_power,
                                    is_proper_noun=best_info.get("best_is_proper_noun", False))
                cost = (20.0 - adj)
                sub_sigs.append((sig, cost, best_info["best_word"]))
        
        # Sort sub_sigs by cost (best words first)
        sub_sigs.sort(key=lambda x: x[1])
        
        # Limit to top sub_sigs to prevent combinatorial explosion
        sub_sigs = sub_sigs[:150] 
        
        combinations = []
        multi_word_penalty_per_word = 2.0 # Penalty for each word beyond the first

        def search(remaining_sig, current_combination, current_cost, start_index, current_freqs):
            if not remaining_sig:
                if len(current_combination) > 1:
                    detailed = {"freqs": [round(f, 2) for f in current_freqs]}
                    combinations.append((current_combination, current_cost, detailed))
                return
            
            if len(current_combination) >= max_words:
                return
            
            for i in range(start_index, len(sub_sigs)):
                sig, cost, best_word = sub_sigs[i]
                if multiset_contains(remaining_sig, sig):
                    # We need the adjusted frequency for detailed scores
                    # cost is (20.0 - adj), so adj = 20.0 - cost
                    adj = 20.0 - cost
                    search(
                        multiset_leftover_sorted(remaining_sig, sig),
                        current_combination + [best_word],
                        current_cost + cost,
                        i,
                        current_freqs + [adj]
                    )
                    if len(combinations) >= limit:
                        return
        
        search(target_sig, [], 0, 0, [])
        
        final_candidates = []
        for combo, cost, detailed in combinations:
            source = " ".join(combo)
            # Skip if it's just a literal split (no letters reordered)
            if "".join(combo) == target:
                continue
            
            total_score = cost + self.op_penalty + (len(combo) - 1) * multi_word_penalty_per_word
            final_candidates.append(Candidate(source=source, produced=target, score=total_score, strictness="MULTISET", detailed_scores=detailed))
        
        return final_candidates

    def apply_llm(self, candidate: Candidate, llm_scorer, corpus=None, target_synonyms: List[str] = None) -> None:
        if not llm_scorer:
            return
        
        words = candidate.source.split()
        if len(words) > 1:
            # Multi-word anagram: coherence bonus (making sense as a sentence)
            synonyms_map = self._get_synonyms_map(words, corpus, llm_scorer)
            coherence, best_phrase = llm_scorer.get_best_coherence(words, synonyms_map)
            
            bonus = coherence * self.llm_weight
            candidate.score -= bonus
            candidate.detailed_scores["llm_coherence"] = round(coherence, 3)
            if best_phrase != " ".join(words):
                candidate.detailed_scores["best_surface"] = best_phrase
            
            # Also apply definition context bonus for multi-word
            super().apply_llm(candidate, llm_scorer, corpus, target_synonyms)
        else:
            # Single word anagram: context with target
            super().apply_llm(candidate, llm_scorer, corpus, target_synonyms)
