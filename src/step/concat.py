from __future__ import annotations

from typing import List, Tuple
from src.step.step import Step, Candidate, BaseStepGenerator
from src.scoring_config import config

class ConcatStep(BaseStepGenerator):
    name = "CHARADE"

    def __init__(self, **kwargs):
        super().__init__("concat", **kwargs)
        c = config.get("concat")
        self.per_chunk_penalty = kwargs.get("per_chunk_penalty", c.get("per_chunk_penalty", 5.0))
        self.max_chunks = kwargs.get("max_chunks", c.get("max_chunks", 4))
        self.insertion_penalty = kwargs.get("insertion_penalty", 5.0)

    def generate(self, finder, corpus, target: str, *, limit: int = 50, num_chunks: int | None = None, llm_scorer=None) -> List[Step]:
        t = str(target).lower()
        n = len(t)
        if n < 2:
            return []

        # We will try splitting target into linear chunks or wrap-around chunks
        all_splits = []
        
        # 2 chunks (Linear and Wrap-around)
        if num_chunks is None or num_chunks == 2:
            # Linear
            for i in range(1, n):
                all_splits.append(((t[:i], t[i:]), "JOIN"))
            
            # Wrap-around (Insertion style)
            # e.g. "flat" -> "f-t" (outer) and "la" (inner)
            for i in range(1, n):
                for j in range(i + 1, n):
                    outer = t[:i] + t[j:]
                    inner = t[i:j]
                    # Format as (outer, inner) for the insertion case
                    all_splits.append(((outer, inner), "INSERTION"))
            
        # 3 chunks (Linear only for now)
        if (num_chunks is None and self.max_chunks >= 3) or num_chunks == 3:
            if n >= 3:
                for i in range(1, n - 1):
                    for j in range(i + 1, n):
                        all_splits.append(((t[:i], t[i:j], t[j:]), "JOIN"))
                    
        # 4 chunks (Linear only)
        if (num_chunks is None and self.max_chunks >= 4) or num_chunks == 4:
            if n >= 4:
                # Only for longer words to keep it reasonable
                if n >= 6:
                    for i in range(1, n - 2):
                        for j in range(i + 1, n - 1):
                            for k in range(j + 1, n):
                                all_splits.append(((t[:i], t[i:j], t[j:k], t[k:]), "JOIN"))

        results = []
        for split, mode in all_splits:
            # For each chunk in the split, find all viable non-PD and PD options
            all_chunk_non_pd = []
            all_chunk_pd = []
            
            valid_split = True
            for chunk in split:
                non_pds, pds = self._get_chunk_options(finder, corpus, chunk, t)
                if not non_pds and not pds:
                    valid_split = False
                    break
                all_chunk_non_pd.append(non_pds)
                all_chunk_pd.append(pds)
            
            if not valid_split:
                continue

            # We want to build combinations where AT MOST one chunk uses a positional deletion.
            combinations_to_try = []
            
            # Helper to get all combinations from a list of lists of Steps
            import itertools
            
            # 1. All Non-PD
            for combo in itertools.product(*all_chunk_non_pd):
                combinations_to_try.append(list(combo))
            
            # 2. Exactly one PD
            for i in range(len(split)):
                if not all_chunk_pd[i]:
                    continue
                
                # Combine PD options for chunk i with Non-PD options for other chunks
                others = []
                for j in range(len(split)):
                    if i == j:
                        others.append(all_chunk_pd[i])
                    else:
                        others.append(all_chunk_non_pd[j])
                
                for combo in itertools.product(*others):
                    combinations_to_try.append(list(combo))

            for chunk_steps in combinations_to_try:
                split_score = self.op_penalty + (len(split) - 2) * self.per_chunk_penalty
                if mode == "INSERTION":
                    split_score += self.insertion_penalty

                for s in chunk_steps:
                    split_score += s.best_score()

                # Create a combined step
                if mode == "INSERTION":
                    # outer is index 0, inner is index 1
                    source = f"{chunk_steps[1].best().source} in {chunk_steps[0].best().source}"
                else:
                    source = " + ".join([s.best().source for s in chunk_steps])
                
                detailed = {}
                
                combined_cand = Candidate(
                    source=source,
                    produced=t,
                    score=split_score,
                    detailed_scores=detailed
                )
                step = Step(op=self.name, target=t, candidates=[combined_cand], chunk_steps=chunk_steps)
                step.combined_score = split_score
                results.append(step)

        # Sort and limit
        results.sort(key=lambda s: s.best_score())
        return results[:limit]

    def apply_llm(self, candidate: Candidate, llm_scorer, corpus=None, target_synonyms: List[str] = None) -> None:
        if not llm_scorer:
            return
        
        # Concat source is "chunk1 + chunk2 + ..."
        phrase = candidate.source.replace(" + ", " ")
        words = phrase.split()
        if len(words) > 1:
            coherence = llm_scorer.score_coherence(phrase)
            
            bonus = coherence * self.llm_weight
            candidate.score -= bonus
            candidate.detailed_scores["llm_coherence"] = round(coherence, 3)
            
            # Also apply definition context bonus
            super().apply_llm(candidate, llm_scorer, corpus, target_synonyms)
        else:
            super().apply_llm(candidate, llm_scorer, corpus, target_synonyms)

    def _get_chunk_options(self, finder, corpus, chunk: str, original_target: str) -> Tuple[List[Step], List[Step]]:
        # Returns (non_pd_options, pd_options)
        non_pd_options = []
        pd_options = []
        
        # 1. Literal / Anagram
        from src.step.anagram import AnagramStep
        literal_an = AnagramStep(allow_identity=True)
        # Get more than 5 for literal/anagram to ensure we don't miss simple but good ones
        an_step = literal_an.generate(corpus, chunk, limit=10, forbidden_source=original_target)
        
        if an_step.candidates:
            identity_cand = next((c for c in an_step.candidates if c.source == chunk), None)
            if identity_cand:
                non_pd_options.append(Step(op="LITERAL", target=chunk, candidates=[identity_cand]))
            
            if len(chunk) >= 3:
                # Include some anagrams too
                an_cands = [c for c in an_step.candidates if c.source != chunk]
                for c in an_cands[:3]:
                    non_pd_options.append(Step(op="ANAGRAM", target=chunk, candidates=[c]))
        
        # 1.5 Positional Deletion (beheaded, curtailed, heart, gutted)
        if len(chunk) >= 2:
            pd_step = finder.pd.generate(corpus, chunk, limit=5, forbidden_source=original_target)
            if pd_step.candidates:
                for c in pd_step.candidates[:3]:
                    pd_options.append(Step(op="POSITIONAL_DELETION", target=chunk, candidates=[c]))
            
        # 2. Reversal
        rv_step = finder.rv.generate(corpus, chunk, limit=3, forbidden_source=original_target)
        if rv_step.candidates:
            for c in rv_step.candidates:
                non_pd_options.append(Step(op="REVERSAL", target=chunk, candidates=[c]))
            
        # 3. Substitution
        su_step = finder.su.generate(corpus, chunk, limit=5, forbidden_source=original_target)
        if su_step.candidates:
            for c in su_step.candidates[:3]:
                non_pd_options.append(Step(op="SUBSTITUTION", target=chunk, candidates=[c]))

        # 3.5 Alternating Letters
        al_step = finder.al.generate(corpus, chunk, limit=3, forbidden_source=original_target)
        if al_step.candidates:
            for c in al_step.candidates:
                non_pd_options.append(Step(op="ALTERNATING_LETTERS", target=chunk, candidates=[c]))

        # 4. More complex wordplay for "larger" chunks (length >= 5)
        if len(chunk) >= 5:
            # Letter Deletion
            ld_step = finder.ld.generate(corpus, chunk, limit=5, forbidden_source=original_target)
            if ld_step.candidates:
                ld_step.candidates = [c for c in ld_step.candidates if c.strictness != "MULTISET"]
                for c in ld_step.candidates[:2]:
                    non_pd_options.append(Step(op="LETTER_DELETION", target=chunk, candidates=[c]))
            
            # Letter Replacement
            lr_step = finder.lr.generate(corpus, chunk, limit=5, forbidden_source=original_target)
            if lr_step.candidates:
                lr_step.candidates = [c for c in lr_step.candidates if c.strictness != "MULTISET"]
                for c in lr_step.candidates[:2]:
                    non_pd_options.append(Step(op="LETTER_REPLACEMENT", target=chunk, candidates=[c]))
                
            # Selection (First, Last, Middle Letters)
            if finder.use_positional_selection:
                for sel_gen, sel_op in [(finder.fl, "FIRST_LETTERS"), (finder.ll, "LAST_LETTERS"), (finder.ml, "MIDDLE_LETTERS")]:
                    sel_step = sel_gen.generate(corpus, chunk, limit=2)
                    if sel_step.candidates:
                        for c in sel_step.candidates:
                            non_pd_options.append(Step(op=sel_op, target=chunk, candidates=[c]))
                
            # Word Deletion
            wd_step = finder.wd.generate(corpus, chunk, limit=5, forbidden_source=original_target)
            if wd_step.candidates:
                wd_step.candidates = [c for c in wd_step.candidates if c.strictness != "MULTISET"]
                for c in wd_step.candidates[:2]:
                    non_pd_options.append(Step(op="WORD_DELETION", target=chunk, candidates=[c]))
                
            # Word Insertion
            wi_step = finder.wi.generate(corpus, chunk, limit=5)
            if wi_step.candidates:
                wi_step.candidates = [c for c in wi_step.candidates if c.strictness != "MULTISET"]
                for c in wi_step.candidates[:2]:
                    non_pd_options.append(Step(op="WORD_INSERTION", target=chunk, candidates=[c]))

        return non_pd_options, pd_options
