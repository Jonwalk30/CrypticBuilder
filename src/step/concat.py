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
            # For each chunk in the split, find the best non-PD step and best PD step
            all_chunk_options = []
            valid_split = True
            for chunk in split:
                non_pd, pd = self._get_chunk_options(finder, corpus, chunk, t)
                if not non_pd and not pd:
                    valid_split = False
                    break
                all_chunk_options.append((non_pd, pd))
            
            if not valid_split:
                continue

            # We want to build combinations where AT MOST one chunk uses a positional deletion.
            combinations_to_try = []
            
            # 1. All Non-PD
            if all(o[0] for o in all_chunk_options):
                combinations_to_try.append([o[0] for o in all_chunk_options])
            
            # 2. Exactly one PD
            for i in range(len(all_chunk_options)):
                pd_step = all_chunk_options[i][1]
                if not pd_step:
                    continue
                
                combo = []
                possible = True
                for j in range(len(all_chunk_options)):
                    if i == j:
                        combo.append(pd_step)
                    else:
                        if all_chunk_options[j][0]:
                            combo.append(all_chunk_options[j][0])
                        else:
                            possible = False
                            break
                if possible:
                    combinations_to_try.append(combo)

            for chunk_steps in combinations_to_try:
                split_score = self.op_penalty + (len(split) - 2) * self.per_chunk_penalty
                if mode == "INSERTION":
                    split_score += self.insertion_penalty

                for s in chunk_steps:
                    split_score += s.best_score()

                # Create a combined step
                if mode == "INSERTION":
                    # outer is index 0, inner is index 1
                    source = f"{chunk_steps[1].target} in {chunk_steps[0].target}"
                else:
                    source = " + ".join([s.target for s in chunk_steps])
                
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
            synonyms_map = self._get_synonyms_map(words, corpus, llm_scorer)
            coherence, best_phrase = llm_scorer.get_best_coherence(words, synonyms_map)
            
            bonus = coherence * self.llm_weight
            candidate.score -= bonus
            candidate.detailed_scores["llm_coherence"] = round(coherence, 3)
            if best_phrase != phrase:
                candidate.detailed_scores["best_surface"] = best_phrase
            
            # Also apply definition context bonus
            super().apply_llm(candidate, llm_scorer, corpus, target_synonyms)
        else:
            super().apply_llm(candidate, llm_scorer, corpus, target_synonyms)

    def _get_chunk_options(self, finder, corpus, chunk: str, original_target: str) -> Tuple[Step | None, Step | None]:
        # Returns (best_non_pd_step, best_pd_step)
        non_pd_options = []
        pd_options = []
        
        # 1. Literal / Anagram
        from src.step.anagram import AnagramStep
        literal_an = AnagramStep(allow_identity=True)
        an_step = literal_an.generate(corpus, chunk, limit=5, forbidden_source=original_target)
        
        if an_step.candidates:
            identity_cand = next((c for c in an_step.candidates if c.source == chunk), None)
            if identity_cand:
                non_pd_options.append(Step(op="LITERAL", target=chunk, candidates=[identity_cand]))
            
            if len(chunk) >= 3:
                an_cands = [c for c in an_step.candidates if c.source != chunk]
                if an_cands:
                    non_pd_options.append(Step(op="ANAGRAM", target=chunk, candidates=an_cands))
        
        # 1.5 Positional Deletion (beheaded, curtailed, heart, gutted)
        # Allowed for chunks of length >= 2 (gutted requires 2, others can work for 2 too)
        if len(chunk) >= 2:
            pd_step = finder.pd.generate(corpus, chunk, limit=5, forbidden_source=original_target)
            if pd_step:
                pd_options.append(pd_step)
            
        # 2. Reversal
        rv_step = finder.rv.generate(corpus, chunk, limit=5, forbidden_source=original_target)
        if rv_step:
            non_pd_options.append(rv_step)
            
        # 3. Substitution
        su_step = finder.su.generate(corpus, chunk, limit=5, forbidden_source=original_target)
        if su_step:
            non_pd_options.append(su_step)

        # 3.5 Alternating Letters
        al_step = finder.al.generate(corpus, chunk, limit=5, forbidden_source=original_target)
        if al_step:
            non_pd_options.append(al_step)

        # 4. More complex wordplay for "larger" chunks (length >= 5)
        if len(chunk) >= 5:
            # Letter Deletion
            ld_step = finder.ld.generate(corpus, chunk, limit=5, forbidden_source=original_target)
            ld_step.candidates = [c for c in ld_step.candidates if c.strictness != "MULTISET"]
            if ld_step:
                non_pd_options.append(ld_step)
            
            # Letter Replacement
            lr_step = finder.lr.generate(corpus, chunk, limit=5, forbidden_source=original_target)
            lr_step.candidates = [c for c in lr_step.candidates if c.strictness != "MULTISET"]
            if lr_step:
                non_pd_options.append(lr_step)
                
            # Selection (First, Last, Middle Letters)
            if finder.use_positional_selection:
                fl_step = finder.fl.generate(corpus, chunk, limit=5)
                if fl_step:
                    non_pd_options.append(fl_step)
                
                ll_step = finder.ll.generate(corpus, chunk, limit=5)
                if ll_step:
                    non_pd_options.append(ll_step)
                
                ml_step = finder.ml.generate(corpus, chunk, limit=5)
                if ml_step:
                    non_pd_options.append(ml_step)
                
            # Word Deletion
            wd_step = finder.wd.generate(corpus, chunk, limit=5, forbidden_source=original_target)
            wd_step.candidates = [c for c in wd_step.candidates if c.strictness != "MULTISET"]
            if wd_step:
                non_pd_options.append(wd_step)
                
            # Word Insertion
            wi_step = finder.wi.generate(corpus, chunk, limit=5)
            wi_step.candidates = [c for c in wi_step.candidates if c.strictness != "MULTISET"]
            if wi_step:
                non_pd_options.append(wi_step)

        best_non_pd = min(non_pd_options, key=lambda s: s.best_score()) if non_pd_options else None
        best_pd = min(pd_options, key=lambda s: s.best_score()) if pd_options else None
        
        return best_non_pd, best_pd
