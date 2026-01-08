from __future__ import annotations

from typing import List, Tuple
from src.step.step import Step, Candidate, BaseStepGenerator

class PositionalDeletionStep(BaseStepGenerator):
    name = "POSITIONAL_DELETION"

    def __init__(self, **kwargs):
        super().__init__("positional_deletion", **kwargs)

    def generate(self, corpus, target: str, *, limit: int = 200, forbidden_source: str | None = None, llm_scorer=None) -> Step:
        from src.corpus import corpus as corpus_df
        t = str(target).lower()
        step = Step(op=self.name, target=t)
        if not t:
            return step

        df = corpus_df.corpus
        n = len(t)

        df_candidates = df
        if forbidden_source:
            df_candidates = df[df["entry"] != forbidden_source.lower()]

        # 1. Beheaded (all but the first): source[1:] == target
        # Source length n+1
        mask_behead = (df_candidates["entry"].str.len() == n + 1) & (df_candidates["entry"].str.endswith(t))
        hits_behead = df_candidates[mask_behead]
        for row in hits_behead.itertuples(index=False):
            w = str(row.entry)
            adj = self._get_adj(row)
            
            cost = (20.0 - adj) + self.op_penalty
            detailed = {"freq": round(adj, 2)}
            step.candidates.append(Candidate(source=f"{w} (beheaded)", produced=t, score=cost, strictness="SUBSTRING", detailed_scores=detailed))

        # 2. Curtailed (all but the last): source[:-1] == target
        # Source length n+1
        mask_curtail = (df_candidates["entry"].str.len() == n + 1) & (df_candidates["entry"].str.startswith(t))
        hits_curtail = df_candidates[mask_curtail]
        for row in hits_curtail.itertuples(index=False):
            w = str(row.entry)
            adj = self._get_adj(row)
            
            cost = (20.0 - adj) + self.op_penalty
            detailed = {"freq": round(adj, 2)}
            step.candidates.append(Candidate(source=f"{w} (curtailed)", produced=t, score=cost, strictness="SUBSTRING", detailed_scores=detailed))

        # 3. Heart (all but the first & last): source[1:-1] == target
        # Source length n+2. Only consider odd length source words (so n must be odd).
        if n % 2 == 1:
            mask_heart = (df_candidates["entry"].str.len() == n + 2) & (df_candidates["entry"].str.slice(1, -1) == t)
            hits_heart = df_candidates[mask_heart]
            for row in hits_heart.itertuples(index=False):
                w = str(row.entry)
                adj = self._get_adj(row)
                
                cost = (20.0 - adj) + self.op_penalty
                detailed = {"freq": round(adj, 2)}
                step.candidates.append(Candidate(source=f"{w} (heart)", produced=t, score=cost, strictness="SUBSTRING", detailed_scores=detailed))

        # 4. Gutted (all but the middle letters -> leave only the outside):
        # User requested: "gutted should remove all middle letters leaving just the first and last"
        # This implies it ONLY works if the target is exactly the first and last letters of the source.
        # Thus the target length must be 2.
        if n == 2:
            prefix = t[0]
            suffix = t[1]
            # Source must be at least length 3 to have something to remove
            mask_gutted = (df_candidates["entry"].str.len() >= 3) & \
                          (df_candidates["entry"].str.startswith(prefix)) & \
                          (df_candidates["entry"].str.endswith(suffix))
            hits_gutted = df_candidates[mask_gutted]
            for row in hits_gutted.itertuples(index=False):
                w = str(row.entry)
                # The produced string is always the first and last letters.
                # Since n == 2 and we matched prefix/suffix, t is already w[0]+w[-1].
                adj = self._get_adj(row)
                
                cost = (20.0 - adj) + self.op_penalty
                detailed = {"freq": round(adj, 2)}
                step.candidates.append(Candidate(source=f"{w} (gutted)", produced=t, score=cost, strictness="SUBSTRING", detailed_scores=detailed))

        step.candidates.sort(key=lambda c: c.score)
        step.candidates = step.candidates[:limit]
        return step
