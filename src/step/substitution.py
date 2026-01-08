from __future__ import annotations

from src.step.step import Step, Candidate, BaseStepGenerator

class SubstitutionStep(BaseStepGenerator):
    name = "SUBSTITUTION"

    def __init__(self, **kwargs):
        super().__init__("substitution", **kwargs)

    def generate(self, corpus_obj, target: str, *, limit: int = 200, forbidden_source: str | None = None) -> Step:
        t = str(target).lower()
        step = Step(op=self.name, target=t)
        
        if forbidden_source and t == forbidden_source.lower():
            # This is weird for substitution but let's keep it for consistency
            return step

        candidates = []
        if t in corpus_obj.substitutions:
            source = t
            candidates.append(
                Candidate(
                    source=source,
                    produced=t,
                    leftover_sorted="",
                    leftover_words=(),
                    strictness="SUBSTRING",
                    score=self.op_penalty
                )
            )

        step.candidates = sorted(candidates, key=lambda c: c.score)[:limit]
        return step
