from __future__ import annotations
from src.corpus import corpus as corpus_df
from src.step.step import Step, Candidate, BaseStepGenerator
from src.utils import get_word_display

class ReversalStep(BaseStepGenerator):
    name = "REVERSAL"

    def __init__(self, **kwargs):
        super().__init__("reversal", **kwargs)

    def generate(self, corpus, target: str, *, limit: int = 200, forbidden_source: str | None = None, llm_scorer=None) -> Step:
        t = str(target).lower()
        rev_t = t[::-1]
        step = Step(op=self.name, target=t)

        if forbidden_source and rev_t == forbidden_source.lower():
            return step

        df = corpus_df.corpus
        # Find words that are the reverse of the target
        hits = df[df["entry"] == rev_t]

        for _, row in hits.iterrows():
            w = str(row["entry"])
            
            # GOLF SCORING
            adj = self._get_adj(row)
            
            cost = (20.0 - adj) + self.op_penalty
            detailed = {"freq": round(adj, 2)}

            step.candidates.append(Candidate(source=get_word_display(corpus, w), produced=t, score=cost, strictness="SUBSTRING", detailed_scores=detailed))

            if len(step.candidates) >= limit:
                break

        step.candidates.sort(key=lambda c: c.score, reverse=False)
        return step
