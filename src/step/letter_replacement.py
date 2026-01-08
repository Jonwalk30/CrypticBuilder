from __future__ import annotations

from collections import Counter
from src.corpus.corpus import corpus as corpus_df
from src.step.step import Step, Candidate, BaseStepGenerator
from src.utils import adjusted_freq
from src.scoring_config import config

class LetterReplacementStep(BaseStepGenerator):
    name = "LETTER_REPLACEMENT"

    def __init__(self, **kwargs):
        super().__init__("letter_replacement", **kwargs)
        c = config.get("letter_replacement")
        self.w_source = kwargs.get("w_source", c.get("w_source", 1.0))
        # Exponential reward for more letters replaced
        # base_replacement_bonus: bonus for the first letter
        # multi_replacement_multiplier: multiplier for each additional letter replaced
        self.base_replacement_bonus = kwargs.get("base_replacement_bonus", c.get("base_replacement_bonus", 0.0))
        self.multi_replacement_exponent = kwargs.get("multi_replacement_exponent", c.get("multi_replacement_exponent", 2.0))

    def generate(self, corpus, target: str, *, limit: int = 200, forbidden_source: str | None = None, llm_scorer=None) -> Step:
        t = str(target).lower()
        df = corpus_df.corpus
        step = Step(op=self.name, target=t)

        # Candidates must have the same length as target
        possible_sources = df[df["entry"].str.len() == len(t)]
        
        if forbidden_source:
            possible_sources = possible_sources[possible_sources["entry"] != forbidden_source.lower()]
        
        for _, row in possible_sources.iterrows():
            source = str(row["entry"])
            if source == t:
                continue
            
            # Find the differences
            # We want words that are the same except for one type of letter being replaced by another
            # e.g. fill (f->p) pill
            # e.g. telephone (e->a) talaphana (if that were a word)
            
            diff_indices = [i for i, (c1, c2) in enumerate(zip(source, t)) if c1 != c2]
            
            if not diff_indices:
                continue
            
            # Letters at these indices in source must all be the same
            source_replaced_letters = {source[i] for i in diff_indices}
            # Letters at these indices in target must all be the same
            target_new_letters = {t[i] for i in diff_indices}
            
            if len(source_replaced_letters) == 1 and len(target_new_letters) == 1:
                old_char = list(source_replaced_letters)[0]
                new_char = list(target_new_letters)[0]
                num_replaced = len(diff_indices)
                
                # Check if ALL occurrences of old_char in source were replaced by new_char
                # The user said "if we replaced all of the e's in telephone to make a new word... that would be very interesting"
                # This suggests that replacing ALL occurrences is the "fun" part.
                # However, for 'fill' to 'pill', only one 'f' is replaced (there is only one 'f').
                # If we have 'feeling' and 'peeling', f->p replaces the only 'f'.
                # If we have 'geese' and 'geese' doesn't have a good example... 
                # Let's say 'added' and 'aided'. 'd' is not replaced, only 'e' -> 'i'.
                
                source_chars_count = source.count(old_char)
                is_full_replacement = (num_replaced == source_chars_count)
                
                # If it's not a full replacement of that letter, it's less "clean" but maybe still valid?
                # The prompt says "equally fill to mill or till would be options"
                # In 'fill', there is only one 'f'.
                
                adj = self._get_adj(row)
                
                # Scoring: 
                # Start with op_penalty. 
                # Subtract bonus that grows exponentially with num_replaced.
                # We want 1 letter replaced to be "poor" (high cost).
                # More letters = lower cost.
                
                # base_replacement_bonus might be negative (a penalty) for 1 letter,
                # or we just rely on op_penalty being high.
                
                # Bonus = base_replacement_bonus + (num_replaced ^ exponent)
                replacement_bonus = self.base_replacement_bonus + (float(num_replaced) ** self.multi_replacement_exponent)
                
                cost = self.w_source * (20.0 - adj) + self.op_penalty - replacement_bonus
                detailed = {"freq": round(adj, 2)}
                
                # Check for substitutions for old and new chars
                # Usually substitutions are for strings, but single letters are common too
                old_subs = corpus.substitutions.get(old_char, [])
                new_subs = corpus.substitutions.get(new_char, [])
                
                source_display = source
                if old_subs or new_subs:
                    # Mark if there are substitutions available
                    source_display = f"{source}* (subst. avail)"

                step.candidates.append(
                    Candidate(
                        source=f"{source_display} ({old_char.upper()}->{new_char.upper()})",
                        produced=t,
                        score=cost,
                        strictness="SUBSTRING" if is_full_replacement else "MULTISET",
                        detailed_scores=detailed
                    )
                )
            
            if len(step.candidates) >= limit * 2: # Get more and then sort/limit
                break

        step.candidates.sort(key=lambda c: c.score)
        step.candidates = step.candidates[:limit]
        return step
