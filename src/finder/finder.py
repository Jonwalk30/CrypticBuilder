from __future__ import annotations

from typing import List

from src.step.anagram import AnagramStep
from src.step.letter_deletion import WordDeletionStep, LetterDeletionStep
from src.step.letter_replacement import LetterReplacementStep
from src.step.letter_selection import FirstLetterStep, LastLetterStep, MiddleLetterStep, AlternatingLetterStep
from src.step.positional_deletion import PositionalDeletionStep
from src.step.reversal import ReversalStep
from src.step.word_insertion import WordInsertionStep
from src.step.hidden import HiddenStep
from src.step.substitution import SubstitutionStep
from src.step.concat import ConcatStep
from src.step.step import Step
from src.llm.llm_scorer import LLMScorer
from src.llm.indicator_suggestor import IndicatorSuggestor

class ClueFinder:
    def __init__(
        self,
        limit1: int = 200,
        limit_child: int = 120,
        top_k: int = 25,
        depth_penalty: float = 20.0,
        max_fodder_words: int = 4,
        llm_scorer: LLMScorer | None = None,
        use_llm: bool = True,
        **kwargs
    ):
        self.limit1 = limit1
        self.limit_child = limit_child
        self.top_k = top_k
        self.depth_penalty = depth_penalty
        self.max_fodder_words = max_fodder_words
        
        self.llm_scorer = llm_scorer if use_llm else None
        if use_llm and not self.llm_scorer:
            self.llm_scorer = LLMScorer()
        
        self.indicator_suggestor = IndicatorSuggestor(self.llm_scorer) if self.llm_scorer else None

        # Feature toggles
        self.use_positional_selection = kwargs.get("use_positional_selection", True)
        self.use_llm_for_positional = kwargs.get("use_llm_for_positional", True)

        # Step generators
        self.an = AnagramStep()
        self.wd = WordDeletionStep()
        self.ld = LetterDeletionStep()
        self.fl = FirstLetterStep()
        self.rv = ReversalStep()
        self.wi = WordInsertionStep()
        self.lr = LetterReplacementStep()
        self.hi = HiddenStep()
        self.su = SubstitutionStep()
        self.co = ConcatStep()
        self.ll = LastLetterStep()
        self.ml = MiddleLetterStep()
        self.al = AlternatingLetterStep()
        self.pd = PositionalDeletionStep()

    def _get_step(self, generator, corpus, target: str, limit: int = None, **kwargs) -> List[Step]:
        l = limit or self.limit1
        step = generator.generate(corpus, target, limit=l, **kwargs)
        if not step or not step.candidates:
            return []
        return [Step(op=step.op, target=step.target, candidates=[c]) for c in step.candidates]

    def get_anagrams(self, corpus, target: str, limit: int = None, max_fodder_words: int | None = None) -> List[Step]:
        mfw = max_fodder_words if max_fodder_words is not None else self.max_fodder_words
        return self._get_step(self.an, corpus, target, limit=limit, max_fodder_words=mfw)

    def get_word_deletions(self, corpus, target: str, limit: int = None) -> List[Step]:
        return self._get_step(self.wd, corpus, target, limit=limit)

    def get_letter_deletions(self, corpus, target: str, limit: int = None) -> List[Step]:
        return self._get_step(self.ld, corpus, target, limit=limit)

    def get_first_letters(self, corpus, target: str, limit: int = None) -> List[Step]:
        return self._get_step(self.fl, corpus, target, limit=limit)

    def get_reversals(self, corpus, target: str, limit: int = None) -> List[Step]:
        return self._get_step(self.rv, corpus, target, limit=limit)

    def get_word_insertions(self, corpus, target: str, limit: int = None) -> List[Step]:
        return self._get_step(self.wi, corpus, target, limit=limit)

    def get_letter_replacements(self, corpus, target: str, limit: int = None) -> List[Step]:
        return self._get_step(self.lr, corpus, target, limit=limit)

    def get_hiddens(self, corpus, target: str, limit: int = None) -> List[Step]:
        return self._get_step(self.hi, corpus, target, limit=limit)

    def get_substitutions(self, corpus, target: str, limit: int = None) -> List[Step]:
        return self._get_step(self.su, corpus, target, limit=limit)

    def get_concats(self, corpus, target: str, limit: int = None, num_chunks: int | None = None) -> List[Step]:
        l = limit or 50 # Concat is expensive, keep limit reasonable
        return self.co.generate(self, corpus, target, limit=l, num_chunks=num_chunks)

    def get_last_letters(self, corpus, target: str, limit: int = None) -> List[Step]:
        return self._get_step(self.ll, corpus, target, limit=limit)

    def get_middle_letters(self, corpus, target: str, limit: int = None) -> List[Step]:
        return self._get_step(self.ml, corpus, target, limit=limit)

    def get_alternating_letters(self, corpus, target: str, limit: int = None) -> List[Step]:
        return self._get_step(self.al, corpus, target, limit=limit)

    def get_positional_deletions(self, corpus, target: str, limit: int = None) -> List[Step]:
        return self._get_step(self.pd, corpus, target, limit=limit)

    def get_nested_deletions(self, corpus, target: str, limit: int = None) -> List[Step]:
        t = str(target).lower()
        out: List[Step] = []
        l = limit or self.limit1
        # We need the raw candidates from wd.generate here
        s_wd = self.wd.generate(corpus, t, limit=l)

        for cand in s_wd.candidates[:self.top_k]:
            lo = cand.leftover_sorted
            if not lo:
                continue

            # Try ANAGRAM on leftover letters
            child1 = self.an.generate(corpus, lo, limit=self.limit_child)
            if child1.candidates:
                nested = Step(op=s_wd.op, target=s_wd.target, candidates=[cand], child=child1)
                nested.combined_score = cand.score + child1.best_score() + self.depth_penalty
                out.append(nested)

            # Try FIRST_LETTERS on leftover letters
            if self.use_positional_selection:
                child2 = self.fl.generate(corpus, lo, limit=self.limit_child)
                if child2.candidates:
                    nested = Step(op=s_wd.op, target=s_wd.target, candidates=[cand], child=child2)
                    nested.combined_score = cand.score + child2.best_score() + self.depth_penalty
                    out.append(nested)
        return out

    def _get_all_standard_steps(self, corpus, target: str, limit: int = None, max_fodder_words: int | None = None) -> List[Step]:
        out = []
        out.extend(self.get_anagrams(corpus, target, limit=limit, max_fodder_words=max_fodder_words))
        out.extend(self.get_word_deletions(corpus, target, limit=limit))
        out.extend(self.get_word_insertions(corpus, target, limit=limit))
        out.extend(self.get_letter_deletions(corpus, target, limit=limit))
        out.extend(self.get_letter_replacements(corpus, target, limit=limit))
        out.extend(self.get_hiddens(corpus, target, limit=limit))
        out.extend(self.get_reversals(corpus, target, limit=limit))
        out.extend(self.get_substitutions(corpus, target, limit=limit))
        out.extend(self.get_alternating_letters(corpus, target, limit=limit))
        if self.use_positional_selection:
            out.extend(self.get_first_letters(corpus, target, limit=limit))
            out.extend(self.get_last_letters(corpus, target, limit=limit))
            out.extend(self.get_middle_letters(corpus, target, limit=limit))
        out.extend(self.get_positional_deletions(corpus, target, limit=limit))
        return out

    def _sort_and_deduplicate(self, steps: List[Step]) -> List[Step]:
        # Sort overall steps by their best available score (combined if nested)
        steps.sort(key=lambda s: s.best_score(), reverse=False)

        # Deduplicate: for each (source, leftover), keep only the one with the best score
        seen = {}
        unique_out = []
        for s in steps:
            best_cand = s.best()
            if not best_cand:
                continue
            # Some candidates might not have leftover_sorted (e.g. Anagram)
            key = (best_cand.source, best_cand.leftover_sorted)
            if key not in seen:
                seen[key] = True
                unique_out.append(s)
        return unique_out

    def apply_llm_refinement(self, steps: List[Step], corpus=None, limit: int = 50, score_threshold: float = 40.0) -> List[Step]:
        if not self.llm_scorer:
            return steps
            
        # 1. Identify candidates to refine
        to_refine = []
        count = 0
        for s in steps:
            if count >= limit: break
            if s.best_score() > score_threshold: continue
            
            # Skip AI for positional letters if disabled
            if not self.use_llm_for_positional and s.op in ["FIRST_LETTERS", "LAST_LETTERS", "MIDDLE_LETTERS"]:
                continue

            best_c = s.best()
            if not best_c: continue
            
            to_refine.append((s, best_c))
            count += 1
            
        if not to_refine:
            return steps
            
        # 2. Prefetch all LLM data in batches
        self._prefetch_llm_data(to_refine, corpus)
        
        op_map = {
            "ANAGRAM": self.an, "WORD_DELETION": self.wd, "LETTER_DELETION": self.ld,
            "FIRST_LETTERS": self.fl, "LAST_LETTERS": self.ll, "MIDDLE_LETTERS": self.ml,
            "ALTERNATING_LETTERS": self.al, "REVERSAL": self.rv, "WORD_INSERTION": self.wi,
            "POSITIONAL_DELETION": self.pd, "LETTER_REPLACEMENT": self.lr, "CHARADE": self.co,
            "HIDDEN": self.hi, "SUBSTITUTION": self.su
        }
        
        # Get synonyms/definitions for the target word to check for fodder-definition context
        target = steps[0].target if steps else None
        synonyms = self.llm_scorer.get_definitions(target) if target else []

        for s, best_c in to_refine:
            gen = op_map.get(s.op)
            if gen:
                old_score = best_c.score
                # Refined apply_llm now handles both coherence/context AND definition matching with synonyms
                gen.apply_llm(best_c, self.llm_scorer, corpus, synonyms)
                
                diff = best_c.score - old_score
                if s.combined_score is not None:
                    s.combined_score += diff
        
        # Re-sort after refinement
        steps.sort(key=lambda x: x.best_score())
        return steps

    def _prefetch_llm_data(self, to_refine, corpus):
        import re
        all_words = set()
        target = to_refine[0][1].produced
        all_words.add(target)
        
        for s, c in to_refine:
            source_clean = re.sub(r"\s*\(.*?\)", "", c.source).replace("+", " ").replace(" in ", " ")
            for w in source_clean.split():
                all_words.add(w)
            if s.op in ["WORD_DELETION", "WORD_INSERTION"] and c.leftover_words:
                all_words.add(c.leftover_words[0].split(" (")[0])
        
        # 1. Fetch all definitions (synonyms) for all words
        self.llm_scorer.batch_get_definitions(list(all_words))
        
        # 2. Collect all pairs that apply_llm will likely ask for
        target_syns = [target] + self.llm_scorer.cache.get(f"definitions:{target}", [])[:5]
        all_pairs = []
        all_phrases = []
        
        for s, c in to_refine:
            source_clean = re.sub(r"\s*\(.*?\)", "", c.source).replace("+", " ").replace(" in ", " ")
            words = source_clean.split()
            all_phrases.append(source_clean)
            
            # Words and their synonyms
            words_and_syns = []
            for w in words:
                syns = [w] + self.llm_scorer.cache.get(f"definitions:{w}", [])[:5]
                words_and_syns.append(syns)
                
                # Coherence variations (one-word substitution)
                for syn in syns[1:]:
                    new_words = [syn if x == w else x for x in words]
                    all_phrases.append(" ".join(new_words))
            
            # Pairwise context for each word (or syn) vs each target (or syn)
            for w_options in words_and_syns:
                for opt in w_options:
                    for ts in target_syns:
                        all_pairs.append((opt, ts))

            # Special case for WORD_DELETION/INSERTION
            if s.op in ["WORD_DELETION", "WORD_INSERTION"] and c.leftover_words:
                best_lo = c.leftover_words[0].split(" (")[0]
                lo_syns = [best_lo] + self.llm_scorer.cache.get(f"definitions:{best_lo}", [])[:5]
                
                container = c.source
                if s.op == "WORD_INSERTION" and " in " in c.source:
                    container = c.source.split(" in ")[1]
                
                cont_syns = [container] + self.llm_scorer.cache.get(f"definitions:{container}", [])[:5]
                
                for ls in lo_syns:
                    for cs in cont_syns:
                        all_pairs.append((ls, cs))

        # 3. Batch fetch everything
        self.llm_scorer.batch_get_contextual_scores(all_pairs)
        self.llm_scorer.batch_score_coherence(all_phrases)

    def find_all(self, corpus, target: str, num_chunks: int | None = None, max_fodder_words: int | None = None, enabled_ops: List[str] | None = None) -> List[Step]:
        """
        Produces a sorted list of all possible clue options with a larger limit.
        """
        t = str(target).lower()
        out: List[Step] = []

        # Increase limits for find_all
        large_limit = self.limit1 * 2

        if enabled_ops is None:
            # All ops enabled by default if not specified
            enabled_ops = [
                "ANAGRAM", "WORD_DELETION", "LETTER_DELETION", "FIRST_LETTERS", "LAST_LETTERS",
                "MIDDLE_LETTERS", "ALTERNATING_LETTERS", "REVERSAL", "WORD_INSERTION", "HIDDEN",
                "SUBSTITUTION", "CHARADE", "POSITIONAL_DELETION", "LETTER_REPLACEMENT"
            ]

        if num_chunks is None or num_chunks == 1:
            # Get all standard steps and filter them
            std_steps = self._get_all_standard_steps(corpus, t, limit=large_limit, max_fodder_words=max_fodder_words)
            out.extend([s for s in std_steps if s.op in enabled_ops])
            
            # Nested deletions (count as WORD_DELETION for enablement check)
            if "WORD_DELETION" in enabled_ops:
                out.extend(self.get_nested_deletions(corpus, t, limit=large_limit))

        if num_chunks is None or num_chunks > 1:
            if "CHARADE" in enabled_ops:
                out.extend(self.get_concats(corpus, t, limit=50, num_chunks=num_chunks))

        final_steps = self._sort_and_deduplicate(out)
        final_steps = self.apply_llm_refinement(final_steps, corpus=corpus)
        
        return final_steps

    def build_clue(self, corpus, target: str, max_fodder_words: int | None = None) -> List[Step]:
        """
        Produces a sorted list of all possible clue options.
        """
        t = str(target).lower()
        out = self._get_all_standard_steps(corpus, t, max_fodder_words=max_fodder_words)
        out.extend(self.get_nested_deletions(corpus, t))
        
        final_steps = self._sort_and_deduplicate(out)
        final_steps = self.apply_llm_refinement(final_steps, corpus=corpus)
        
        return final_steps
