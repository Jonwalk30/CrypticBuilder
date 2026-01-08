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
from src.step.double_definition import DoubleDefinitionStep
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
        use_llm_ranking: bool = False,
        use_llm_suggestions: bool = True,
        **kwargs
    ):
        self.limit1 = limit1
        self.limit_child = limit_child
        self.top_k = top_k
        self.depth_penalty = depth_penalty
        self.max_fodder_words = max_fodder_words
        
        self.use_llm_ranking = use_llm_ranking
        self.use_llm_suggestions = use_llm_suggestions

        self.llm_scorer = llm_scorer
        if (self.use_llm_ranking or self.use_llm_suggestions) and not self.llm_scorer:
            self.llm_scorer = LLMScorer()
        
        self.indicator_suggestor = IndicatorSuggestor(self.llm_scorer) if (self.use_llm_suggestions and self.llm_scorer) else None

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
        self.dd = DoubleDefinitionStep()
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

    def get_double_definitions(self, corpus, target: str, limit: int = None) -> List[Step]:
        return self._get_step(self.dd, corpus, target, limit=limit, llm_scorer=self.llm_scorer)

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

    def _get_all_standard_steps(self, corpus, target: str, limit: int = None, max_fodder_words: int | None = None, original_target: str | None = None) -> List[Step]:
        out = []
        t_for_dd = original_target or target
        
        out.extend(self.get_anagrams(corpus, target, limit=limit, max_fodder_words=max_fodder_words))
        out.extend(self.get_word_deletions(corpus, target, limit=limit))
        out.extend(self.get_word_insertions(corpus, target, limit=limit))
        out.extend(self.get_letter_deletions(corpus, target, limit=limit))
        out.extend(self.get_letter_replacements(corpus, target, limit=limit))
        out.extend(self.get_hiddens(corpus, target, limit=limit))
        out.extend(self.get_reversals(corpus, target, limit=limit))
        out.extend(self.get_substitutions(corpus, target, limit=limit))
        out.extend(self.get_alternating_letters(corpus, target, limit=limit))
        out.extend(self.get_double_definitions(corpus, t_for_dd, limit=limit))
        if self.use_positional_selection:
            out.extend(self.get_first_letters(corpus, target, limit=limit))
            out.extend(self.get_last_letters(corpus, target, limit=limit))
            out.extend(self.get_middle_letters(corpus, target, limit=limit))
        out.extend(self.get_positional_deletions(corpus, target, limit=limit))
        return out

    def _sort_and_deduplicate(self, steps: List[Step]) -> List[Step]:
        # Sort overall steps by their best available score (combined if nested)
        # Filter out steps with ridiculously high scores (meaning they failed but were included)
        steps = [s for s in steps if s.best_score() < 1e9]
        steps.sort(key=lambda s: s.best_score(), reverse=False)

        # Deduplicate: for each (source, leftover), keep only the one with the best score
        seen = {}
        unique_out = []
        for s in steps:
            best_cand = s.best()
            if not best_cand:
                continue
            # Some candidates might not have leftover_sorted (e.g. Anagram)
            key = (s.op, best_cand.source, best_cand.leftover_sorted)
            if key not in seen:
                seen[key] = True
                unique_out.append(s)
        return unique_out

    def apply_llm_refinement(self, steps: List[Step], corpus=None, limit: int = 50, score_threshold: float = 100.0) -> List[Step]:
        if not self.llm_scorer or not self.use_llm_ranking:
            return steps
            
        # 1. Identify candidates to refine
        to_refine = []
        # We'll look through the steps and pick the top 'limit' ones that ARE NOT skipped
        for s in steps:
            if len(to_refine) >= limit: break
            
            # Use a higher default threshold or allow all top-K
            if s.best_score() > score_threshold: continue
            
            # Skip AI for positional letters if disabled
            if not self.use_llm_for_positional and s.op in ["FIRST_LETTERS", "LAST_LETTERS", "MIDDLE_LETTERS"]:
                continue

            best_c = s.best()
            if not best_c: continue
            
            to_refine.append((s, best_c))
            
        if not to_refine:
            return steps
            
        # 2. Prefetch all LLM data in batches
        self._prefetch_llm_data(to_refine, corpus)
        
        op_map = {
            "ANAGRAM": self.an, "WORD_DELETION": self.wd, "LETTER_DELETION": self.ld,
            "FIRST_LETTERS": self.fl, "LAST_LETTERS": self.ll, "MIDDLE_LETTERS": self.ml,
            "ALTERNATING_LETTERS": self.al, "REVERSAL": self.rv, "WORD_INSERTION": self.wi,
            "POSITIONAL_DELETION": self.pd, "LETTER_REPLACEMENT": self.lr, "CHARADE": self.co,
            "HIDDEN": self.hi, "SUBSTITUTION": self.su, "DOUBLE_DEFINITION": self.dd
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
        
        # 1. Fetch definitions for TARGET only (needed for definition context bonus)
        self.llm_scorer.batch_get_definitions([target])
        
        # 2. Collect all pairs that apply_llm will ask for
        target_syns = [target] + self.llm_scorer.cache.get(f"definitions:{target}", [])[:5]
        all_pairs = []
        all_phrases = []
        all_sim_pairs = []
        
        for s, c in to_refine:
            source_clean = re.sub(r"\s*\(.*?\)", "", c.source).replace("+", " ").replace(" in ", " ")
            words = source_clean.split()
            all_phrases.append(source_clean)
            
            # Default context for source_clean vs target
            all_pairs.append((source_clean, target))
            
            # Definition context pairs
            for w in words:
                for ts in target_syns:
                    all_pairs.append((w, ts))

            # Special case for DOUBLE_DEFINITION similarity
            if s.op == "DOUBLE_DEFINITION" and len(words) >= 2:
                all_sim_pairs.append((words[0], words[1]))

            # Special case for WORD_DELETION/INSERTION parts context
            if s.op in ["WORD_DELETION", "WORD_INSERTION"] and c.leftover_words:
                best_lo = c.leftover_words[0].split(" (")[0]
                container = source_clean
                if s.op == "WORD_INSERTION" and " in " in source_clean:
                    container = source_clean.split(" in ")[1]
                all_pairs.append((best_lo, container))
            
            # Positional selection target context
            if s.op in ["FIRST_LETTERS", "LAST_LETTERS", "MIDDLE_LETTERS"]:
                for w in words:
                    all_pairs.append((w, target))

        # 3. Batch fetch everything
        self.llm_scorer.batch_get_contextual_scores(all_pairs)
        self.llm_scorer.batch_score_coherence(all_phrases)
        if all_sim_pairs:
            self.llm_scorer.batch_get_similarity_scores(all_sim_pairs)

    def find_all(self, corpus, target: str, num_chunks: int | None = None, max_fodder_words: int | None = None, enabled_ops: List[str] | None = None, progress_callback=None):
        """
        Produces a sorted list of all possible clue options with a larger limit.
        Yields results incrementally if possible.
        """
        t = str(target).lower()
        t_clean = t.replace(" ", "")
        out: List[Step] = []

        if progress_callback:
            progress_callback(0.1, "Initializing search...")

        # Increase limits for find_all
        large_limit = self.limit1 * 2

        if enabled_ops is None:
            # All ops enabled by default if not specified
            enabled_ops = [
                "ANAGRAM", "WORD_DELETION", "LETTER_DELETION", "FIRST_LETTERS", "LAST_LETTERS",
                "MIDDLE_LETTERS", "ALTERNATING_LETTERS", "REVERSAL", "WORD_INSERTION", "HIDDEN",
                "SUBSTITUTION", "CHARADE", "POSITIONAL_DELETION", "LETTER_REPLACEMENT", "DOUBLE_DEFINITION"
            ]

        # 1-chunk steps
        if num_chunks is None or num_chunks >= 1:
            if progress_callback:
                progress_callback(0.2, "Searching 1-chunk patterns...")
            
            # Use a slightly more conservative limit for 1-chunk standard steps
            # to avoid flooding with low-quality hiddens/positionals.
            std_limit = self.limit1
            std_steps = self._get_all_standard_steps(corpus, t_clean, limit=std_limit, max_fodder_words=max_fodder_words, original_target=t)
            
            # Restore the original target with spaces for display
            for s in std_steps:
                s.target = t
                for c in s.candidates:
                    c.produced = t

            filtered_1chunk = [s for s in std_steps if s.op in enabled_ops]
            out.extend(filtered_1chunk)
            
            # Nested deletions (count as WORD_DELETION for enablement check)
            if "WORD_DELETION" in enabled_ops:
                nested = self.get_nested_deletions(corpus, t_clean, limit=large_limit)
                for s in nested:
                    s.target = t
                    for c in s.candidates:
                        c.produced = t
                out.extend(nested)

            # Yield 1-chunk results immediately
            if progress_callback:
                current = self._sort_and_deduplicate(out)
                progress_callback(0.4, f"Found {len(current)} 1-chunk patterns. Searching 2-chunk...", current)

        # 2+ chunk steps
        if num_chunks is None or num_chunks >= 2:
            if "CHARADE" in enabled_ops:
                max_n = num_chunks or 3 # Reduced default from 4 to 3 for performance
                for n in range(2, max_n + 1):
                    if progress_callback:
                        current = self._sort_and_deduplicate(out)
                        progress_callback(0.4 + (0.3 * (n-1)/max_n), f"Searching {n}-chunk patterns...", current)
                    # Concat handles the target as is
                    out.extend(self.get_concats(corpus, t, limit=50, num_chunks=n))

        if progress_callback:
            progress_callback(0.8, "Sorting and deduplicating...")
        final_steps = self._sort_and_deduplicate(out)
        
        if progress_callback:
            progress_callback(0.85, "Applying AI refinement (this may take a moment)...", final_steps)
        
        final_steps = self.apply_llm_refinement(final_steps, corpus=corpus)
        
        if progress_callback:
            progress_callback(1.0, "Search complete!", final_steps)
        
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
