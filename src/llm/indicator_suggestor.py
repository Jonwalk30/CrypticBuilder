from typing import List
from src.step.step import Step, Candidate

class IndicatorSuggestor:
    def __init__(self, llm_scorer):
        self.llm_scorer = llm_scorer

    def suggest_indicators(self, step_op: str, source_words: List[str]) -> List[str]:
        """
        Suggests cryptic indicators that work well contextually with the source words.
        """
        if not self.llm_scorer or not self.llm_scorer.client:
            return ["(LLM not configured)"]

        op_descriptions = {
            "ANAGRAM": "anagram (reordering letters)",
            "WORD_DELETION": "deletion (removing a word from another)",
            "LETTER_DELETION": "letter deletion (removing first, last, or specific letters)",
            "REVERSAL": "reversal (flipping the word)",
            "WORD_INSERTION": "insertion (putting one word inside another)",
            "HIDDEN": "hidden word (found inside a phrase)",
            "FIRST_LETTERS": "selection of first letters (e.g., 'initially', 'starting')",
            "LAST_LETTERS": "selection of last letters (e.g., 'finally', 'at the end')",
            "MIDDLE_LETTERS": "selection of middle letters (e.g., 'at heart', 'internally')",
            "ALTERNATING_LETTERS": "alternating letters (e.g., 'oddly', 'evenly', 'regularly')",
            "LETTER_REPLACEMENT": "letter replacement",
            "SUBSTITUTION": "substitution",
            "CONCAT": "concatenation (joining words)",
            "CHARADE": "charade (joining words)"
        }

        op_desc = op_descriptions.get(step_op, step_op)
        fodder = ", ".join(source_words)
        
        # Check cache
        cache_key = f"indicators:{step_op}:{fodder}"
        if self.llm_scorer and cache_key in self.llm_scorer.cache:
            return self.llm_scorer.cache[cache_key]

        prompt = (
            f"You are a cryptic crossword expert. "
            f"The wordplay operation is: {op_desc}.\n"
            f"The fodder words are: {fodder}.\n"
            f"Suggest 5 appropriate 'indicators' for this operation.\n"
            f"Requirements:\n"
            f"1. Some should be contextually appropriate and blend well with the fodder words in a surface reading.\n"
            f"2. Some should be standard, recognizable cryptic indicators for this operation type.\n"
            f"For example, if fodder is 'doctor' and 'sober' for an anagram, 'anxious' or 'tipsy' might be good.\n"
            f"If it's alternating letters, 'oddly' or 'evenly' are good.\n"
            f"Return ONLY a comma-separated list of the 5 indicators."
        )

        try:
            print(f"Calling LLM ({self.llm_scorer.model}) for indicator suggestions...")
            self.llm_scorer.api_call_count += 1
            response = self.llm_scorer.client.chat.completions.create(
                model=self.llm_scorer.model,
                messages=[
                    {"role": "system", "content": "You are a cryptic crossword assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=60,
                temperature=0.7
            )
            content = response.choices[0].message.content.strip()
            # Basic cleanup in case it added numbers or extra text
            indicators = [i.strip().strip('"').strip("'") for i in content.split(",")]
            
            # Save to cache
            if self.llm_scorer:
                with self.llm_scorer.lock:
                    self.llm_scorer.cache[cache_key] = indicators[:5]
                    self.llm_scorer._save_cache()
            
            return indicators[:5]
        except Exception as e:
            return [f"Error: {str(e)}"]

    def suggest_for_candidate(self, step: Step, candidate: Candidate) -> List[str]:
        # Check if we have a better surface reading from synonyms
        source = candidate.detailed_scores.get("best_surface", candidate.source)
        
        from src.utils import clean_source_fodder
        source_clean = clean_source_fodder(source)
        words = source_clean.split()
            
        op = step.op
        # Special case for advanced charades (insertions)
        if op == "CHARADE" and " in " in candidate.source:
            op = "WORD_INSERTION"
            
        return self.suggest_indicators(op, words)

    def suggest_synonyms(self, candidate: Candidate, corpus=None) -> dict[str, List[str]]:
        """
        Suggests synonyms for fodder words that could be used for substitutions.
        """
        from src.utils import clean_source_fodder
        source_clean = clean_source_fodder(candidate.source)
        words = source_clean.split()
        
        results = {}
        for w in words:
            # We want to check even single letters if they are in the substitution list
            # because they might be part of the wordplay (e.g. 'f' -> 'fellow')
            syns = []
            
            # 1. Check substitutions.yml
            if corpus and w in corpus.substitutions:
                syns.extend(corpus.substitutions[w])
            
            # 2. Check LLM definitions (only for words length >= 2 to avoid junk)
            if self.llm_scorer and len(w) >= 2:
                llm_syns = self.llm_scorer.get_definitions(w)
                if llm_syns:
                    for s in llm_syns:
                        if s not in syns:
                            syns.append(s)
            
            if syns:
                results[w] = syns[:10]
        return results
