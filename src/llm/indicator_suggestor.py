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
            "FIRST_LETTERS": "first letters",
            "LAST_LETTERS": "last letters",
            "MIDDLE_LETTERS": "middle letters",
            "ALTERNATING_LETTERS": "alternating letters",
            "LETTER_REPLACEMENT": "letter replacement",
            "SUBSTITUTION": "substitution",
            "CONCAT": "concatenation (joining words)",
            "CHARADE": "charade (joining words)"
        }

        op_desc = op_descriptions.get(step_op, step_op)
        fodder = ", ".join(source_words)
        
        prompt = (
            f"You are a cryptic crossword expert. "
            f"The wordplay operation is: {op_desc}.\n"
            f"The fodder words are: {fodder}.\n"
            f"Suggest 5 contextually appropriate 'indicators' for this operation that would blend well with these fodder words in a surface reading.\n"
            f"For example, if fodder is 'doctor' and 'sober' for an anagram, 'anxious' or 'tipsy' might be good.\n"
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
            return indicators[:5]
        except Exception as e:
            return [f"Error: {str(e)}"]

    def suggest_for_candidate(self, step: Step, candidate: Candidate) -> List[str]:
        # Check if we have a better surface reading from synonyms
        source = candidate.detailed_scores.get("best_surface", candidate.source)
        
        # Remove markers like (beheaded), (even), etc.
        import re
        source_clean = re.sub(r"\s*\(.*?\)", "", source)
        
        # Split into words, keeping phrases
        if " + " in source_clean:
            words = source_clean.split(" + ")
        elif " in " in source_clean:
            words = source_clean.split(" in ")
        else:
            words = source_clean.split()
            
        op = step.op
        # Special case for advanced charades (insertions)
        if op == "CHARADE" and " in " in source_clean:
            op = "WORD_INSERTION"
            
        return self.suggest_indicators(op, words)

    def suggest_synonyms(self, candidate: Candidate, corpus=None) -> dict[str, List[str]]:
        """
        Suggests synonyms for fodder words that could be used for substitutions.
        """
        import re
        source_clean = re.sub(r"\s*\(.*?\)", "", candidate.source).replace("+", " ").replace(" in ", " ")
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
