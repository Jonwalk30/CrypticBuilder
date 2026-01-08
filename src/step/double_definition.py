from __future__ import annotations
import json
from typing import List
from src.step.step import Step, Candidate, BaseStepGenerator

class DoubleDefinitionStep(BaseStepGenerator):
    name = "DOUBLE_DEFINITION"

    def __init__(self, **kwargs):
        super().__init__("double_definition", **kwargs)

    def generate(self, corpus_obj, target: str, *, limit: int = 10, llm_scorer=None) -> Step:
        t = str(target).lower()
        step = Step(op=self.name, target=t)
        
        if not llm_scorer:
            return step

        # Double definitions are purely LLM-based.
        # Let's ask the LLM to provide pairs of definitions that form a natural phrase.
        prompt = (
            f"Provide 5-8 pairs of short definitions for the word '{t}' where each pair, "
            "when concatenated, forms a natural-sounding English phrase or sentence.\n"
            "CRITICAL: The two definitions in each pair MUST NOT be synonyms of each other. "
            "They should ideally represent different senses of the word or at least be distinct concepts.\n"
            "Example for 'trades': 'swaps' and 'jobs' -> 'swaps jobs'. (Correct, swaps != jobs)\n"
            "Example for 'borders': 'edges' and 'limits' -> 'edges limits'. (INCORRECT, edges == limits)\n"
            "Respond ONLY with a JSON object with a key 'pairs' containing a list of strings (each string being the two definitions joined by a space)."
        )
        
        try:
            # We use a lower level call or just rely on cache if possible
            # But since it's unique per target, we'll likely hit the LLM.
            # Check cache first
            cache_key = f"double_defs:{t}"
            if cache_key in llm_scorer.cache:
                pairs = llm_scorer.cache[cache_key]
            else:
                if not llm_scorer.client:
                    return step
                
                print(f"Calling LLM for double definitions of '{t}'...")
                llm_scorer.api_call_count += 1
                response = llm_scorer.client.chat.completions.create(
                    model=llm_scorer.model,
                    messages=[
                        {"role": "system", "content": "You are a cryptic crossword lexicographer. Respond in valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.7
                )
                data = json.loads(response.choices[0].message.content)
                pairs = data.get("pairs", [])
                llm_scorer.cache[cache_key] = pairs
                llm_scorer._save_cache()

            for phrase in pairs:
                # Baseline score
                score = self.op_penalty + 15.0
                step.candidates.append(Candidate(source=phrase, produced=t, score=score))
        
        except Exception as e:
            print(f"Error generating double definitions: {e}")
            # Fallback to simple concatenation of synonyms
            defs = llm_scorer.get_definitions(t)
            if len(defs) >= 2:
                for i in range(min(len(defs), 5)):
                    for j in range(i + 1, min(len(defs), 5)):
                        phrase = f"{defs[i]} {defs[j]}"
                        step.candidates.append(Candidate(source=phrase, produced=t, score=self.op_penalty + 15.0))
        
        step.candidates = sorted(step.candidates, key=lambda c: c.score)[:limit]
        return step

    def apply_llm(self, candidate: Candidate, llm_scorer, corpus=None, target_synonyms: List[str] = None) -> None:
        if not llm_scorer:
            return
        
        phrase = candidate.source
        parts = phrase.split()
        
        # 1. Score coherence of the double definition phrase
        coherence = llm_scorer.score_coherence(phrase)
        
        # 2. Score similarity between the two definitions (we want LOW similarity)
        similarity = 0.0
        if len(parts) == 2:
            similarity = llm_scorer.get_similarity_score(parts[0], parts[1])
        elif len(parts) > 2:
            # For longer phrases, split roughly in half and check similarity?
            # Or just check first and last word if it's a simple split.
            # Most double defs we generate are 2 words.
            pass

        # Penalize similarity: high similarity -> higher score (bad)
        # 0.0 similarity -> 0 penalty
        # 1.0 similarity -> 30 points penalty
        similarity_penalty = similarity * 30.0
        
        # Double definition bonus is weighted higher because it's the whole clue
        bonus = coherence * self.llm_weight * 1.5
        candidate.score -= bonus
        candidate.score += similarity_penalty
        
        candidate.detailed_scores["llm_coherence"] = round(coherence, 3)
        if similarity > 0:
            candidate.detailed_scores["similarity_penalty"] = round(similarity_penalty, 2)
