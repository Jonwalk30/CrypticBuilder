import os
import json
import re
from typing import Dict, List, Tuple
from openai import OpenAI

class LLMScorer:
    def __init__(self, model: str = "gpt-4o-mini", cache_path: str = ".llm_cache.json"):
        self.model = model
        self.cache_path = cache_path
        self.cache = self._load_cache()
        self.api_call_count = 0
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            self.client = None

    def _load_cache(self) -> Dict[str, float]:
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_cache(self):
        try:
            with open(self.cache_path, "w") as f:
                json.dump(self.cache, f)
        except Exception as e:
            print(f"Failed to save LLM cache: {e}")

    def get_contextual_score(self, word1: str, word2: str) -> float:
        """
        Returns a score from 0.0 (unlikely) to 1.0 (very likely)
        representing how likely these two words appear together in context.
        """
        w1 = str(word1).lower().strip()
        w2 = str(word2).lower().strip()
        if not w1 or not w2:
            return 0.0
            
        # Sort words to keep cache key consistent
        sorted_words = sorted([w1, w2])
        key = f"{sorted_words[0]}:{sorted_words[1]}"
        
        if key in self.cache:
            return self.cache[key]

        if not self.client:
            # Fallback if no API key
            return 0.1 # Low baseline for unknown pairs

        try:
            print(f"Calling LLM ({self.model}) for context: '{w1}' and '{w2}'...")
            self.api_call_count += 1
            prompt = f"On a scale of 0 to 100, how likely are the words '{w1}' and '{w2}' to appear in the same sentence or close context in natural English? Respond ONLY with the number."
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a linguistic analyzer. Give a numeric score between 0 and 100."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=10,
                temperature=0
            )
            content = response.choices[0].message.content.strip()
            match = re.search(r"(\d+)", content)
            if match:
                score = float(match.group(1)) / 100.0
                self.cache[key] = score
                self._save_cache()
            else:
                score = 0.0
        except Exception:
            # Don't cache the fallback score on error so it can be retried
            score = 0.0

        return score

    def score_phrase(self, words: List[str]) -> float:
        """
        Scores a phrase by averaging pair-wise contextual scores.
        """
        if len(words) < 2:
            return 1.0
        
        total_score = 0.0
        count = 0
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                total_score += self.get_contextual_score(words[i], words[j])
                count += 1
        
        return total_score / count if count > 0 else 1.0

    def score_coherence(self, phrase: str) -> float:
        """
        Scores how much a phrase makes sense as a natural English sentence or fragment.
        """
        p = phrase.lower().strip()
        if not p:
            return 0.0
            
        key = f"coherence:{p}"
        if key in self.cache:
            return self.cache[key]
            
        if not self.client:
            return 0.1
            
        try:
            print(f"Calling LLM ({self.model}) for coherence: '{p}'...")
            self.api_call_count += 1
            prompt = f"On a scale of 0 to 100, how much does the phrase '{p}' make sense as a natural, coherent English sentence or meaningful fragment? Respond ONLY with the number."
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a linguistic analyzer. Give a numeric score between 0 and 100."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=10,
                temperature=0
            )
            content = response.choices[0].message.content.strip()
            match = re.search(r"(\d+)", content)
            if match:
                score = float(match.group(1)) / 100.0
                self.cache[key] = score
                self._save_cache()
            else:
                score = 0.0
        except Exception:
            score = 0.0
            
        return score

    def get_best_coherence(self, words: List[str], synonyms_map: Dict[str, List[str]] = None) -> Tuple[float, str]:
        """
        Finds the most coherent phrase by potentially substituting words with their synonyms.
        """
        if not synonyms_map:
            phrase = " ".join(words)
            return self.score_coherence(phrase), phrase
            
        # Try all combinations (up to a limit) and return the best score and the phrase.
        base_phrase = " ".join(words)
        best_score = self.score_coherence(base_phrase)
        best_phrase = base_phrase
        
        # To avoid combinatorial explosion, we only try substituting one word at a time.
        # This is usually enough for crossword surface improvements.
        for i, w in enumerate(words):
            if w in synonyms_map:
                for syn in synonyms_map[w]:
                    new_words = list(words)
                    new_words[i] = syn
                    new_phrase = " ".join(new_words)
                    score = self.score_coherence(new_phrase)
                    if score > best_score:
                        best_score = score
                        best_phrase = new_phrase
        
        return best_score, best_phrase

    def get_definitions(self, word: str) -> List[str]:
        """
        Returns a list of common synonyms or brief definitions of the word.
        """
        p = word.lower().strip()
        if not p:
            return []
            
        key = f"definitions:{p}"
        if key in self.cache:
            return self.cache[key]
            
        if not self.client:
            return []
        
        # If we reach here, we are making a single call. 
        # But we'll implement batching below.
        res = self.batch_get_definitions([p])
        return res.get(p, [])

    def batch_get_definitions(self, words: List[str]) -> Dict[str, List[str]]:
        words = [w.lower().strip() for w in words if w.lower().strip()]
        to_fetch = [w for w in words if f"definitions:{w}" not in self.cache]
        
        if not to_fetch or not self.client:
            return {w: self.cache.get(f"definitions:{w}", []) for w in words}

        # Batch in groups of 20
        results = {}
        for i in range(0, len(to_fetch), 20):
            batch = to_fetch[i : i + 20]
            print(f"Calling LLM ({self.model}) for batch definitions: {batch}...")
            self.api_call_count += 1
            
            prompt = (
                f"List 10 common synonyms or brief 1-word definitions for each of the following words: {', '.join(batch)}.\n"
                "Respond ONLY with a JSON object where keys are the words and values are comma-separated strings of synonyms."
            )
            
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a lexicographer. Respond in valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0
                )
                data = json.loads(response.choices[0].message.content)
                for w, syns_str in data.items():
                    defs = [d.strip().lower().strip('"').strip("'") for d in str(syns_str).split(",")]
                    self.cache[f"definitions:{w}"] = defs
                    results[w] = defs
            except Exception as e:
                print(f"Error in batch definitions: {e}")
                
        self._save_cache()
        return {w: self.cache.get(f"definitions:{w}", []) for w in words}

    def batch_get_contextual_scores(self, pairs: List[Tuple[str, str]]) -> Dict[Tuple[str, str], float]:
        unique_pairs = []
        for w1, w2 in pairs:
            sw = sorted([w1.lower().strip(), w2.lower().strip()])
            if sw not in unique_pairs:
                unique_pairs.append(sw)
        
        to_fetch = []
        for sw in unique_pairs:
            key = f"{sw[0]}:{sw[1]}"
            if key not in self.cache:
                to_fetch.append(sw)
        
        if not to_fetch or not self.client:
            out = {}
            for w1, w2 in pairs:
                sw = sorted([w1.lower().strip(), w2.lower().strip()])
                out[(w1, w2)] = self.cache.get(f"{sw[0]}:{sw[1]}", 0.1)
            return out

        # Batch in groups of 50
        for i in range(0, len(to_fetch), 50):
            batch = to_fetch[i : i + 50]
            print(f"Calling LLM ({self.model}) for batch context scores (size {len(batch)})...")
            self.api_call_count += 1
            
            items_str = [f"{p[0]}|{p[1]}" for p in batch]
            prompt = (
                f"On a scale of 0 to 100, how likely are these word pairs to appear in the same sentence or close context in natural English?\n"
                f"Pairs: {', '.join(items_str)}\n"
                "Respond ONLY with a JSON object where keys are the pairs (word1|word2) and values are the numeric scores (0-100)."
            )
            
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a linguistic analyzer. Respond in valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0
                )
                data = json.loads(response.choices[0].message.content)
                for key_str, score in data.items():
                    if "|" in key_str:
                        p = sorted(key_str.lower().split("|"))
                        self.cache[f"{p[0]}:{p[1]}"] = float(score) / 100.0
            except Exception as e:
                print(f"Error in batch context scores: {e}")

        self._save_cache()
        out = {}
        for w1, w2 in pairs:
            sw = sorted([w1.lower().strip(), w2.lower().strip()])
            out[(w1, w2)] = self.cache.get(f"{sw[0]}:{sw[1]}", 0.0)
        return out

    def batch_score_coherence(self, phrases: List[str]) -> Dict[str, float]:
        phrases = [p.lower().strip() for p in phrases if p.lower().strip()]
        to_fetch = [p for p in phrases if f"coherence:{p}" not in self.cache]
        
        if not to_fetch or not self.client:
            return {p: self.cache.get(f"coherence:{p}", 0.1) for p in phrases}

        # Batch in groups of 30
        for i in range(0, len(to_fetch), 30):
            batch = to_fetch[i : i + 30]
            print(f"Calling LLM ({self.model}) for batch coherence (size {len(batch)})...")
            self.api_call_count += 1
            
            prompt = (
                "On a scale of 0 to 100, how much does each of the following phrases make sense as a natural, coherent English sentence or meaningful fragment?\n"
                f"Phrases: {json.dumps(batch)}\n"
                "Respond ONLY with a JSON object where keys are the phrases and values are the numeric scores (0-100)."
            )
            
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a linguistic analyzer. Respond in valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0
                )
                data = json.loads(response.choices[0].message.content)
                for p, score in data.items():
                    self.cache[f"coherence:{p.lower().strip()}"] = float(score) / 100.0
            except Exception as e:
                print(f"Error in batch coherence: {e}")

        self._save_cache()
        return {p: self.cache.get(f"coherence:{p}", 0.0) for p in phrases}

    def print_cost(self):
        print(f"Total LLM API calls in this run: {self.api_call_count}")
