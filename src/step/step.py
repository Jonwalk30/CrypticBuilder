from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Literal, Dict

from src.corpus import Corpus
from src.utils import pretty
from src.scoring_config import config

# -------------------------
# Data containers
# -------------------------

Strictness = Literal["SUBSTRING", "SUBSEQUENCE", "MULTISET"]

@dataclass
class Candidate:
    source: str
    produced: str
    leftover_sorted: str = ""                   # letters left over (signature)
    leftover_words: Tuple[str, ...] = ()        # anagram-resolved leftover words (if any)
    strictness: Strictness = "MULTISET"
    score: float = 0.0
    detailed_scores: Dict[str, float] = field(default_factory=dict)

@dataclass
class Step:
    op: str
    target: str
    candidates: List[Candidate] = field(default_factory=list)
    child: Optional["Step"] = None              # depth-2: clue leftover
    chunk_steps: List["Step"] = field(default_factory=list) # for CONCAT/CHARADE tree
    combined_score: Optional[float] = None      # for nested steps, store combined score

    def best(self) -> Optional[Candidate]:
        return min(self.candidates, key=lambda c: c.score, default=None)

    def best_score(self) -> float:
        if self.combined_score is not None:
            return float(self.combined_score)
        b = self.best()
        return float(b.score) if b else 1e18

    @property
    def complexity(self) -> int:
        if self.op == "CHARADE":
            return len(self.chunk_steps)
        if self.child:
            return 2 # Nested is currently 2-step
        return 1


class BaseStepGenerator:
    name: str = "BASE"

    def __init__(self, config_section: str, **kwargs):
        c = config.get(config_section)
        common = config.get("common")
        
        self.op_penalty = kwargs.get("op_penalty", c.get("op_penalty", 0.0))
        self.stopword_penalty = kwargs.get("stopword_penalty", common.get("stopword_penalty", 2.0))
        self.stopword_power = kwargs.get("stopword_power", common.get("stopword_power", 1.0))
        self.llm_weight = kwargs.get("llm_weight", c.get("llm_weight", common.get("llm_weight", 10.0)))

    def _get_adj(self, row) -> float:
        from src.utils import adjusted_freq
        is_n = bool(row["is_proper_noun"]) if "is_proper_noun" in (row.index if hasattr(row, 'index') else row.keys()) else False
        return adjusted_freq(float(row["frequency"]), float(row["stopword_ratio_entry"]),
                             self.stopword_penalty, self.stopword_power, is_proper_noun=is_n)

    def _get_word_adj(self, corpus, word: str) -> Optional[float]:
        df = corpus.corpus
        row = df[df["entry"] == word]
        if row.empty:
            return None
        return self._get_adj(row.iloc[0])

    def apply_llm(self, candidate: Candidate, llm_scorer, corpus=None, target_synonyms: List[str] = None) -> None:
        """
        Default implementation for applying LLM scoring to a candidate.
        Most wordplay types care about the context between the source word(s) and the target word.
        """
        if not llm_scorer:
            return

        source = candidate.source
        # Strip markers like (beheaded), (even), (-S), etc.
        source_clean = source.split(" (")[0]
        target = candidate.produced

        # 1. Context bonus (using the words themselves)
        words = source_clean.replace("+", " ").replace(" in ", " ").split()
        
        # Check context score for the whole cleaned source against target
        best_context_score = llm_scorer.get_contextual_score(source_clean, target)

        bonus = best_context_score * self.llm_weight
        candidate.score -= bonus
        candidate.detailed_scores["llm_context"] = round(best_context_score, 3)

        # 2. Definition Context bonus (fodder words vs target synonyms)
        if target_synonyms:
            best_def_match = 0.0
            for w in words:
                for ts in target_synonyms:
                    match_score = llm_scorer.get_contextual_score(w, ts)
                    if match_score > best_def_match:
                        best_def_match = match_score
            
            if best_def_match > 0.5:
                # Reward contextual fodder-definition connection
                def_bonus = best_def_match * 10.0 # up to 10 points bonus
                candidate.score -= def_bonus
                candidate.detailed_scores["definition_context"] = round(best_def_match, 3)

    def _get_synonyms_map(self, words: List[str], corpus, llm_scorer) -> Dict[str, List[str]]:
        results = {}
        for w in words:
            syns = []
            # 1. From substitutions.yml (if available)
            if corpus and w in corpus.substitutions:
                syns.extend(corpus.substitutions[w])
            
            # 2. From LLM
            if llm_scorer:
                llm_syns = llm_scorer.get_definitions(w)
                if llm_syns:
                    for s in llm_syns:
                        if s not in syns:
                            syns.append(s)
            
            if syns:
                results[w] = syns[:5] # Limit to top 5
        return results


if __name__ == "__main__":

    # for s in steps[:10]:
    #     print(pretty(s))
    #     print("----")
    pass