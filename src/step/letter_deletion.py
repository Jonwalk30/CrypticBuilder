from __future__ import annotations

from typing import Tuple, List
from collections import Counter

from src.step.step import Strictness, Step, Candidate, BaseStepGenerator
from src.utils import adjusted_freq, best_leftover_meta, sort_word, subsequence_match_indices, leftover_from_indices, \
    multiset_contains, multiset_leftover_sorted, get_word_display
from src.scoring_config import config

class WordDeletionStep(BaseStepGenerator):
    name = "WORD_DELETION"

    def __init__(self, **kwargs):
        # Determine config section with fallback
        section = "word_deletion"
        if not config.get(section):
            section = "letter_deletion"
        super().__init__(section, **kwargs)
        c = config.get(section)

        self.w_container = kwargs.get("w_container", c.get("w_container", 1.0))
        self.w_leftover = kwargs.get("w_leftover", c.get("w_leftover", 1.0))
        self.unresolved_leftover_penalty_per_char = kwargs.get(
            "unresolved_leftover_penalty_per_char",
            c.get("unresolved_leftover_penalty_per_char", 0.35)
        )
        self.short_leftover_penalty = kwargs.get("short_leftover_penalty", c.get("short_leftover_penalty", 0.0))
        self.obvious_placement_penalty = kwargs.get("obvious_placement_penalty", c.get("obvious_placement_penalty", 10.0))
        self.max_scan = kwargs.get("max_scan", c.get("max_scan", 50_000))
        self.prefer_stricter = kwargs.get("prefer_stricter", c.get("prefer_stricter", True))

        self.strictness_bonus = {
            "SUBSTRING": c.get("substring_bonus", 5.0),
            "SUBSEQUENCE": c.get("subsequence_bonus", -5.0),
            "MULTISET": c.get("multiset_bonus", 0.0),
        }

    def _score(self, corpus, row, leftover_sorted: str, strictness: Strictness, target: str, llm_scorer=None) -> tuple[float, Tuple[str, ...], dict[str, float]]:
        cont_adj = self._get_adj(row)
        container = str(row.entry) if hasattr(row, 'entry') else str(row["entry"])

        leftover_words, lo_best_freq, lo_best_stop, lo_is_name = best_leftover_meta(corpus, leftover_sorted)
        lo_adj = adjusted_freq(lo_best_freq, lo_best_stop, self.stopword_penalty, self.stopword_power, is_proper_noun=lo_is_name) if lo_best_freq > 0 else 0.0

        # unresolved leftover penalty if there are leftover letters but no word anagram found
        unresolved_cost = 0.0
        if leftover_sorted and not leftover_words:
            unresolved_cost = self.unresolved_leftover_penalty_per_char * len(leftover_sorted)
        
        # Penalize short leftovers (less than 3 chars) even if they are words
        short_penalty = 0.0
        if 0 < len(leftover_sorted) < 3:
            short_penalty = self.short_leftover_penalty

        # Obvious placement penalty: container starts or ends with target
        placement_penalty = 0.0
        if container.startswith(target) or container.endswith(target):
            placement_penalty = self.obvious_placement_penalty

        # GOLF SCORING:
        # High frequency is low cost. 
        # Cost = (20 - container_freq) + (20 - leftover_freq) + op_cost + unresolved_cost + short_penalty + placement_penalty - strictness_discount
        cost = (
            self.w_container * (20.0 - cont_adj)
            + self.w_leftover * (20.0 - lo_adj)
            + self.op_penalty
            + unresolved_cost
            + short_penalty
            + placement_penalty
            - float(self.strictness_bonus[strictness])
        )
        
        detailed = {
            "container_freq": round(cont_adj, 2),
            "leftover_freq": round(lo_adj, 2),
        }

        return cost, leftover_words, detailed

    def generate(self, corpus, target: str, *, limit: int = 200, forbidden_source: str | None = None, llm_scorer=None) -> Step:
        from src.corpus import corpus as corpus_df
        t = str(target).lower()
        df = corpus_df.corpus
        step = Step(op=self.name, target=t)

        df2 = df[df["entry"].str.len() >= len(t)][["entry", "frequency", "stopword_ratio_entry", "is_proper_noun"]]
        
        if forbidden_source:
            df2 = df2[df2["entry"] != forbidden_source.lower()]

        # Sort by frequency to ensure we find the "best" candidates first during limited scans
        df2 = df2.sort_values("frequency", ascending=False)

        # 2) SUBSEQUENCE (scan limited)
        scanned = 0
        df_seq = df2[df2["entry"].str.contains(t[0], regex=False) & df2["entry"].str.contains(t[-1], regex=False)]
        for row in df_seq.itertuples(index=False):
            scanned += 1
            if scanned > self.max_scan:
                break
            container = str(row.entry)
            
            # Exclude substring matches as they are "too easy and boring"
            if container.find(t) != -1:
                continue
                
            idxs = subsequence_match_indices(container, t)
            if idxs is None:
                continue
            
            strictness = "SUBSEQUENCE"

            leftover = leftover_from_indices(container, idxs)
            leftover_sorted = sort_word(leftover)
            score, leftover_words, detailed = self._score(corpus, row, leftover_sorted, strictness, t, llm_scorer)
            if not leftover_words:
                continue
            
            # Format source to show subtraction: container - removed
            container_display = get_word_display(corpus, container)
            removed_display = leftover_words[0]
            source_display = f"{container_display} - [{removed_display}]"

            step.candidates.append(
                Candidate(
                    source=source_display,
                    produced=t,
                    leftover_sorted=leftover_sorted,
                    leftover_words=leftover_words,
                    strictness=strictness,
                    score=score,
                    detailed_scores=detailed,
                )
            )

        # 3) MULTISET (scan limited)
        scanned = 0
        for row in df2.itertuples(index=False):
            scanned += 1
            if scanned > self.max_scan:
                break
            container = str(row.entry)
            
            # Exclude substring matches
            if container.find(t) != -1:
                continue

            # Check if already covered by stricter match
            if self.prefer_stricter:
                if subsequence_match_indices(container, t):
                    continue
            
            if not multiset_contains(container, t):
                continue
            leftover_sorted = multiset_leftover_sorted(container, t)
            score, leftover_words, detailed = self._score(corpus, row, leftover_sorted, "MULTISET", t, llm_scorer)
            if not leftover_words:
                continue

            # Format source to show subtraction: container - removed
            container_display = get_word_display(corpus, container)
            removed_display = leftover_words[0]
            source_display = f"{container_display} - ({removed_display})"

            step.candidates.append(
                Candidate(
                    source=source_display,
                    produced=t,
                    leftover_sorted=leftover_sorted,
                    leftover_words=leftover_words,
                    strictness="MULTISET",
                    score=score,
                    detailed_scores=detailed,
                )
            )

        # Deduplicate: keep best score per (source, strictness, leftover_sorted)
        uniq = {}
        for c in step.candidates:
            k = (c.source, c.strictness, c.leftover_sorted)
            if k not in uniq or c.score < uniq[k].score:
                uniq[k] = c
        step.candidates = sorted(uniq.values(), key=lambda c: c.score, reverse=False)[:limit]
        return step

    def apply_llm(self, candidate: Candidate, llm_scorer, corpus=None, target_synonyms: List[str] = None) -> None:
        if not llm_scorer or not candidate.leftover_words:
            return
        
        # Use the best leftover word for context with the container
        best_lo = candidate.leftover_words[0].split(" (")[0]
        
        from src.utils import clean_source_fodder
        source_clean = clean_source_fodder(candidate.source)
        # For deletions like "container - removed", we want context between container and removed
        # source_clean will be "container removed"
        parts = source_clean.split()
        if len(parts) >= 2:
            container = parts[0]
            # Check context (ONLY original words)
            context = llm_scorer.get_contextual_score(container, best_lo)

            bonus = context * self.llm_weight
            candidate.score -= bonus
            candidate.detailed_scores["llm_context"] = round(context, 3)
        
        # Also check for definition context
        super().apply_llm(candidate, llm_scorer, corpus, target_synonyms)


class LetterDeletionStep(BaseStepGenerator):
    name = "LETTER_DELETION"

    def __init__(self, **kwargs):
        super().__init__("letter_deletion", **kwargs)
        c = config.get("letter_deletion")
        self.w_container = kwargs.get("w_container", c.get("w_container", 1.0))

    def generate(self, corpus, target: str, *, limit: int = 200, forbidden_source: str | None = None, llm_scorer=None) -> Step:
        from src.corpus import corpus as corpus_df
        t = str(target).lower()
        df = corpus_df.corpus
        step = Step(op=self.name, target=t)
        n = len(t)

        # Optimization: the container must be longer than target.
        possible_containers = df[df["entry"].str.len() > n]
        
        if forbidden_source:
            possible_containers = possible_containers[possible_containers["entry"] != forbidden_source.lower()]
        
        # 1. Positional Deletions (First, Last, First & Last)
        # Beheaded: container[1:] == t
        mask_behead = (possible_containers["entry"].str.len() == n + 1) & (possible_containers["entry"].str.endswith(t))
        for row in possible_containers[mask_behead].itertuples(index=False):
            adj = self._get_adj(row)
            cost = self.w_container * (20.0 - adj) + self.op_penalty
            step.candidates.append(Candidate(source=f"{row.entry} (beheaded)", produced=t, score=cost, strictness="SUBSTRING", detailed_scores={"freq": round(adj, 2)}))

        # Curtailed: container[:-1] == t
        mask_curtail = (possible_containers["entry"].str.len() == n + 1) & (possible_containers["entry"].str.startswith(t))
        for row in possible_containers[mask_curtail].itertuples(index=False):
            adj = self._get_adj(row)
            cost = self.w_container * (20.0 - adj) + self.op_penalty
            step.candidates.append(Candidate(source=f"{row.entry} (curtailed)", produced=t, score=cost, strictness="SUBSTRING", detailed_scores={"freq": round(adj, 2)}))

        # Heart: container[1:-1] == t
        mask_heart = (possible_containers["entry"].str.len() == n + 2) & (possible_containers["entry"].str.slice(1, -1) == t)
        for row in possible_containers[mask_heart].itertuples(index=False):
            adj = self._get_adj(row)
            cost = self.w_container * (20.0 - adj) + self.op_penalty
            step.candidates.append(Candidate(source=f"{row.entry} (heart)", produced=t, score=cost, strictness="SUBSTRING", detailed_scores={"freq": round(adj, 2)}))

        # 2. Letter removal (removing all instances of a single letter)
        scanned = 0
        for row in possible_containers.itertuples(index=False):
            scanned += 1
            if scanned > 50000:
                break
            container = str(row.entry)
            if len(step.candidates) >= limit * 2:
                break
            
            # Find which letters are in container but not in target
            c_counts = Counter(container)
            t_counts = Counter(t)
            
            diff = c_counts - t_counts
            if len(diff) != 1:
                continue
            
            extra_letter, count = list(diff.items())[0]
            
            # Check if removing this letter from container gives target
            if container.replace(extra_letter, "") == t:
                # Exclude substring matches as they are "too easy and boring"
                if container.find(t) != -1:
                    continue
                    
                # Match found!
                adj = self._get_adj(row)
                cost = self.w_container * (20.0 - adj) + self.op_penalty
                
                step.candidates.append(
                    Candidate(
                        source=f"{container} (-{extra_letter.upper()}s)",
                        produced=t,
                        score=cost,
                        strictness="SUBSEQUENCE",
                        detailed_scores={"freq": round(adj, 2)}
                    )
                )
            
        step.candidates.sort(key=lambda c: c.score)
        step.candidates = step.candidates[:limit]
        return step
