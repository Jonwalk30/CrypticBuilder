from __future__ import annotations

from collections import Counter, defaultdict
from typing import Optional, List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from src.step.step import Step


def sort_word(s: str) -> str:
    return "".join(sorted(s))


def normalize(s: str) -> str:
    import re
    # Lowercase and remove non-alphabetic characters (except maybe spaces if allowed, but usually not for entries)
    s = str(s).lower().strip()
    return re.sub(r"[^a-z]", "", s)


def global_frequency(word: str, lang: str = "en") -> float:
    import wordfreq
    return float(wordfreq.zipf_frequency(word, lang))


def is_stopword_entry(word: str) -> bool:
    # Common crossword stopwords/fillers
    stopwords = {
        "a", "about", "all", "also", "an", "and", "any", "as", "at", "be",
        "because", "but", "by", "can", "come", "could", "day", "do", "even",
        "find", "first", "for", "from", "get", "give", "go", "have", "he",
        "her", "here", "him", "his", "how", "i", "if", "in", "into", "it",
        "its", "just", "know", "like", "look", "make", "man", "many", "me",
        "more", "my", "new", "no", "not", "now", "of", "on", "one", "only",
        "or", "other", "our", "out", "over", "people", "say", "see", "she",
        "so", "some", "take", "tell", "than", "that", "the", "their", "them",
        "then", "there", "these", "they", "thing", "think", "this", "those",
        "time", "to", "two", "up", "use", "very", "want", "way", "we", "well",
        "went", "what", "when", "which", "who", "will", "with", "would",
        "year", "you", "your"
    }
    return word.lower() in stopwords


def stopword_ratio(word: str) -> float:
    # For now, binary. Could be more sophisticated.
    return 1.0 if is_stopword_entry(word) else 0.0


def adjusted_freq(freq: float, stop_ratio: float, penalty: float = 2.0, power: float = 1.0, is_proper_noun: bool = False, name_penalty: float = 0.5) -> float:
    score = float(freq) - float(penalty) * (float(stop_ratio) ** float(power))
    if is_proper_noun:
        score -= name_penalty
    return score


def multiset_contains(container: str, target: str) -> bool:
    cc = Counter(container)
    tc = Counter(target)
    return all(cc[ch] >= n for ch, n in tc.items())


def multiset_leftover_sorted(container: str, target: str) -> str:
    c = Counter(container)
    c.subtract(Counter(target))
    return "".join(ch * n for ch, n in sorted((k, v) for k, v in c.items() if v > 0))


def subsequence_match_indices(container: str, target: str) -> Optional[List[int]]:
    if not target:
        return []
    idxs = []
    j = 0
    for i, ch in enumerate(container):
        if ch == target[j]:
            idxs.append(i)
            j += 1
            if j == len(target):
                return idxs
    return None


def leftover_from_indices(container: str, used_indices: List[int]) -> str:
    used = set(used_indices)
    return "".join(ch for i, ch in enumerate(container) if i not in used)


def get_word_display(corpus, word: str) -> str:
    if corpus and word in corpus.substitutions:
        return f"{word} (subst. {', '.join(corpus.substitutions[word][:2])})"
    return word


def best_leftover_meta(corpus, leftover_sorted: str) -> tuple[Tuple[str, ...], float, float, bool]:
    words = list(corpus._anagram_words_sorted.get(leftover_sorted, []))
    
    # Check if the leftover itself is a substitution word (if it's already sorted, this might not work as intended
    # if the substitution key is not sorted, but usually they are short and literal)
    # Actually, substitutions are literal strings. leftover_sorted is sorted letters.
    # To match substitutions, we need to check if any anagram of leftover_sorted is in substitutions.
    
    sub_words = []
    # This is a bit slow but okay for short leftovers
    # For a given signature, we already have all anagram words in 'words'.
    # Let's check which of those words are in substitutions.
    for w in words:
        if w in corpus.substitutions:
            sub_words.append(f"{w} (subst. {', '.join(corpus.substitutions[w][:2])})")
    
    # Also check if the sorted signature itself matches any substitution keys (less likely but possible)
    # but the user said "consider the 2 letter or longer substrings in the substitution list as words"
    # This means if 'al' is a substitution key, it can be treated as a word.
    
    # If no word was found in the corpus, but it IS a substitution key, we add it.
    # We need to find if any permutation of leftover_sorted is a substitution key.
    # Since we don't have a signature-to-substitution-key map, we might need one or just iterate.
    # Let's assume we can add to words.
    
    # To be efficient, let's just check if any of the existing anagram words are substitutions
    # and maybe check the literal strings if they are missing.
    
    all_potential_words = list(words)
    # If the signature itself (if length >= 2) is a substitution key, we should ideally find it.
    # But substitutions are usually not anagrammed in the user's mind.
    # "we can delete 'al' (Capone) from tonal to leave the word ton, even though 'al' wasn't previously a word"
    # In 'tonal', deleting 'ton' leaves 'al'. 'al' sorted is 'al'.
    
    # Let's just check all substitution keys that have this signature.
    for k in corpus.substitutions:
        if sort_word(k) == leftover_sorted:
            if k not in all_potential_words:
                all_potential_words.append(k)

    # Now format them
    final_words = [get_word_display(corpus, w) for w in all_potential_words]

    meta = corpus._anagram_best.get(leftover_sorted) or {}
    return (tuple(final_words),
            float(meta.get("best_freq", 0.0)),
            float(meta.get("best_stop_ratio", 0.0)),
            bool(meta.get("best_is_proper_noun", False)))


def clean_source_fodder(source: str) -> str:
    """
    Cleans a wordplay source string to leave only potential fodder words.
    Removes operational markers (like '(beheaded)', '(even)') and structural 
    symbols (+, -, in, [], ()) while preserving the words within them.
    """
    import re
    # 1. Remove common operation markers that are NOT fodder
    markers = [
        r"\(beheaded\)", r"\(curtailed\)", r"\(heart\)", r"\(gutted\)", 
        r"\(even\)", r"\(odd\)", 
        r"\(subst\. .*?\)",
        r"\(-[A-Z]+s?\)",
        r"\(.*?\->.*?\)", # for LETTER_REPLACEMENT e.g. (E->A)
        r"\(subst\. avail\)"
    ]
    cleaned = source
    for m in markers:
        cleaned = re.sub(m, "", cleaned, flags=re.IGNORECASE)
    
    # Remove asterisk markers often used for substitutions
    cleaned = cleaned.replace("*", "")
    
    # 2. Handle specific structural words
    cleaned = cleaned.replace(" + ", " ").replace(" in ", " ").replace(" - ", " ")
    
    # 3. Strip remaining structural symbols but keep content
    for char in "[]()":
        cleaned = cleaned.replace(char, " ")
        
    return cleaned.strip()


def pretty(step: Step, indent: str = "  ") -> str:
    lines: List[str] = []

    def rec(s: Step, level: int):
        pref = indent * level
        best = s.best()
        target_str = s.target.upper() if s.target else "???"
        lines.append(f"{pref}{target_str}")
        
        if best is None and not s.chunk_steps:
            lines.append(f"{pref}{s.op}: (no candidates)")
            return
            
        extra = f" [{best.strictness}]" if best and s.op in ("LETTER_DELETION", "WORD_DELETION", "WORD_INSERTION") else ""
        
        score_info = ""
        if best and best.detailed_scores:
            parts = []
            for k, v in best.detailed_scores.items():
                parts.append(f"{k}={v}")
            score_info = " (" + ", ".join(parts) + ")"

        if best:
            if best.leftover_sorted:
                lines.append(f"{pref}{s.op}{extra}: {best.source}  leftover={best.leftover_sorted}  "
                             f"leftover_words≈{list(best.leftover_words)[:5]}{score_info}")
            else:
                lines.append(f"{pref}{s.op}{extra}: {best.source}{score_info}")
        else:
            # For combined steps like CONCAT, we might still have detailed_scores on the Candidate
            # but usually it's better to show it on the top line
            cand = s.candidates[0] if s.candidates else None
            score_info = ""
            if cand and cand.detailed_scores:
                parts = []
                for k, v in cand.detailed_scores.items():
                    parts.append(f"{k}={v}")
                score_info = " (" + ", ".join(parts) + ")"
            lines.append(f"{pref}{s.op}{extra}: (combined){score_info}")

        if s.child:
            rec(s.child, level + 1)
            
        for chunk in s.chunk_steps:
            rec(chunk, level + 1)

    rec(step, 0)
    lines.append(f"{indent}TOTAL_SCORE={step.best_score():.3f}")
    return "\n".join(lines)


class CluePresenter:
    @staticmethod
    def print_variety(steps, top_n_total=25, top_n_per_type=3, finder=None):
        by_op = defaultdict(list)

        shown_count = 0
        op_counts = defaultdict(int)

        for s in steps:
            if shown_count >= top_n_total:
                break

            if op_counts[s.op] < top_n_per_type:
                from src.utils import pretty
                print(pretty(s))
                
                # Show suggestions if finder (and LLM) is available
                if finder and finder.indicator_suggestor:
                    best_c = s.best()
                    if best_c:
                        inds = finder.indicator_suggestor.suggest_for_candidate(s, best_c)
                        if inds:
                            print(f"  Suggested indicators: {', '.join(inds)}")
                        
                        syns = finder.indicator_suggestor.suggest_synonyms(best_c)
                        if syns:
                            syn_parts = [f"{k} -> {', '.join(v)}" for k, v in syns.items()]
                            print(f"  Synonym options: {'; '.join(syn_parts)}")
                
                print("-" * 20)
                op_counts[s.op] += 1
                shown_count += 1
