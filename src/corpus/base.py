from typing import Optional
import pandas as pd
import os

class Corpus:
    @classmethod
    def from_parquet(
            cls,
            parquet_path: str,
            anagram_best_path: Optional[str] = None,
            anagram_words_path: Optional[str] = None,
            lang: str = "en",
    ) -> "Corpus":
        """
        Load preprocessed corpus parquet (recommended).

        - anagram_best_path: JSON mapping signature -> {best_word, best_freq, best_stop_ratio}
        - anagram_words_path: JSON mapping signature -> [top anagrams by frequency] (optional)
        """
        df = pd.read_parquet(parquet_path)
        obj = cls(df, lang=lang)
        return obj

    # -------------------------
    # Initialisation
    # -------------------------

    def __init__(self, df: pd.DataFrame, lang: str = "en", substitutions_path: str = "substitutions.yml"):
        if "entry" not in df.columns and "word" not in df.columns:
            raise ValueError("Corpus must contain 'entry' or 'word' column")

        self.lang = lang
        self.corpus = df.copy()

        if "entry" not in self.corpus.columns:
            self.corpus["entry"] = self.corpus["word"]

        # Basic cleanup - ensuring we have no nans and correctly typed frequency
        self.corpus = self.corpus.dropna(subset=["entry", "entry_sorted", "frequency"]).reset_index(drop=True)
        self.corpus["frequency"] = self.corpus["frequency"].astype(float)

        # Build cached anagram indices (using fast itertuples)
        self._build_anagram_indices()

        # Load substitutions
        self.substitutions = self._load_substitutions(substitutions_path)
        
        # 2+ letter substrings in the substitution list as words for insertion/deletion
        self.substitution_words = {k for k in self.substitutions.keys() if len(k) >= 2}

        # Ranking configuration (SUM ONLY)
        self.rank_config = {
            "direct": {
                "w_entry": 1.0,
                "w_leftover": 0.0,
                "w_cov": 0.5,
                "stopword_penalty": 2.0,  # Zipf points to subtract at max stopwordiness
                "stopword_power": 1.0,  # 1.0 linear, >1 harsher
            },
            "hiding": {
                "w_entry": 1.0,
                "w_leftover": 1.0,  # reward good leftover
                "w_cov": 0.1,  # reverse coverage matters a bit
                "stopword_penalty": 2.0,
                "stopword_power": 1.0,
            },
        }

    def _build_anagram_indices(self):
        """
        Builds:
          - self._anagram_words_sorted: signature -> list[str] (sorted by freq desc)
          - self._anagram_best: signature -> dict(best_word, best_freq, best_stop_ratio)

        This is done ONCE so later lookups are O(1) with no sorting.
        Optimized using itertuples for speed.
        """
        self._anagram_words_sorted = {}
        self._anagram_best = {}

        # The corpus MUST be pre-sorted by frequency descending for this to work correctly
        # (preprocess_corpus.py handles this)
        for row in self.corpus.itertuples():
            sig = row.entry_sorted
            
            # Words sorted by frequency (first one seen is the best)
            if sig not in self._anagram_words_sorted:
                self._anagram_words_sorted[sig] = []
                # First time we see this signature, it's the best word
                self._anagram_best[sig] = {
                    "best_word": row.entry,
                    "best_freq": float(row.frequency),
                    "best_stop_ratio": float(getattr(row, "stopword_ratio_entry", 0.0)),
                    "best_is_proper_noun": bool(getattr(row, "is_proper_noun", False))
                }
            
            self._anagram_words_sorted[sig].append(row.entry)

    def _load_substitutions(self, path: str) -> dict[str, list[str]]:
        import os
        subs = {}
        if not os.path.exists(path):
            return subs
        
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if ":" not in line:
                    continue
                key, val = line.split(":", 1)
                key = key.strip().lower()
                val = val.strip()
                if key not in subs:
                    subs[key] = []
                subs[key].append(val)
        return subs
