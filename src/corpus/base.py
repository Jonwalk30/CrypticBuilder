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

        # Load proper nouns list
        self.proper_nouns = self._load_proper_nouns()

        # Add missing proper nouns to corpus
        existing_entries = set(self.corpus["entry"])
        new_rows = []
        from src.utils import global_frequency, sort_word, stopword_ratio, is_stopword_entry
        for n in self.proper_nouns:
            if n not in existing_entries:
                # Basic length filter for names (at least 2 chars)
                if len(n) >= 2:
                    new_rows.append({
                        "entry": n,
                        "entry_sorted": sort_word(n),
                        "frequency": global_frequency(n, self.lang),
                        "is_stopword": is_stopword_entry(n),
                        "stopword_ratio_entry": stopword_ratio(n)
                    })
                    existing_entries.add(n)
        if new_rows:
            self.corpus = pd.concat([self.corpus, pd.DataFrame(new_rows)], ignore_index=True)

        # Mark proper nouns
        self.corpus["is_proper_noun"] = self.corpus["entry"].isin(self.proper_nouns)

        # Basic cleanup
        self.corpus = self.corpus.dropna().drop_duplicates().reset_index(drop=True)

        # Remove obscure short words (trash cleanup)
        # 1-letter words must be 'a' or 'i'
        mask_1 = (self.corpus["entry"].str.len() == 1) & (~self.corpus["entry"].isin(["a", "i"]))
        
        # Length-dependent Zipf thresholds to filter out obscure strings/names
        # Short words need higher frequency to be considered "known" words.
        # This helps eliminate obscure abbreviations or names like 'petr'.
        freq = self.corpus["frequency"]
        length = self.corpus["entry"].str.len()
        is_name = self.corpus["is_proper_noun"]

        mask_obscure = (
            ((length == 2) & (freq < 3.5)) |
            ((length == 3) & (freq < 3.0)) |
            ((length == 4) & (freq < 3.0)) |
            ((length == 5) & (freq < 2.5)) |
            ((length >= 6) & (freq < 2.0))
        ) & (~is_name) # PROTECT NAMES
        
        # Explicitly remove common non-word characters that might have high frequency but aren't useful
        self.corpus = self.corpus[~mask_1 & ~mask_obscure].reset_index(drop=True)

        # Pre-sort by frequency descending so that .head() always gives most common words
        self.corpus = self.corpus.sort_values("frequency", ascending=False).reset_index(drop=True)

        # Build cached anagram indices (sorted once)
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

    def _load_proper_nouns(self) -> set[str]:
        import os
        proper_nouns = set()
        if os.path.exists("/usr/share/dict/propernames"):
            with open("/usr/share/dict/propernames", "r") as f:
                for line in f:
                    n = line.strip().lower()
                    if n:
                        proper_nouns.add(n)

        # Supplemental common surnames
        surnames = {
            "smith", "johnson", "williams", "brown", "jones", "garcia", "miller", "davis",
            "rodriguez", "martinez", "hernandez", "lopez", "gonzalez", "wilson", "anderson",
            "thomas", "taylor", "moore", "jackson", "martin", "lee", "perez", "thompson",
            "white", "harris", "sanchez", "clark", "ramirez", "lewis", "robinson", "walker",
            "young", "allen", "king", "wright", "scott", "torres", "nguyen", "hill", "flores",
            "green", "adams", "nelson", "baker", "hall", "rivera", "campbell", "mitchell",
            "carter", "roberts"
        }
        proper_nouns.update(surnames)

        # Countries
        countries = {
            "afghanistan", "albania", "algeria", "andorra", "angola", "argentina", "armenia", "australia", "austria",
            "azerbaijan", "bahamas", "bahrain", "bangladesh", "barbados", "belarus", "belgium", "belize", "benin",
            "bhutan", "bolivia", "bosnia", "botswana", "brazil", "brunei", "bulgaria", "burundi", "cambodia", "cameroon",
            "canada", "chad", "chile", "china", "colombia", "comoros", "congo", "croatia", "cuba", "cyprus", "czechia",
            "denmark", "djibouti", "dominica", "ecuador", "egypt", "eritrea", "estonia", "ethiopia", "fiji", "finland",
            "france", "gabon", "gambia", "georgia", "germany", "ghana", "greece", "grenada", "guatemala", "guinea",
            "guyana", "haiti", "honduras", "hungary", "iceland", "india", "indonesia", "iran", "iraq", "ireland",
            "israel", "italy", "jamaica", "japan", "jordan", "kazakhstan", "kenya", "kiribati", "kuwait", "kyrgyzstan",
            "laos", "latvia", "lebanon", "lesotho", "liberia", "libya", "lithuania", "luxembourg", "madagascar", "malawi",
            "malaysia", "maldives", "mali", "malta", "mauritania", "mauritius", "mexico", "micronesia", "moldova",
            "monaco", "mongolia", "montenegro", "morocco", "mozambique", "myanmar", "namibia", "nauru", "nepal",
            "netherlands", "new zealand", "nicaragua", "niger", "nigeria", "norway", "oman", "pakistan", "palau",
            "panama", "paraguay", "peru", "philippines", "poland", "portugal", "qatar", "romania", "russia", "rwanda",
            "samoa", "senegal", "serbia", "seychelles", "singapore", "slovakia", "slovenia", "somalia", "spain", "sudan",
            "suriname", "sweden", "switzerland", "syria", "taiwan", "tajikistan", "tanzania", "thailand", "togo", "tonga",
            "tunisia", "turkey", "turkmenistan", "tuvalu", "uganda", "ukraine", "uae", "uruguay", "uzbekistan", "vanuatu",
            "vatican", "venezuela", "vietnam", "yemen", "zambia", "zimbabwe"
        }
        proper_nouns.update(countries)

        # Cities
        cities = {
            "tokyo", "delhi", "shanghai", "sao paulo", "mexico city", "cairo", "mumbai", "beijing", "dhaka", "osaka",
            "karachi", "chongqing", "istanbul", "buenos aires", "kolkata", "lagos", "kinshasa", "manila", "rio de janeiro",
            "guangzhou", "lahore", "shenzhen", "bangalore", "moscow", "tianjin", "jakarta", "london", "lima", "bangkok",
            "seoul", "hyderabad", "chennai", "chicago", "taipei", "bogota", "wuhan", "hong kong", "hangzhou", "foshan",
            "luanda", "baghdad", "amman", "kuala lumpur", "surat", "suzhou", "houston", "madrid", "pune", "ahmedabad",
            "toronto", "washington", "belo horizonte", "philadelphia", "atlanta", "fukuoka", "dallas", "riyadh",
            "singapore", "barcelona", "saint petersburg", "santiago", "amsterdam", "paris", "berlin", "rome", "sydney",
            "melbourne", "vienna", "prague", "warsaw", "brussels", "budapest", "munich", "milan", "zurich", "geneva",
            "oslo", "stockholm", "copenhagen", "helsinki", "lisbon", "dublin", "venice", "florence", "naples", "athens"
        }
        proper_nouns.update(cities)

        # Companies
        companies = {
            "apple", "google", "microsoft", "amazon", "meta", "facebook", "tesla", "ford", "toyota", "honda", "samsung",
            "sony", "nike", "adidas", "coca-cola", "mcdonalds", "starbucks", "netflix", "disney", "visa", "mastercard",
            "intel", "nvidia", "amd", "dell", "hp", "ibm", "oracle", "sap", "cisco", "uber", "airbnb", "boeing",
            "airbus", "ferrari", "bmw", "mercedes", "audi", "volkswagen", "renault", "peugeot", "fiat", "shell", "bp",
            "exxon", "chevron", "total", "nestle", "unilever", "pepsico", "walmart", "costco", "target", "alibaba",
            "tencent", "baidu", "tiktok", "spotify", "paypal", "goldman sachs"
        }
        proper_nouns.update(companies)

        # Events and Misc
        misc = {
            "christmas", "easter", "ramadan", "diwali", "hanukkah", "halloween", "thanksgiving", "olympics", "world cup",
            "everest", "kilimanjaro", "alps", "andes", "rockies", "himalayas", "sahara", "amazon", "nile", "mississippi",
            "danube", "thames", "seine", "hudson", "ganges", "yangtze", "pacific", "atlantic", "indian", "arctic",
            "antarctic", "mediterranean", "caspian", "caribbean", "baltic", "bering", "adriatic", "africa",
            "antarctica", "asia", "europe", "oceania", "america", "mercury", "venus", "earth", "mars", "jupiter",
            "saturn", "uranus", "neptune", "pluto", "january", "february", "march", "april", "may", "june", "july",
            "august", "september", "october", "november", "december", "monday", "tuesday", "wednesday", "thursday",
            "friday", "saturday", "sunday"
        }
        proper_nouns.update(misc)

        # Normalize and handle multi-word entries
        normalized_proper_nouns = set()
        from src.utils import normalize
        for pn in proper_nouns:
            norm = normalize(pn)
            if norm:
                normalized_proper_nouns.add(norm)

        return normalized_proper_nouns

    # -------------------------
    # Precomputed indices (fast)
    # -------------------------

    def _build_anagram_indices(self):
        """
        Builds:
          - self._anagram_words_sorted: signature -> list[str] (sorted by freq desc)
          - self._anagram_best: signature -> dict(best_word, best_freq, best_stop_ratio)

        This is done ONCE so later lookups are O(1) with no sorting.
        """
        grouped = (
            self.corpus.groupby("entry_sorted")[["entry", "frequency", "stopword_ratio_entry", "is_proper_noun"]]
            .apply(lambda g: g.to_dict("records"))
            .to_dict()
        )

        self._anagram_words_sorted = {}
        self._anagram_best = {}

        for sig, records in grouped.items():
            records_sorted = sorted(records, key=lambda r: r["frequency"], reverse=True)

            words = [r["entry"] for r in records_sorted]
            best = records_sorted[0]

            self._anagram_words_sorted[sig] = words
            self._anagram_best[sig] = {
                "best_word": best["entry"],
                "best_freq": float(best["frequency"]),
                "best_stop_ratio": float(best.get("stopword_ratio_entry", 0.0)),
                "best_is_proper_noun": bool(best.get("is_proper_noun", False))
            }

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
# Global instance for easy access
parquet_path = os.getenv("CORPUS_PARQUET_PATH", "corpus.parquet")
corpus = Corpus.from_parquet(parquet_path)
