# cryptic/preprocess_corpus.py
import json
import pandas as pd
import os

from src.utils import normalize, sort_word, global_frequency, is_stopword_entry, stopword_ratio


def _load_proper_nouns() -> set[str]:
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
    for pn in proper_nouns:
        norm = normalize(pn)
        if norm:
            normalized_proper_nouns.add(norm)
    return normalized_proper_nouns


def augment_with_common_words(df, lang="en", top_n=20000):
    """
    Find common words from wordfreq that are missing in the current df and add them.
    """
    import wordfreq
    existing_entries = set(df["entry"])
    # top_n_list generally returns words in descending frequency order
    common_words = wordfreq.top_n_list(lang, top_n)
    
    new_rows = []
    for w in common_words:
        w_norm = normalize(w)
        if w_norm and w_norm not in existing_entries:
            # Only add if it's reasonably long or a known good short word (a, i)
            if len(w_norm) >= 2 or w_norm in ["a", "i"]:
                new_rows.append({
                    "entry": w_norm,
                    "entry_sorted": sort_word(w_norm),
                    "frequency": global_frequency(w_norm, lang),
                    "is_stopword": is_stopword_entry(w_norm),
                    "stopword_ratio_entry": stopword_ratio(w_norm)
                })
                existing_entries.add(w_norm)
            
    if new_rows:
        augment_df = pd.DataFrame(new_rows)
        return pd.concat([df, augment_df], ignore_index=True)
    return df


def preprocess_wordlist_csv(
    csv_path: str,
    out_parquet_path: str,
    lang: str = "en",
    min_len: int = 1,
    min_freq: float = 0,
    exclude_stopwords: bool = False,
    augment_common_top_n: int = 20000
):
    """
    Preprocess a one-column wordlist CSV into a fast parquet.
    """
    # Try to load robustly:
    if csv_path and os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path, header=None, names=["entry"])
        except Exception:
            df = pd.read_csv(csv_path).rename(columns={"A": "entry"})
    else:
        df = pd.DataFrame(columns=["entry"])

    df = df.dropna()
    df["entry"] = df["entry"].astype(str).apply(normalize)
    # Remove empty entries after normalization
    df = df[df["entry"].str.len() > 0]
    
    if augment_common_top_n > 0:
        df = augment_with_common_words(df, lang=lang, top_n=augment_common_top_n)

    # Load proper nouns
    proper_nouns = _load_proper_nouns()
    
    # Add missing proper nouns to corpus
    existing_entries = set(df["entry"])
    new_rows = []
    for n in proper_nouns:
        if n not in existing_entries:
            if len(n) >= 2:
                new_rows.append({"entry": n})
                existing_entries.add(n)
    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

    # Compute columns
    df["entry_sorted"] = df["entry"].apply(sort_word)
    df["frequency"] = df["entry"].apply(lambda x: float(global_frequency(x, lang)))
    df["is_stopword"] = df["entry"].apply(is_stopword_entry)
    df["stopword_ratio_entry"] = df["entry"].apply(stopword_ratio)
    df["is_proper_noun"] = df["entry"].isin(proper_nouns)

    # Filter
    df = df[
        (df["entry"].str.len() >= min_len)
        & (df["frequency"] >= min_freq)
    ]
    if exclude_stopwords:
        df = df[~df["is_stopword"]]
        
    # Remove obscure words (trash cleanup)
    # This logic is moved from Corpus.__init__
    mask_1 = (df["entry"].str.len() == 1) & (~df["entry"].isin(["a", "i"]))
    freq = df["frequency"]
    length = df["entry"].str.len()
    is_name = df["is_proper_noun"]
    
    mask_obscure = (
        ((length == 2) & (freq < 3.5)) |
        ((length == 3) & (freq < 3.0)) |
        ((length == 4) & (freq < 3.0)) |
        ((length == 5) & (freq < 2.5)) |
        ((length >= 6) & (freq < 2.0))
    ) & (~is_name)
    
    df = df[~mask_1 & ~mask_obscure]
    
    df = df.drop_duplicates(subset=["entry"]).reset_index(drop=True)
    
    # Sort by frequency desc (CRITICAL for fast g.first() later)
    df = df.sort_values("frequency", ascending=False).reset_index(drop=True)

    # Save parquet
    df.to_parquet(out_parquet_path, index=False)
    return df


if __name__ == "__main__":
    # Example usage:
    preprocess_wordlist_csv(
        csv_path="en.csv",
        out_parquet_path="corpus.parquet",
        lang="en",
        min_len=1,
        min_freq=0,
        augment_common_top_n=100_000
    )
    print("Preprocessing complete.")