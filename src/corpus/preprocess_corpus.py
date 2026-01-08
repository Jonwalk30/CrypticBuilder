# cryptic/preprocess_corpus.py
import json
import pandas as pd

from src.utils import normalize, sort_word, global_frequency, is_stopword_entry, stopword_ratio


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
    Preprocess a one-column wordlist CSV into a fast parquet + optional anagram index.
    """
    # Try to load robustly:
    try:
        df = pd.read_csv(csv_path, header=None, names=["entry"])
    except Exception:
        df = pd.read_csv(csv_path).rename(columns={"A": "entry"})

    if "entry" not in df.columns:
        df = pd.read_csv(csv_path).rename(columns={"A": "entry"})

    df = df.dropna()
    df["entry"] = df["entry"].astype(str).apply(normalize)
    # Remove empty entries after normalization
    df = df[df["entry"].str.len() > 0]
    
    df["entry_sorted"] = df["entry"].apply(sort_word)
    df["frequency"] = df["entry"].apply(lambda x: float(global_frequency(x, lang)))
    df["is_stopword"] = df["entry"].apply(is_stopword_entry)
    df["stopword_ratio_entry"] = df["entry"].apply(stopword_ratio)

    if augment_common_top_n > 0:
        df = augment_with_common_words(df, lang=lang, top_n=augment_common_top_n)

    df = df[
        (df["entry"].str.len() >= min_len)
        & (df["frequency"] >= min_freq)
    ]
    if exclude_stopwords:
        df = df[~df["is_stopword"]]
    df = df.drop_duplicates(subset=["entry"]).reset_index(drop=True)

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