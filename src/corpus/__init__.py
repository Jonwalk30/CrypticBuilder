from .base import Corpus
import os

_corpus = None

def __getattr__(name):
    if name == "corpus":
        global _corpus
        if _corpus is None:
            parquet_path = os.getenv("CORPUS_PARQUET_PATH", "corpus.parquet")
            if not os.path.exists(parquet_path):
                from .preprocess_corpus import preprocess_wordlist_csv
                csv_path = "en.csv"
                print(f"Corpus file '{parquet_path}' not found. Generating from '{csv_path}'...")
                preprocess_wordlist_csv(
                    csv_path=csv_path,
                    out_parquet_path=parquet_path,
                    lang="en",
                    min_len=1,
                    min_freq=0,
                    augment_common_top_n=100000
                )
            _corpus = Corpus.from_parquet(parquet_path)
        return _corpus
    raise AttributeError(f"module {__name__} has no attribute {name}")
