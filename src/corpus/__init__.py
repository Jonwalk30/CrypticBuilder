from .base import Corpus
import os

_corpus = None

def _get_corpus():
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
    return Corpus.from_parquet(parquet_path)

def __getattr__(name):
    if name == "corpus":
        global _corpus
        if _corpus is None:
            # Try to use streamlit caching if we are in a streamlit app
            try:
                import streamlit as st
                # Use a wrapper to avoid streamlit hashing issues with the module itself
                @st.cache_resource(show_spinner="Loading corpus...")
                def get_cached_corpus():
                    return _get_corpus()
                _corpus = get_cached_corpus()
            except ImportError:
                _corpus = _get_corpus()
        return _corpus
    raise AttributeError(f"module {__name__} has no attribute {name}")
