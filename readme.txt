# Boetticher (Cryptic Assistant)

A Python toolkit for building cryptic crossword clues.

Features:
- Corpus-based wordplay search (anagrams, hidden words, alternating, leftovers)
- Phrase-aware (not just single words)
- Ranking by **general language frequency** (via `wordfreq`)
- LLM-powered helpers for:
  - synonyms
  - definition suggestions
  - indicator phrases
  - double definitions
- Mobile-friendly Streamlit app

## Install

```bash
pip install pandas wordfreq openai streamlit
```

## Running the UI

To start the mobile-friendly web interface:

```bash
streamlit run app.py
```