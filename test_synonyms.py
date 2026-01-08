from src.finder.finder import ClueFinder
from src.corpus import corpus
from src.step.step import Candidate, Step
import streamlit as st

# Mock a candidate that looks like a word deletion
# ab - [a]
candidate = Candidate(
    source="ab - [a]",
    produced="b",
    leftover_sorted="a",
    leftover_words=("a",)
)
step = Step(op="WORD_DELETION", target="b", candidates=[candidate])

finder = ClueFinder(use_llm_ranking=False, use_llm_suggestions=True)

print("Testing suggest_synonyms for WORD_DELETION 'ab - [a]'")
syns = finder.indicator_suggestor.suggest_synonyms(candidate, corpus=corpus)
print(f"Synonyms found: {list(syns.keys())}")

# Mock a candidate that looks like a word insertion
candidate_ins = Candidate(
    source="(a) in ab",
    produced="aab",
    leftover_words=("a",)
)
print("\nTesting suggest_synonyms for WORD_INSERTION '(a) in ab'")
syns_ins = finder.indicator_suggestor.suggest_synonyms(candidate_ins, corpus=corpus)
print(f"Synonyms found: {list(syns_ins.keys())}")
