from __future__ import annotations
from src.finder.finder import ClueFinder
from src.corpus import corpus
from src.utils import CluePresenter

USE_LLM_RANKING = False
USE_LLM_SUGGESTIONS = True

finder = ClueFinder(max_fodder_words=3, use_llm_ranking=USE_LLM_RANKING, use_llm_suggestions=USE_LLM_SUGGESTIONS)
target_word = ("borders")

for n in [1, 2, 3]:
    print(f"\n=== Variety of options for {n} chunk(s) ===")
    steps = finder.find_all(corpus, target_word, num_chunks=n)
    CluePresenter.print_variety(steps, top_n_total=50, top_n_per_type=10, finder=finder)

if finder.llm_scorer:
    print("\n" + "="*30)
    finder.llm_scorer.print_cost()
    print("="*30)
