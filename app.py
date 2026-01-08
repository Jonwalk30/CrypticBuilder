import streamlit as st
import os
import json
import re
from src.finder.finder import ClueFinder
from src.corpus.corpus import corpus
from src.utils import pretty

# --- Page Config ---
st.set_page_config(
    page_title="Boetticher",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- App Styles ---
st.markdown("""
<style>
    .stExpander {
        border: 1px solid #e6e9ef;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .clue-tree {
        font-family: monospace;
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        white-space: pre-wrap;
    }
    .score-badge {
        background-color: #4CAF50;
        color: white;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar Configuration ---
with st.sidebar:
    st.title("🧩 Boetticher")
    
    target_word = st.text_input("Target Word", value="borders", help="The word you want to find cryptic clues for.").lower().strip()
    
    st.divider()
    st.subheader("🔍 Search Settings")
    
    num_chunks = st.select_slider(
        "Max Complexity (Chunks)",
        options=[1, 2, 3],
        value=2,
        help="How many pieces the word can be broken into (e.g. 1=single operation, 2=two parts combined)"
    )
    
    max_fodder = st.number_input("Max Anagram Words", min_value=1, max_value=4, value=3)
    
    use_positional = st.toggle("Enable First/Last/Middle Letters", value=True, help="Toggle search for first/last/middle letter combinations (can be slow).")

    available_ops = [
        "ANAGRAM", "WORD_DELETION", "LETTER_DELETION", "FIRST_LETTERS", "LAST_LETTERS",
        "MIDDLE_LETTERS", "ALTERNATING_LETTERS", "REVERSAL", "WORD_INSERTION", "HIDDEN",
        "SUBSTITUTION", "CHARADE", "POSITIONAL_DELETION", "LETTER_REPLACEMENT"
    ]
    
    selected_ops_to_run = st.multiselect(
        "Enabled Clue Types",
        options=available_ops,
        default=available_ops,
        help="Select which wordplay methods to search for."
    )

    st.divider()
    st.subheader("🤖 AI Settings")
    
    use_llm = st.toggle("Enable LLM AI", value=True, help="Use OpenAI to score surface coherence and suggest indicators.")
    
    # Specific AI toggle for positional letters
    if use_llm and use_positional:
        use_llm_for_positional = st.checkbox("AI-Score First/Last/Middle Letters", value=False, help="Using AI to score these specific types of clues is significantly slower and more expensive.")
    else:
        use_llm_for_positional = False
    
    st.divider()

    generate_clicked = st.button("🚀 Generate Clues", type="primary", use_container_width=True)

    st.divider()
    
    if st.button("🗑️ Clear AI Cache"):
        if os.path.exists(".llm_cache.json"):
            os.remove(".llm_cache.json")
        st.cache_resource.clear()
        st.success("Cache cleared!")
        st.rerun()

# --- Finder Initialization ---
@st.cache_resource
def get_finder(llm_on, positional_on, m_fodder, llm_pos_on):
    return ClueFinder(
        use_llm=llm_on, 
        use_positional_selection=positional_on, 
        max_fodder_words=m_fodder,
        use_llm_for_positional=llm_pos_on
    )

finder = get_finder(use_llm, use_positional, max_fodder, use_llm_for_positional)

# --- Main Logic ---
if "all_steps" not in st.session_state:
    st.session_state.all_steps = []
if "last_target" not in st.session_state:
    st.session_state.last_target = ""

if not target_word:
    st.info("Enter a word in the sidebar to start building clues.")
    st.stop()

# Perform Search
if generate_clicked:
    with st.spinner(f"🕵️ Searching for clue patterns for '{target_word.upper()}'..."):
        # Note: corpus is imported from src.corpus.corpus and should be persistent
        st.session_state.all_steps = finder.find_all(
            corpus, target_word, num_chunks=num_chunks, enabled_ops=selected_ops_to_run
        )
        st.session_state.last_target = target_word

all_steps = st.session_state.all_steps

if not all_steps:
    if generate_clicked:
        st.error(f"No valid clue patterns found for '{target_word}'. Try reducing complexity or check your word spelling.")
    else:
        st.info("Click 'Generate Clues' in the sidebar to begin.")
    st.stop()

# Header & Summary
st.title(f"🧩 Boetticher: Clue Ideas for `{target_word.upper()}`")

col_a, col_b = st.columns([1, 1])
with col_a:
    st.metric("Total Patterns Found", len(all_steps))
with col_b:
    if finder.llm_scorer:
        st.metric("AI API Calls", finder.llm_scorer.api_call_count)

# --- Filters ---
all_ops = sorted(list(set(s.op for s in all_steps)))
selected_ops = st.multiselect("Filter by Clue Types", options=all_ops, default=all_ops)
filtered_steps = [s for s in all_steps if s.op in selected_ops]

# --- Shortlist ---
if filtered_steps:
    st.subheader("🌟 Top Recommendations")
    shortlist_cols = st.columns(min(3, len(filtered_steps[:3])))
    for i, s in enumerate(filtered_steps[:3]):
        best_c = s.best()
        with shortlist_cols[i]:
            st.info(f"**{s.op}**\n\n`{best_c.source}`\n\nScore: {best_c.score:.2f}")

    st.divider()

# --- All Results ---
st.subheader("📚 All Clue Patterns")

for s in filtered_steps:
    best_c = s.best()
    label = f"[{s.op}] {best_c.source} (Score: {best_c.score:.2f})"
    
    with st.expander(label):
        tab1, tab2, tab3 = st.tabs(["🏗️ Logical Tree", "📖 Definitions & Indicators", "🔄 Substitutions"])
        
        with tab1:
            st.markdown("#### Clue Construction Tree")
            st.code(pretty(s), language="text")
            
        with tab2:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### Target Definitions")
                if finder.llm_scorer:
                    defs = finder.llm_scorer.get_definitions(target_word)
                    if defs:
                        st.write(", ".join([f"`{d}`" for d in defs]))
                    else:
                        st.write("No definitions found.")
                else:
                    st.write("AI disabled.")
            
            with c2:
                st.markdown("#### Suggested Indicators")
                if finder.indicator_suggestor:
                    inds = finder.indicator_suggestor.suggest_for_candidate(s, best_c)
                    if inds:
                        st.write(", ".join([f"`{i}`" for i in inds]))
                else:
                    st.write("AI disabled.")
                    
        with tab3:
            st.markdown("#### Fodder Substitutions")
            if finder.indicator_suggestor:
                syns = finder.indicator_suggestor.suggest_synonyms(best_c, corpus=corpus)
                if syns:
                    if len(syns) > 1:
                        sub_tabs = st.tabs(list(syns.keys()))
                        for i, (word, options) in enumerate(syns.items()):
                            with sub_tabs[i]:
                                st.write(f"Options for **{word}**:")
                                st.write(", ".join([f"`{o}`" for o in options]))
                    else:
                        for word, options in syns.items():
                            st.write(f"**{word}** ➡️ {', '.join([f'`{o}`' for o in options])}")
                else:
                    st.write("No substitutions found.")
            else:
                st.write("AI disabled.")

# --- Footer ---
st.caption("Cryptic Builder v0.1.0 • Built with Streamlit & GPT-4o-mini")
