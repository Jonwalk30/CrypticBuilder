import streamlit as st
import html
import os
import json
import re
from src.finder.finder import ClueFinder
from src.utils import pretty

# --- Page Config ---
st.set_page_config(
    page_title="Boetticher",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Icons ---
OP_ICONS = {
    "ANAGRAM": "🔀",
    "WORD_DELETION": "✂️",
    "LETTER_DELETION": "📉",
    "FIRST_LETTERS": "🔝",
    "LAST_LETTERS": "🔚",
    "MIDDLE_LETTERS": "🎯",
    "ALTERNATING_LETTERS": "🦓",
    "REVERSAL": "◀️",
    "WORD_INSERTION": "➕",
    "HIDDEN": "🫣",
    "SUBSTITUTION": "🔄",
    "CHARADE": "🧱",
    "POSITIONAL_DELETION": "🔻",
    "LETTER_REPLACEMENT": "✏️",
    "LITERAL": "🔡",
    "NESTED": "🪆",
    "DOUBLE_DEFINITION": "👥"
}

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
        float: right;
        background-color: #f8f9fb;
        color: #555;
        padding: 2px 10px;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: 500;
        border: 1px solid #ddd;
        margin-top: -5px;
    }
    .clue-label {
        font-weight: 600;
    }
    .clue-source {
        color: #1f77b4;
        font-family: monospace;
    }
    .op-tag {
        font-weight: bold;
        color: #1f77b4;
    }
    .starred-btn {
        color: #FFD700;
    }
    .copy-btn {
        background-color: #f0f2f6;
        border: 1px solid #d1d5db;
        border-radius: 4px;
        padding: 4px 8px;
        font-size: 0.8rem;
        cursor: pointer;
    }
    /* Green primary buttons for starred and generate */
    button[kind="primary"], button[data-testid="baseButton-primary"] {
        background-color: #28a745 !important;
        border-color: #28a745 !important;
        color: white !important;
    }
    button[kind="primary"]:hover, button[data-testid="baseButton-primary"]:hover {
        background-color: #218838 !important;
        border-color: #1e7e34 !important;
    }
    /* Pagination styling */
    .pagination-container {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 15px;
        margin: 20px 0;
        padding: 10px;
        background-color: #f0f2f6;
        border-radius: 8px;
    }
    /* Tree visualization */
    .tree-container {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #e6e9ef;
    }
    .tree-branch {
        margin-left: 20px;
        border-left: 1px dashed #ccc;
        padding-left: 15px;
        margin-top: 5px;
    }
    .tree-leaf {
        margin-bottom: 8px;
    }
    .tree-target {
        font-weight: bold;
        color: #0e1117;
        font-size: 1.05rem;
    }
    .tree-op-badge {
        background-color: #e1f5fe;
        color: #01579b;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        margin-right: 8px;
    }
    .tree-source-text {
        font-family: 'Source Code Pro', monospace;
        color: #d32f2f;
        font-weight: 600;
    }
    .tree-meta {
        color: #666;
        font-size: 0.8rem;
        margin-top: 2px;
    }
</style>
""", unsafe_allow_html=True)

# --- Session State Initialization ---
if "all_steps" not in st.session_state:
    st.session_state.all_steps = []
if "last_target" not in st.session_state:
    st.session_state.last_target = ""
if "starred_clues" not in st.session_state:
    st.session_state.starred_clues = []
if "search_history" not in st.session_state:
    st.session_state.search_history = []
if "page_num" not in st.session_state:
    st.session_state.page_num = 1
if "fav_page_num" not in st.session_state:
    st.session_state.fav_page_num = 1
if "hist_page_num" not in st.session_state:
    st.session_state.hist_page_num = 1

AVAILABLE_OPS = [
    "ANAGRAM", "WORD_DELETION", "LETTER_DELETION", "FIRST_LETTERS", "LAST_LETTERS",
    "MIDDLE_LETTERS", "ALTERNATING_LETTERS", "REVERSAL", "WORD_INSERTION", "HIDDEN",
    "SUBSTITUTION", "CHARADE", "POSITIONAL_DELETION", "LETTER_REPLACEMENT", "DOUBLE_DEFINITION"
]

if "selected_ops" not in st.session_state:
    st.session_state.selected_ops = AVAILABLE_OPS

# --- Sidebar Configuration ---
with st.sidebar:
    st.title("🧩 Boetticher")
    
    target_word = st.text_input("Target Word", value="cryptic", help="The word you want to find cryptic clues for.").lower().strip()
    
    st.divider()
    st.subheader("🔍 Search Settings")
    
    num_chunks = st.select_slider(
        "Max Complexity (Chunks)",
        options=[1, 2, 3],
        value=2,
        help="Max pieces the word can be broken into (e.g. 1=single operation, 2=two parts combined)"
    )
    
    max_fodder = st.number_input("Max Anagram Words", min_value=1, max_value=4, value=3)
    
    use_positional = st.toggle("Include First/Last/Middle Letters", value=False, help="Toggle search for first/last/middle letter combinations (can be slow).")

    # Handle interaction between use_positional and selected_ops
    pos_ops = ["FIRST_LETTERS", "LAST_LETTERS", "MIDDLE_LETTERS"]
    
    with st.popover("⚙️ Enabled Clue Types", use_container_width=True):
        select_all = st.checkbox("Select All", value=True, key="select_all_ops")
        
        # If select_all is clicked, update selected_ops in session state
        # but don't disable individual checkboxes
        if select_all:
            if use_positional:
                st.session_state.selected_ops = AVAILABLE_OPS
            else:
                st.session_state.selected_ops = [op for op in AVAILABLE_OPS if op not in pos_ops]
        
        selected_ops_to_run = []
        for op in AVAILABLE_OPS:
            is_pos = op in pos_ops
            # Positional ops are only available if the toggle is ON
            if is_pos and not use_positional:
                checked = False
                disabled = True
            else:
                # If select_all is ON, it's checked. 
                # If select_all is OFF, it follows session state.
                checked = op in st.session_state.selected_ops or select_all
                disabled = False
            
            icon = OP_ICONS.get(op, "")
            if st.checkbox(f"{icon} {op}", value=checked, disabled=disabled, key=f"sidebar_op_{op}"):
                selected_ops_to_run.append(op)
        
        if not select_all:
            st.session_state.selected_ops = selected_ops_to_run
        else:
            # When select_all is on, we take the filtered list
            st.session_state.selected_ops = selected_ops_to_run

    st.divider()
    st.subheader("🤖 AI Settings")
    
    use_llm_ranking = st.toggle("AI for Ranking (Context)", value=False, help="Use OpenAI to score surface coherence and context. Higher quality, but slower.")
    use_llm_suggestions = st.toggle("AI for Suggestions", value=True, help="Use OpenAI to suggest indicators and substitutions.")
    
    # Specific AI toggle for positional letters
    if use_llm_ranking and use_positional:
        use_llm_for_positional = st.checkbox("AI-Score First/Last/Middle Letters", value=False, help="Using AI to score these specific types of clues is significantly slower and more expensive.")
    else:
        use_llm_for_positional = False
    
    st.divider()

    generate_clicked = st.button("🚀 Generate Clues", type="primary", use_container_width=True)

    if "is_searching" not in st.session_state:
        st.session_state.is_searching = False

    if generate_clicked:
        st.session_state.is_searching = True

    if st.session_state.is_searching:
        if st.button("⏹️ Cancel Search"):
            st.session_state.is_searching = False
            st.rerun()

    st.divider()
    
    if st.button("🗑️ Clear AI Cache"):
        if os.path.exists(".llm_cache.json"):
            os.remove(".llm_cache.json")
        st.cache_resource.clear()
        st.success("Cache cleared!")
        st.rerun()

def get_corpus():
    from src.corpus import corpus
    return corpus

# --- Finder Initialization ---
@st.cache_resource
def get_finder(llm_ranking_on, llm_suggestions_on, positional_on, m_fodder, llm_pos_on):
    return ClueFinder(
        use_llm_ranking=llm_ranking_on,
        use_llm_suggestions=llm_suggestions_on,
        use_positional_selection=positional_on, 
        max_fodder_words=m_fodder,
        use_llm_for_positional=llm_pos_on
    )

@st.cache_data(show_spinner="Finding best clue patterns...")
def get_search_results(target_word, num_chunks, max_fodder, use_positional, enabled_ops, use_llm_ranking, use_llm_suggestions, use_llm_for_positional):
    # This wrapper function is cached by Streamlit
    from src.corpus import corpus as corpus_obj
    f = get_finder(use_llm_ranking, use_llm_suggestions, use_positional, max_fodder, use_llm_for_positional)
    return f.find_all(
        corpus_obj, 
        target_word, 
        num_chunks=num_chunks, 
        enabled_ops=enabled_ops
    )

@st.cache_data(show_spinner=False)
def get_definitions(display_word, _llm_scorer):
    if not _llm_scorer:
        return []
    return _llm_scorer.get_definitions(display_word)

@st.cache_data(show_spinner=False)
def get_indicator_suggestions(op, source, _indicator_suggestor):
    if not _indicator_suggestor:
        return []
    
    from src.utils import clean_source_fodder
    source_clean = clean_source_fodder(source)
    words = source_clean.split()
    
    return _indicator_suggestor.suggest_indicators(op, words)

@st.cache_data(show_spinner=False)
def get_fodder_synonyms(source, _indicator_suggestor):
    if not _indicator_suggestor:
        return {}
    
    from src.corpus import corpus as corpus_obj
    
    # suggest_synonyms only needs an object with a .source attribute
    class MockCandidate:
        def __init__(self, s): self.source = s
    
    return _indicator_suggestor.suggest_synonyms(MockCandidate(source), corpus=corpus_obj)

def get_finder_instance():
    return get_finder(use_llm_ranking, use_llm_suggestions, use_positional, max_fodder, use_llm_for_positional)

def suggest_full_clue(finder, step, candidate):
    """Uses AI to recommend a full cryptic clue."""
    if not finder.llm_scorer or not finder.llm_scorer.client:
        return "AI not configured for clue generation."
    
    target = step.target
    op = step.op
    source = candidate.source
    
    # Get length indicator accurately for phrases
    clean_target = target.replace("-", " ").replace(",", " ")
    parts = clean_target.split()
    if len(parts) > 1:
        len_str = f"({','.join(str(len(p)) for p in parts)})"
    else:
        len_str = f"({len(target)})"

    # Get definitions and indicators using cached helpers
    defs = get_definitions(target, finder.llm_scorer)
    indicators = get_indicator_suggestions(op, source, finder.indicator_suggestor)
    
    prompt = f"""
    You are a cryptic crossword expert. Write a single, elegant cryptic crossword clue for the word or phrase "{target.upper()}".
    
    The wordplay pattern found is: {op} using "{source}".
    
    Some potential definitions for "{target.upper()}": {', '.join(defs[:5])}
    Some potential indicators for {op}: {', '.join(indicators[:5])}
    
    Requirements:
    1. Use one of the definitions (or a very close synonym) at either the beginning or the end of the clue.
    2. Use the wordplay parts: {source}.
    3. Use a suitable cryptic indicator if needed for {op}.
    4. The surface reading should be natural and coherent.
    5. Respond ONLY with the clue text, followed by the length {len_str} in parentheses.
    
    Example output format:
    Definition indicator wordplay {len_str}
    """

    try:
        response = finder.llm_scorer.client.chat.completions.create(
            model=finder.llm_scorer.model,
            messages=[
                {"role": "system", "content": "You are a cryptic crossword master. You write concise, elegant clues."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            temperature=0.7
        )
        clue = response.choices[0].message.content.strip()
        # Ensure it ends with the correct length if AI forgot or got it wrong
        if not clue.endswith(len_str):
            # Strip any existing (X) and add the correct one
            clue = re.sub(r"\s*\(\d+(?:,\d+)*\)$", "", clue)
            clue = f"{clue} {len_str}"
        return clue
    except Exception as e:
        return f"Error generating clue: {str(e)}"

def render_step_html(s):
    best = s.best()
    if not best:
        return ""
    
    op_label = html.escape(s.op)
    source = html.escape(best.source)
    target = html.escape(s.target.upper())
    
    score_info = ""
    if best.detailed_scores:
        # Filter out very long score info for the tree (e.g. lists of frequencies)
        short_scores = {k: v for k, v in best.detailed_scores.items() if not isinstance(v, (list, dict))}
        if short_scores:
            score_info = f" <span class='tree-meta'>({', '.join(f'{k}={v}' for k, v in short_scores.items())})</span>"
    
    leftover_info = ""
    if best.leftover_sorted:
        words_preview = html.escape(", ".join(list(map(str, best.leftover_words))[:3]))
        leftover_info = f"<div class='tree-meta'>Leftover: <code>{html.escape(best.leftover_sorted)}</code> &asymp; [{words_preview}]</div>"

    strictness_map = {
        "SUBSTRING": "block",
        "SUBSEQUENCE": "ordered",
        "MULTISET": "multiset"
    }
    strictness_label = strictness_map.get(best.strictness, best.strictness.lower())
    
    # Always show for deletions/insertions as requested. 
    # For others, only show if not MULTISET (to avoid clutter on anagrams)
    if s.op in ["WORD_DELETION", "WORD_INSERTION", "LETTER_DELETION", "LETTER_REPLACEMENT"]:
        strictness_info = f" <small style='color: #666;'>[{strictness_label}]</small>"
    else:
        strictness_info = f" <small style='color: #666;'>[{strictness_label}]</small>" if best.strictness != "MULTISET" else ""
    
    html_out = (
        f"<div class='tree-leaf'>"
        f"<div><span class='tree-target'>{target}</span></div>"
        f"<div style='margin-top: 4px;'>"
        f"<span class='tree-op-badge'>{op_label}</span>"
        f"<span class='tree-source-text'>{source}</span>"
        f"{strictness_info}"
        f"{score_info}"
        f"</div>"
        f"{leftover_info}"
        f"</div>"
    )
    
    if s.child or s.chunk_steps:
        html_out += "<div class='tree-branch'>"
        if s.child:
            html_out += render_step_html(s.child)
        for chunk in s.chunk_steps:
            html_out += render_step_html(chunk)
        html_out += "</div>"
        
    return html_out

def render_clue_details(s, best_c, finder, key_suffix=""):
    t_tree, t_inds, t_syns = st.tabs(["🏗️ Logical Tree", "📖 Indicators", "🔄 Substitutions"])
    
    with t_tree:
        html = f"<div class='tree-container'>{render_step_html(s)}</div>"
        st.markdown(html, unsafe_allow_html=True)
        with st.expander("Technical View (Raw)"):
            st.code(pretty(s), language="text")

    with t_inds:
        st.markdown("#### 📖 Suggested Indicators")
        if finder.indicator_suggestor:
            inds = get_indicator_suggestions(s.op, best_c.source, finder.indicator_suggestor)
            if inds:
                st.write(", ".join([f"`{i}`" for i in inds]))
            else:
                st.write("No indicators suggested.")
        else:
            st.write("AI disabled.")

    with t_syns:
        st.markdown("#### 🔄 Fodder Substitutions")
        if finder.indicator_suggestor:
            syns = get_fodder_synonyms(best_c.source, finder.indicator_suggestor)
            if syns:
                for word, options in syns.items():
                    st.write(f"**{word}** ➡️ {', '.join([f'`{o}`' for o in options[:5]])}")
            else:
                st.write("No substitutions found.")
        else:
            st.write("AI disabled.")

# --- Main Content ---
st.title("🧩 Boetticher")
st.markdown("""
**Boetticher** is your intelligent assistant for building cryptic crossword clues. 
Enter a target word, select your desired complexity and wordplay types, and Boetticher will 
find the best mathematical patterns, score them for surface reading quality, and suggest 
indicators and synonyms to help you write the perfect clue.
""")

tab_search, tab_favorites, tab_history = st.tabs([
    "🔍 Find Clues", 
    f"🌟 Starred Clues ({len(st.session_state.starred_clues)})", 
    f"📜 Search History ({len(st.session_state.search_history)})"
])

with tab_search:
    finder = get_finder_instance()
    if not target_word:
        st.info("👈 Enter a word in the sidebar to start building clues.")
    else:
        # Perform Search
        if st.session_state.is_searching:
            if st.button("⏹️ Cancel Search"):
                st.session_state.is_searching = False
                st.rerun()

            all_steps = get_search_results(
                target_word, 
                num_chunks, 
                max_fodder, 
                use_positional, 
                tuple(selected_ops_to_run), 
                use_llm_ranking, 
                use_llm_suggestions, 
                use_llm_for_positional
            )
            
            st.session_state.all_steps = all_steps
            st.session_state.last_target = target_word
            st.session_state.is_searching = False
            
            # Add to history (avoid duplicates)
            history_targets = [h['target'] for h in st.session_state.search_history]
            if target_word not in history_targets:
                st.session_state.search_history.insert(0, {
                    "target": target_word,
                    "count": len(all_steps),
                    "steps": all_steps
                })
            else:
                # Update existing entry and move to top
                for i, h in enumerate(st.session_state.search_history):
                    if h['target'] == target_word:
                        st.session_state.search_history.pop(i)
                        break
                st.session_state.search_history.insert(0, {
                    "target": target_word,
                    "count": len(all_steps),
                    "steps": all_steps
                })
            
            st.rerun() # Refresh to show results properly

        all_steps = st.session_state.all_steps

        if not all_steps:
            if st.session_state.last_target == target_word and target_word != "":
                st.error(f"No valid clue patterns found for '{target_word}'. Try reducing complexity.")
            else:
                st.info("Click 'Generate Clues' in the sidebar to begin.")
        else:
            display_word = st.session_state.last_target or target_word
            st.subheader(f"Clue Ideas for `{display_word.upper()}`")
            
            # --- Target Definitions (Top of Page) ---
            if finder.llm_scorer:
                with st.container(border=True):
                    defs = get_definitions(display_word, finder.llm_scorer)
                    if defs:
                        st.markdown(f"**Definitions for {display_word.upper()}:** " + ", ".join([f"`{d}`" for d in defs]))
                    else:
                        st.caption(f"No AI definitions found for '{display_word}'.")
            
            col_a, col_b = st.columns([1, 1])
            with col_a:
                st.metric("Total Patterns Found", len(all_steps))
            # Removing AI API Calls metric as requested

            # --- Filters ---
            all_ops_found = sorted(list(set(s.op for s in all_steps)))
            all_complexities = sorted(list(set(s.complexity for s in all_steps)))
            
            f_col1, f_col2, f_col3 = st.columns([2, 2, 1])
            with f_col1:
                with st.popover("Filter by Clue Types", use_container_width=True):
                    filter_all = st.checkbox("Show All Types", value=True, key="filter_all")
                    selected_filters = []
                    for op in all_ops_found:
                        checked = filter_all or op in st.session_state.get("last_selected_filters", all_ops_found)
                        icon = OP_ICONS.get(op, "")
                        if st.checkbox(f"{icon} {op}", value=checked, disabled=filter_all, key=f"filter_op_{op}"):
                            selected_filters.append(op)
                    
                    if not filter_all:
                        st.session_state.last_selected_filters = selected_filters
                    else:
                        selected_filters = all_ops_found
            with f_col2:
                selected_complexities = st.multiselect("Complexity", options=all_complexities, default=all_complexities, help="1=Single op, 2=Nested/2-part, 3=3-part")
            with f_col3:
                # Spacer or additional metrics if needed
                pass
            
            filtered_steps = [
                s for s in all_steps 
                if s.op in selected_filters 
                and s.best_score() < 150 
                and s.complexity in selected_complexities
            ]

            if not filtered_steps:
                st.info("No patterns found matching current filters/threshold.")
            else:
                # --- Pagination ---
                p_col1, p_col2 = st.columns([1, 4])
                with p_col1:
                    page_size = st.selectbox("Clues per page", [10, 20, 50], index=1)
                
                total_pages = (len(filtered_steps) - 1) // page_size + 1
                
                # Reset page_num if it's out of bounds
                if st.session_state.page_num > total_pages:
                    st.session_state.page_num = 1
                
                if total_pages > 1:
                    st.write("")
                    p_cols = st.columns([5, 1, 2, 1, 5])
                    with p_cols[1]:
                        if st.button("⬅️", disabled=(st.session_state.page_num <= 1), key="prev_p", use_container_width=True):
                            st.session_state.page_num -= 1
                            st.rerun()
                    with p_cols[2]:
                        st.markdown(f"<div style='text-align: center; padding-top: 5px;'>Page <b>{st.session_state.page_num}</b> of {total_pages}</div>", unsafe_allow_html=True)
                    with p_cols[3]:
                        if st.button("➡️", disabled=(st.session_state.page_num >= total_pages), key="next_p", use_container_width=True):
                            st.session_state.page_num += 1
                            st.rerun()
                    st.write("")
                
                page_num = st.session_state.page_num
                start_idx = (page_num - 1) * page_size
                end_idx = start_idx + page_size
                
                st.write(f"Showing {start_idx + 1} - {min(end_idx, len(filtered_steps))} of {len(filtered_steps)}")

                # --- Results ---
                starred_keys = {(sc['source'], sc['target']) for sc in st.session_state.starred_clues}
                for i, s in enumerate(filtered_steps[start_idx:end_idx], start=start_idx):
                    best_c = s.best()
                    op_label = s.op if not s.child else "NESTED"
                    icon = OP_ICONS.get(op_label, "❓")
                    
                    label = f"{icon} {s.op}{' (Nested)' if s.child else ''}: {best_c.source}"
                    
                    # Row with Star, Generate, and Expander
                    col_star, col_gen, col_main = st.columns([0.5, 0.5, 9.0])
                    
                    is_starred = (best_c.source, s.target) in starred_keys
                    
                    with col_star:
                        star_icon = "🌟" if is_starred else "⭐"
                        if st.button(star_icon, key=f"star_{s.target}_{s.op}_{best_c.source}_{i}", help="Star/Unstar", type="primary" if is_starred else "secondary"):
                            if not is_starred:
                                st.session_state.starred_clues.append({
                                    "target": s.target,
                                    "op": s.op,
                                    "source": best_c.source,
                                    "score": best_c.score,
                                    "step": s
                                })
                                st.toast(f"Starred {best_c.source}!")
                            else:
                                st.session_state.starred_clues = [sc for sc in st.session_state.starred_clues if not (sc['source'] == best_c.source and sc['target'] == s.target)]
                                st.toast("Unstarred clue.")
                            st.rerun()

                    with col_gen:
                        gen_clicked = st.button("📝", key=f"gen_btn_{s.target}_{s.op}_{best_c.source}_{i}", help="Generate Full Clue")

                    with col_main:
                        with st.expander(label, expanded=gen_clicked):
                            score_display = f"{best_c.score:.2f}" if best_c.score < 150 else "N/A"
                            st.markdown(f'<span class="score-badge">Score: {score_display}</span>', unsafe_allow_html=True)
                            st.write("") # Spacer for float
                            
                            if gen_clicked:
                                with st.spinner("Writing clue..."):
                                    clue = suggest_full_clue(finder, s, best_c)
                                    st.success(f"**AI Recommended Clue:** {clue}")
                                    st.code(clue, language="text")
                                    st.divider()

                            render_clue_details(s, best_c, finder, key_suffix=f"search_{i}")

with tab_favorites:
    finder = get_finder_instance()
    if not st.session_state.starred_clues:
        st.info("No clues starred yet. Click the star icon on any result to save it here.")
    else:
        # Sort favorites by score
        sorted_favs = sorted(st.session_state.starred_clues, key=lambda x: x['score'])
        
        # Pagination for favorites
        fav_page_size = 10
        total_fav_pages = (len(sorted_favs) - 1) // fav_page_size + 1
        
        if st.session_state.fav_page_num > total_fav_pages:
            st.session_state.fav_page_num = 1
            
        if total_fav_pages > 1:
            fp_cols = st.columns([5, 1, 2, 1, 5])
            with fp_cols[1]:
                if st.button("⬅️", disabled=(st.session_state.fav_page_num <= 1), key="prev_fav_p", use_container_width=True):
                    st.session_state.fav_page_num -= 1
                    st.rerun()
            with fp_cols[2]:
                st.markdown(f"<div style='text-align: center; padding-top: 5px;'>Page <b>{st.session_state.fav_page_num}</b> of {total_fav_pages}</div>", unsafe_allow_html=True)
            with fp_cols[3]:
                if st.button("➡️", disabled=(st.session_state.fav_page_num >= total_fav_pages), key="next_fav_p", use_container_width=True):
                    st.session_state.fav_page_num += 1
                    st.rerun()
        
        fav_start_idx = (st.session_state.fav_page_num - 1) * fav_page_size
        fav_end_idx = fav_start_idx + fav_page_size
        
        for i, sc in enumerate(sorted_favs[fav_start_idx:fav_end_idx]):
            abs_i = fav_start_idx + i
            s = sc['step']
            best_c = s.best()
            op_label = s.op if not s.child else "NESTED"
            icon = OP_ICONS.get(op_label, "❓")
                
            label = f"{icon} {s.op} for {sc['target'].upper()}: {best_c.source}"
                
            c1, c2, c3 = st.columns([0.5, 0.5, 9.0])
            with c1:
                if st.button("❌", key=f"remove_star_{abs_i}_{best_c.source}", help="Remove Favorite"):
                    st.session_state.starred_clues = [f for f in st.session_state.starred_clues if not (f['source'] == best_c.source and f['target'] == sc['target'])]
                    st.rerun()
            with c2:
                gen_clicked_fav = st.button("📝", key=f"gen_btn_fav_{abs_i}_{best_c.source}", help="Generate Full Clue")
            
            with c3:
                with st.expander(label, expanded=gen_clicked_fav):
                    score_display = f"{best_c.score:.2f}" if best_c.score < 150 else "N/A"
                    st.markdown(f'<span class="score-badge">Score: {score_display}</span>', unsafe_allow_html=True)
                    st.write("")
                    
                    if gen_clicked_fav:
                        with st.spinner("Writing clue..."):
                            clue = suggest_full_clue(finder, s, best_c)
                            st.success(f"**AI Recommended Clue:** {clue}")
                            st.code(clue, language="text")
                            st.divider()

                    render_clue_details(s, best_c, finder, key_suffix=f"fav_{abs_i}")

with tab_history:
    if not st.session_state.search_history:
        st.info("Your search history will appear here.")
    else:
        h_col1, h_col2 = st.columns([3, 1])
        with h_col1:
            st.subheader("Recent Searches")
        with h_col2:
            st.write("") # alignment
            if st.button("🗑️ Clear All", use_container_width=True, help="Clear all search history"):
                st.session_state.search_history = []
                st.session_state.hist_page_num = 1
                st.rerun()
        
        # Pagination for history
        hist_page_size = 5
        total_hist_pages = (len(st.session_state.search_history) - 1) // hist_page_size + 1
        
        if st.session_state.hist_page_num > total_hist_pages:
            st.session_state.hist_page_num = 1
            
        if total_hist_pages > 1:
            hp_cols = st.columns([5, 1, 2, 1, 5])
            with hp_cols[1]:
                if st.button("⬅️", disabled=(st.session_state.hist_page_num <= 1), key="prev_hist_p", use_container_width=True):
                    st.session_state.hist_page_num -= 1
                    st.rerun()
            with hp_cols[2]:
                st.markdown(f"<div style='text-align: center; padding-top: 5px;'>Page <b>{st.session_state.hist_page_num}</b> of {total_hist_pages}</div>", unsafe_allow_html=True)
            with hp_cols[3]:
                if st.button("➡️", disabled=(st.session_state.hist_page_num >= total_hist_pages), key="next_hist_p", use_container_width=True):
                    st.session_state.hist_page_num += 1
                    st.rerun()
        
        hist_start_idx = (st.session_state.hist_page_num - 1) * hist_page_size
        hist_end_idx = hist_start_idx + hist_page_size
        
        for i, entry in enumerate(st.session_state.search_history[hist_start_idx:hist_end_idx]):
            abs_i = hist_start_idx + i
            with st.container(border=True):
                c1, c2, c3 = st.columns([3, 1, 1])
                with c1:
                    st.write(f"**{entry['target'].upper()}**")
                    st.caption(f"{entry['count']} patterns found")
                with c2:
                    if st.button("📂 Load", key=f"load_hist_{abs_i}_{entry['target']}"):
                        st.session_state.all_steps = entry['steps']
                        st.session_state.last_target = entry['target']
                        st.rerun()
                with c3:
                    if st.button("🗑️ Delete", key=f"del_hist_{abs_i}_{entry['target']}"):
                        st.session_state.search_history.pop(abs_i)
                        st.rerun()

# --- Footer ---
st.caption("Boetticher v0.1.0 • Built with Streamlit & GPT-4o-mini")
