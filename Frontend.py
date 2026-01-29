import streamlit as st
from Backend import run_rag


st.set_page_config(page_title="YouTube Q&A", layout="wide")

st.title("🎥 YouTube Video Q&A")
st.caption("Ask questions from caption-enabled YouTube videos")

# -------- Sidebar --------
st.sidebar.header("Controls")
show_sources = st.sidebar.checkbox("Show Sources", True)

# -------- Layout --------
col1, col2 = st.columns([3, 2])

with col1:
    youtube_url = st.text_input(
        "YouTube URL",
        placeholder="https://www.youtube.com/watch?v=..."
    )

    if youtube_url:
        st.video(youtube_url)

with col2:
    query = st.text_input(
        "Ask a question",
        placeholder="What is self-attention?"
    )

    ask = st.button("Ask")

    if ask:
        if not youtube_url or not query:
            st.warning("Please enter both URL and question")
        else:
            with st.spinner("Thinking..."):
                try:
                    answer, sources = run_rag(youtube_url, query)
                    st.markdown("### Answer")
                    st.write(answer)
                    
                    if show_sources:
                        with st.expander("Sources"):
                            for i, doc in enumerate(sources, 1):
                                st.write(f"{i}. {doc.page_content[:300]}...")
                except ValueError as e:
                    st.warning("⚠️ This video does not provide captions. Try another video.")
