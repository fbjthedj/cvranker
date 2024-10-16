import streamlit as st
import pandas as pd
import re
from io import StringIO
from collections import Counter
from pdfminer.high_level import extract_text_to_fp
from pdfminer.layout import LAParams
import concurrent.futures

# Set page config at the very beginning of the script
st.set_page_config(layout="wide", page_title="Aceli CV Parser and Ranker", page_icon="🌍")

# ... (previous helper functions remain unchanged)

def main():
    # Custom CSS for Notion-inspired UI
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap');
    
    :root {
        --notion-black: #000000;
        --notion-white: #ffffff;
        --notion-blue: #2997ff;
    }
    
    body {
        font-family: 'Inter', sans-serif;
        background-color: var(--notion-white);
        color: var(--notion-black);
    }
    .main {
        padding: 2rem;
        max-width: 1200px;
        margin: 0 auto;
    }
    h1 {
        font-size: 4rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    h2 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    h3 {
        font-size: 1.5rem;
        font-weight: 700;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }
    .stButton > button {
        background-color: var(--notion-blue);
        color: var(--notion-white);
        font-weight: 600;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 4px;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton > button:hover {
        background-color: #1a7ae2;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .stTextInput > div > div > input, .stTextArea > div > div > textarea {
        border: 1px solid #e0e0e0;
        border-radius: 4px;
        padding: 0.5rem;
        font-size: 1rem;
    }
    .stDataFrame {
        border: 1px solid #e0e0e0;
        border-radius: 4px;
        overflow: hidden;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stDataFrame th {
        background-color: #f7f7f7;
        color: var(--notion-black);
        font-weight: 600;
    }
    .highlight {
        background-color: #ffeaa7;
        padding: 0 2px;
        border-radius: 2px;
    }
    .footer {
        margin-top: 4rem;
        text-align: center;
        font-size: 0.9rem;
        color: #777;
    }
    .trust-logos {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-top: 2rem;
        gap: 2rem;
    }
    .trust-logos img {
        height: 30px;
        opacity: 0.6;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("Build something beautiful.")
    st.markdown("### Aceli CV Parser and Ranker helps you and your team find the best candidates with peace of mind.")

    # Main content
    col1, col2 = st.columns([3, 2])

    with col1:
        st.header("Get started")
        job_description = st.text_area("Enter the job description:", height=150)
        keywords = st.text_input("Enter keywords (comma-separated):")
        keywords = [keyword.strip().lower() for keyword in keywords.split(',') if keyword.strip()]

        uploaded_files = st.file_uploader("Choose PDF CV files", accept_multiple_files=True, type=['pdf'])
        st.info(f"Number of files uploaded: {len(uploaded_files)}")

        if st.button("Process and Rank CVs"):
            # ... (processing logic remains unchanged)
            pass  # Remove this 'pass' and uncomment your actual processing logic

    with col2:
        st.image("https://via.placeholder.com/400x300.png?text=Illustration", use_column_width=True)

    # How to Use guide
    with st.expander("How to Use"):
        st.markdown("""
        1. Enter the job description in the text area provided.
        2. Input relevant keywords separated by commas.
        3. Upload PDF CV files.
        4. Click "Process and Rank CVs" to analyze the files.
        5. Adjust thresholds to filter top candidates.
        6. Review results and relevant sentences from CVs.
        """)

    # Trust logos
    st.markdown("""
    <div class="trust-logos">
        <img src="https://via.placeholder.com/100x30.png?text=Logo1" alt="Client Logo 1">
        <img src="https://via.placeholder.com/100x30.png?text=Logo2" alt="Client Logo 2">
        <img src="https://via.placeholder.com/100x30.png?text=Logo3" alt="Client Logo 3">
    </div>
    """, unsafe_allow_html=True)

    # Footer
    st.markdown(
        """
        <div class="footer">
            Designed by Aceli Africa
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
