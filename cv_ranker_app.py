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
    # Custom CSS for a clean, user-friendly UI
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
    
    body {
        font-family: 'Inter', sans-serif;
        background-color: #f8f9fa;
        color: #333;
    }
    .main {
        padding: 2rem;
    }
    h1 {
        color: #1E88E5;
        font-weight: 600;
    }
    h2 {
        color: #1E88E5;
        font-weight: 600;
        margin-top: 2rem;
    }
    .stButton > button {
        background-color: #1E88E5;
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 4px;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #1565C0;
    }
    .stTextInput > div > div > input, .stTextArea > div > div > textarea {
        background-color: white;
        border: 1px solid #ddd;
        border-radius: 4px;
        padding: 0.5rem;
    }
    .stDataFrame {
        border: 1px solid #ddd;
        border-radius: 4px;
        overflow: hidden;
    }
    .stDataFrame th {
        background-color: #f1f3f5;
    }
    .highlight {
        background-color: #fff3cd;
        padding: 0 2px;
        border-radius: 2px;
    }
    .footer {
        margin-top: 3rem;
        text-align: center;
        font-size: 0.9rem;
        color: #6c757d;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("🌍 Aceli CV Parser and Ranker")
    st.markdown("### Streamline your recruitment process")

    # How to Use guide
    with st.expander("How to Use"):
        st.markdown("""
        1. Enter the job description in the text area provided.
        2. Input relevant keywords separated by commas.
        3. Upload PDF CV files using the file uploader.
        4. Click "Process and Rank CVs" to analyze the files.
        5. Adjust the similarity score and keyword frequency thresholds to filter top candidates.
        6. Review the ranked results, selected candidates, and keyword frequency.
        7. Explore relevant sentences from each CV containing the keywords.
        """)

    # Job Description and Keywords
    st.header("Job Description and Keywords")
    job_description = st.text_area("Enter the job description:", height=150)
    keywords = st.text_input("Enter keywords (comma-separated):")
    keywords = [keyword.strip().lower() for keyword in keywords.split(',') if keyword.strip()]

    # Upload CVs
    st.header("Upload CVs")
    uploaded_files = st.file_uploader("Choose PDF CV files", accept_multiple_files=True, type=['pdf'])
    st.info(f"Number of files uploaded: {len(uploaded_files)}")

    if st.button("Process and Rank CVs"):
        if not uploaded_files:
            st.warning("Please upload some PDF CV files.")
            return

        results = []

        with st.spinner('Processing CVs...'):
            progress_bar = st.progress(0)
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(process_cv, file, keywords) for file in uploaded_files]
                for i, future in enumerate(concurrent.futures.as_completed(futures)):
                    result = future.result()
                    results.append(result)
                    progress_bar.progress((i + 1) / len(uploaded_files))

        # Separate successful results and errors
        successful_results = [r for r in results if "Error" not in r]
        error_results = [r for r in results if "Error" in r]

        # Create a DataFrame and sort by similarity score
        df = pd.DataFrame(successful_results)
        if not df.empty:
            df = df.sort_values("Similarity Score", ascending=False).reset_index(drop=True)

            st.header("Ranked CVs")
            
            col1, col2 = st.columns(2)
            with col1:
                similarity_threshold = st.slider(
                    "Similarity score threshold (%)", 
                    0, 100, 60,
                    help="Minimum similarity score for candidate selection."
                )
            with col2:
                max_frequency = int(df["Keyword Frequency"].max())
                frequency_threshold = st.slider(
                    "Keyword frequency threshold", 
                    0, max_frequency, max_frequency // 2,
                    help="Minimum keyword frequency for candidate selection."
                )
            
            # Function to highlight rows based on thresholds
            def highlight_selected(row):
                if row['Similarity Score'] >= similarity_threshold / 100 and row['Keyword Frequency'] >= frequency_threshold:
                    return ['background-color: #e3f2fd'] * len(row)
                return [''] * len(row)

            # Display the DataFrame with highlighting
            st.dataframe(
                df[["Filename", "Similarity Score", "Keyword Frequency"]]
                .style.format({"Similarity Score": "{:.2%}", "Keyword Frequency": "{:,d}"})
                .apply(highlight_selected, axis=1)
            )

            # Display selected candidates
            selected_candidates = df[(df['Similarity Score'] >= similarity_threshold / 100) & (df['Keyword Frequency'] >= frequency_threshold)]
            if not selected_candidates.empty:
                st.success(f"Selected Candidates (Similarity Score ≥ {similarity_threshold}% and Keyword Frequency ≥ {frequency_threshold}):")
                for _, candidate in selected_candidates.iterrows():
                    st.markdown(f"- **{candidate['Filename']}** (Score: {candidate['Similarity Score']:.2%}, Keyword Frequency: {candidate['Keyword Frequency']})")
            else:
                st.warning(f"No candidates meet both the {similarity_threshold}% similarity threshold and the keyword frequency threshold of {frequency_threshold}.")

            st.header("Keyword Frequency")
            all_text = " ".join(r["Full Text"] for r in successful_results)
            word_freq = Counter(preprocess_text(all_text).split())
            keyword_freq = {word: freq for word, freq in word_freq.items() if word in keywords}
            st.bar_chart(keyword_freq)

            st.header("Relevant Sentences from CVs")
            for i, result in enumerate(successful_results):
                with st.expander(f"Show relevant sentences from {result['Filename']}"):
                    if result['Relevant Sentences']:
                        for sentence in result['Relevant Sentences']:
                            highlighted_sentence = sentence
                            for keyword in keywords:
                                highlighted_sentence = re.sub(
                                    f'({re.escape(keyword)})',
                                    r'<span class="highlight">\1</span>',
                                    highlighted_sentence,
                                    flags=re.IGNORECASE
                                )
                            st.markdown(f"• {highlighted_sentence}", unsafe_allow_html=True)
                    else:
                        st.info("No sentences with keywords found in this CV.")

        # Display errors, if any
        if error_results:
            st.header("Errors")
            for result in error_results:
                st.error(f"Error processing {result['Filename']}: {result['Error']}")

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
