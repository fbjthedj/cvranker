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

def extract_text_from_pdf(file):
    output_string = StringIO()
    laparams = LAParams()
    extract_text_to_fp(file, output_string, laparams=laparams)
    return output_string.getvalue()

def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text.lower())
    return text

def calculate_similarity(cv_text, keywords):
    cv_words = set(cv_text.split())
    keyword_count = sum(1 for keyword in keywords if keyword.lower() in cv_words)
    return keyword_count / len(keywords) if keywords else 0

def calculate_keyword_frequency(cv_text, keywords):
    cv_words = cv_text.split()
    keyword_freq = sum(cv_words.count(keyword.lower()) for keyword in keywords)
    return keyword_freq

def extract_relevant_sentences(text, keywords):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    relevant_sentences = []
    for sentence in sentences:
        if any(keyword.lower() in sentence.lower() for keyword in keywords):
            relevant_sentences.append(sentence.strip())
    return relevant_sentences

@st.cache_data
def process_cv(file, keywords):
    try:
        text = extract_text_from_pdf(file)
        processed_text = preprocess_text(text)
        similarity_score = calculate_similarity(processed_text, keywords)
        keyword_frequency = calculate_keyword_frequency(processed_text, keywords)
        relevant_sentences = extract_relevant_sentences(text, keywords)
        return {
            "Filename": file.name,
            "Similarity Score": similarity_score,
            "Keyword Frequency": keyword_frequency,
            "Relevant Sentences": relevant_sentences,
            "Full Text": text
        }
    except Exception as e:
        return {
            "Filename": file.name,
            "Error": str(e)
        }

def main():
    st.title("🌍 Aceli CV Parser and Ranker")
    st.markdown("### App Instructions")

    # How to Use guide
    with st.expander("How to Use"):
        st.markdown("""
        1. Enter the job description in the text area provided.
        2. Input relevant keywords separated by commas in the designated field.
        3. Upload PDF CV files using the file uploader.
        4. Click the "Process and Rank CVs" button to analyze the uploaded files.
        5. Enter the desired similarity score and keyword frequency thresholds.
        6. Click "Update Rankings" to filter top candidates based on the new thresholds.
        7. Review the ranked results, selected candidates, keyword frequency, and similarity scores.
        8. Expand individual CV sections to view relevant sentences containing the keywords from each document.
        """)

    # Add a line separator after the How to Use section
    st.markdown("---")

    # Initialize session state
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'processed' not in st.session_state:
        st.session_state.processed = False

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
        st.session_state.df = pd.DataFrame(successful_results)
        st.session_state.processed = True

        # Display errors, if any
        if error_results:
            st.header("Errors")
            for result in error_results:
                st.error(f"Error processing {result['Filename']}: {result['Error']}")

    if st.session_state.processed and st.session_state.df is not None:
        st.header("Ranking Thresholds")
        
        col1, col2 = st.columns(2)
        with col1:
            similarity_threshold = st.number_input("Similarity score threshold (%)", 
                                                   min_value=0, max_value=100, value=60, step=1)
        with col2:
            max_frequency = int(st.session_state.df["Keyword Frequency"].max())
            frequency_threshold = st.number_input("Keyword frequency threshold", 
                                                  min_value=0, max_value=max_frequency, 
                                                  value=max_frequency // 2, step=1)
        
        if st.button("Update Rankings"):
            # Filter and display ranked CVs
            filtered_df = st.session_state.df[
                (st.session_state.df['Similarity Score'] >= similarity_threshold / 100) & 
                (st.session_state.df['Keyword Frequency'] >= frequency_threshold)
            ]
            filtered_df = filtered_df.sort_values("Similarity Score", ascending=False).reset_index(drop=True)
            
            st.header("Ranked CVs")
            # Display the DataFrame without custom styling
            st.dataframe(filtered_df[["Filename", "Similarity Score", "Keyword Frequency"]])

            # Display selected candidates
            if not filtered_df.empty:
                st.success(f"Selected Candidates (Similarity Score ≥ {similarity_threshold}% and Keyword Frequency ≥ {frequency_threshold}):")
                for _, candidate in filtered_df.iterrows():
                    st.markdown(f"- **{candidate['Filename']}** (Score: {candidate['Similarity Score']:.2%}, Keyword Frequency: {candidate['Keyword Frequency']})")
            else:
                st.warning(f"No candidates meet both the {similarity_threshold}% similarity threshold and the keyword frequency threshold of {frequency_threshold}.")

            # Keyword Frequency
            st.header("Keyword Frequency")
            all_text = " ".join(r["Full Text"] for _, r in st.session_state.df.iterrows())
            word_freq = Counter(preprocess_text(all_text).split())
            keyword_freq = {word: freq for word, freq in word_freq.items() if word in keywords}
            st.bar_chart(keyword_freq)

            # Relevant Sentences
            st.header("Relevant Sentences from CVs")
            for _, result in filtered_df.iterrows():
                with st.expander(f"Show relevant sentences from {result['Filename']}"):
                    if result['Relevant Sentences']:
                        for sentence in result['Relevant Sentences']:
                            highlighted_sentence = sentence
                            for keyword in keywords:
                                highlighted_sentence = re.sub(
                                    f'({re.escape(keyword)})',
                                    r'<span style="background-color: yellow; font-weight: bold;">\1</span>',
                                    highlighted_sentence,
                                    flags=re.IGNORECASE
                                )
                            st.markdown(f"• {highlighted_sentence}", unsafe_allow_html=True)
                    else:
                        st.info("No sentences with keywords found in this CV.")

    st.markdown(
        """
        <div style="margin-top: 2rem; text-align: center; font-size: 0.8rem; color: #888888;">
            Designed by Aceli Africa
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
