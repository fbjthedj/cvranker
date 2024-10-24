import streamlit as st
import pandas as pd
import re
from io import StringIO
from collections import Counter
from pdfminer.high_level import extract_text_to_fp
from pdfminer.layout import LAParams
import concurrent.futures
import yake

# Set page config at the very beginning of the script
st.set_page_config(layout="wide", page_title="Aceli CV Parser and Ranker", page_icon="🌍")

def extract_keywords_from_job_description(text, num_keywords=15):
    # Configure YAKE keyword extractor
    language = "en"
    max_ngram_size = 3  # Allow up to 3-word phrases
    deduplication_threshold = 0.9
    numOfKeywords = num_keywords
    custom_kw_extractor = yake.KeywordExtractor(
        lan=language, 
        n=max_ngram_size, 
        dedupLim=deduplication_threshold, 
        top=numOfKeywords, 
        features=None
    )
    
    # Extract keywords
    keywords = custom_kw_extractor.extract_keywords(text)
    # Return just the keywords, not their scores
    return [keyword[0] for keyword in keywords]

def extract_text_from_pdf(file):
    output_string = StringIO()
    laparams = LAParams()
    extract_text_to_fp(file, output_string, laparams=laparams)
    return output_string.getvalue()

def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text.lower())
    return text

def calculate_keyword_similarity(cv_text, keywords):
    cv_words = set(cv_text.split())
    keyword_count = sum(1 for keyword in keywords if keyword.lower() in cv_words)
    return keyword_count / len(keywords) if keywords else 0

def calculate_keyword_frequency(cv_text, keywords):
    cv_words = cv_text.split()
    keyword_freq = sum(cv_words.count(keyword.lower()) for keyword in keywords)
    return min(keyword_freq, 100)  # Cap at 100

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
        keyword_similarity = calculate_keyword_similarity(processed_text, keywords)
        keyword_frequency = calculate_keyword_frequency(processed_text, keywords)
        relevant_sentences = extract_relevant_sentences(text, keywords)
        return {
            "Filename": file.name,
            "Keyword Similarity": keyword_similarity,
            "Keyword Frequency": keyword_frequency,
            "Overall Score": keyword_similarity,
            "Relevant Sentences": relevant_sentences,
            "Full Text": text
        }
    except Exception as e:
        return {
            "Filename": file.name,
            "Error": str(e)
        }

def display_results(df, keyword_similarity_threshold, frequency_threshold):
    filtered_df = df[
        (df['Keyword Similarity'] >= keyword_similarity_threshold / 100) & 
        (df['Keyword Frequency'] >= frequency_threshold)
    ]
    filtered_df = filtered_df.sort_values("Overall Score", ascending=False).reset_index(drop=True)
    
    st.header("Ranked CVs")
    st.dataframe(filtered_df[["Filename", "Overall Score", "Keyword Similarity", "Keyword Frequency"]])

    if not filtered_df.empty:
        st.success(f"Selected Candidates (Keyword Similarity ≥ {keyword_similarity_threshold}% and Keyword Frequency ≥ {frequency_threshold}):")
        for _, candidate in filtered_df.iterrows():
            st.markdown(f"- **{candidate['Filename']}** (Score: {candidate['Overall Score']:.2%}, Keyword Frequency: {candidate['Keyword Frequency']})")
    else:
        st.warning("No candidates meet all the thresholds. Here are some suggestions:")
        
        # Calculate maximum scores to guide users
        max_keyword_sim = df['Keyword Similarity'].max() * 100
        max_freq = df['Keyword Frequency'].max()
        
        st.markdown(f"""
        Try the following adjustments:
        1. **Lower the thresholds**: 
           - Maximum Keyword Similarity in your candidates: {max_keyword_sim:.1f}%
           - Maximum Keyword Frequency: {int(max_freq)}
        2. **Review your keywords**:
           - Use more specific job-related skills and qualifications
           - Check for common variations or synonyms of key terms
           - Ensure keywords match the language used in the industry
        """)

    return filtered_df

def main():
    st.title("🌍 Aceli CV Parser and Ranker")
    st.markdown("### App Instructions")

    with st.expander("How to Use"):
        st.markdown("""
        1. Either:
           - Enter a job description and click "Extract Keywords", OR
           - Manually input keywords separated by commas
        2. Review and edit the keywords as needed
        3. Upload PDF CV files using the file uploader
        4. Click "Process and Rank CVs" to analyze the files
        5. Review the initial results
        6. Adjust the thresholds if needed
        7. Click "Update Rankings" to filter top candidates
        8. Review the results, keyword frequency, and relevant sentences
        """)

    st.markdown("---")

    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'processed' not in st.session_state:
        st.session_state.processed = False
    if 'filtered_df' not in st.session_state:
        st.session_state.filtered_df = None
    if 'extracted_keywords' not in st.session_state:
        st.session_state.extracted_keywords = None

    # Keyword Input Section
    st.header("Keywords")
    keyword_input_method = st.radio(
        "Choose how to input keywords",
        ["Extract from Job Description", "Manual Input"]
    )

    if keyword_input_method == "Extract from Job Description":
        job_description = st.text_area("Enter the job description:", height=150)
        if st.button("Extract Keywords"):
            if job_description:
                extracted_keywords = extract_keywords_from_job_description(job_description)
                st.session_state.extracted_keywords = extracted_keywords
                st.success("Keywords extracted successfully!")
            else:
                st.warning("Please enter a job description.")

        # Show and allow editing of extracted keywords
        if st.session_state.extracted_keywords:
            st.subheader("Extracted Keywords (edit as needed):")
            keywords = st.text_area(
                "Edit Keywords:",
                value=", ".join(st.session_state.extracted_keywords),
                height=100,
                help="Edit the extracted keywords. Keep them comma-separated."
            )
            keywords = [keyword.strip().lower() for keyword in keywords.split(',') if keyword.strip()]
    else:
        keywords = st.text_input("Enter keywords (comma-separated):")
        keywords = [keyword.strip().lower() for keyword in keywords.split(',') if keyword.strip()]

    st.header("Upload CVs")
    uploaded_files = st.file_uploader("Choose PDF CV files", accept_multiple_files=True, type=['pdf'])
    st.info(f"Number of files uploaded: {len(uploaded_files)}")

    if st.button("Process and Rank CVs"):
        if not uploaded_files:
            st.warning("Please upload some PDF CV files.")
            return
        if not keywords:
            st.warning("Please provide keywords for analysis.")
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

        successful_results = [r for r in results if "Error" not in r]
        error_results = [r for r in results if "Error" in r]

        st.session_state.df = pd.DataFrame(successful_results)
        st.session_state.processed = True

        if error_results:
            st.header("Errors")
            for result in error_results:
                st.error(f"Error processing {result['Filename']}: {result['Error']}")

        # Display initial results
        initial_keyword_similarity_threshold = 60
        initial_frequency_threshold = 1
        st.session_state.filtered_df = display_results(st.session_state.df, initial_keyword_similarity_threshold, initial_frequency_threshold)

    if st.session_state.processed and st.session_state.df is not None:
        st.header("Adjust Ranking Thresholds")
        
        col1, col2 = st.columns(2)
        with col1:
            keyword_similarity_threshold = st.number_input("Keyword similarity threshold (%)", 
                                                           min_value=0, max_value=100, value=60, step=1)
        with col2:
            frequency_threshold = st.number_input("Keyword frequency threshold", 
                                                  min_value=0, max_value=100, value=1, step=1)
        
        if st.button("Update Rankings"):
            st.session_state.filtered_df = display_results(st.session_state.df, keyword_similarity_threshold, frequency_threshold)

        st.header("Keyword Frequency")
        all_text = " ".join(r["Full Text"] for _, r in st.session_state.df.iterrows())
        word_freq = Counter(preprocess_text(all_text).split())
        keyword_freq = {word: freq for word, freq in word_freq.items() if word in keywords}
        st.bar_chart(keyword_freq)

        st.header("Relevant Sentences from CVs")
        if st.session_state.filtered_df is not None and not st.session_state.filtered_df.empty:
            for _, result in st.session_state.filtered_df.iterrows():
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
        else:
            st.info("No CVs match the current thresholds. Try adjusting the thresholds and updating the rankings.")

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
