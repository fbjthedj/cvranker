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
    # Custom CSS for improved UI with Inter font and mobile responsiveness
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;700&display=swap');
    
    :root {
        --app-blue: #1E88E5;
        --app-blue-light: #64B5F6;
    }
    
    body {
        font-family: 'Inter', sans-serif;
        background-color: #F5F5F5;
        color: #333333;
    }
    .main {
        padding: 1rem;
    }
    .stButton > button {
        background-color: var(--app-blue);
        color: white;
        font-weight: 500;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 4px;
        transition: all 0.3s ease;
        width: 100%;
        margin-top: 1rem;
    }
    .stButton > button:hover {
        background-color: var(--app-blue-light);
        color: white;
    }
    .stButton > button:active, .stButton > button:focus {
        background-color: var(--app-blue) !important;
        color: white !important;
        box-shadow: none !important;
    }
    .stTextInput > div > div > input, .stTextArea > div > div > textarea {
        background-color: white;
        border: 1px solid #E0E0E0;
        border-radius: 4px;
        padding: 0.5rem;
        color: black !important;
        font-family: 'Inter', sans-serif;
    }
    h1 {
        color: var(--app-blue);
        font-family: 'Inter', sans-serif;
        font-size: 1.8rem;
    }
    h2, h3 {
        color: var(--app-blue);
        font-family: 'Inter', sans-serif;
        font-size: 1.2rem;
    }
    .card {
        background-color: white;
        border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .stDataFrame {
        border: none;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        overflow-x: auto;
    }
    .stDataFrame th {
        background-color: var(--app-blue);
        color: white;
    }
    .dataframe {
        font-size: 12px;
        font-family: 'Inter', sans-serif;
        width: 100%;
    }
    .dataframe tbody tr:nth-child(even) {
        background-color: #f0f0f0;
    }
    .dataframe tbody tr:nth-child(odd) {
        background-color: #ffffff;
    }
    .tooltip {
        position: relative;
        display: inline-block;
        border-bottom: 1px dotted var(--app-blue);
    }
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #555;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    .footer {
        margin-top: 2rem;
        text-align: center;
        font-size: 0.8rem;
        color: #888888;
    }
    .highlight {
        background-color: yellow;
        font-weight: bold;
    }
    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .main {
            padding: 0.5rem;
        }
        h1 {
            font-size: 1.5rem;
        }
        h2, h3 {
            font-size: 1rem;
        }
        .stDataFrame {
            font-size: 10px;
        }
        .dataframe {
            font-size: 10px;
        }
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("🌍 Aceli CV Parser and Ranker")
    st.markdown("### App Instructions")

    # How to Use guide
    with st.expander("How to Use"):
        st.markdown("""
        1. Enter the job description in the text area provided.
        2. Input relevant keywords separated by commas in the designated field.
        3. Upload PDF CV files using the file uploader.
        4. Click the "Process and Rank CVs" button to analyze the uploaded files.
        5. Adjust the similarity score and keyword frequency thresholds using the sliders to filter top candidates.
        6. Review the ranked results, selected candidates, keyword frequency, and similarity scores.
        7. Expand individual CV sections to view relevant sentences containing the keywords from each document.
        """)

    # Add a line separator after the How to Use section
    st.markdown("---")

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
            st.header("Ranked CVs")
            
            # Add sliders for thresholds
            similarity_threshold = st.slider(
                "Select similarity score threshold (%)", 
                0, 100, 60,
                help="Slide to set the minimum similarity score for candidate selection."
            )
            
            max_frequency = int(df["Keyword Frequency"].max())
            frequency_threshold = st.slider(
                "Select keyword frequency threshold", 
                0, max_frequency, max_frequency // 2,
                help="Slide to set the minimum keyword frequency for candidate selection."
            )
            
            # Function to display and highlight DataFrame
            def display_ranked_cvs():
                # Filter the DataFrame based on current thresholds
                filtered_df = df[(df['Similarity Score'] >= similarity_threshold / 100) & 
                                 (df['Keyword Frequency'] >= frequency_threshold)]
                
                # Sort the filtered DataFrame
                filtered_df = filtered_df.sort_values("Similarity Score", ascending=False).reset_index(drop=True)
                
                # Function to highlight rows
                def highlight_rows(row):
                    return ['background-color: rgba(30, 136, 229, 0.2); font-weight: bold;'] * len(row)

                # Display the DataFrame with highlighting
                st.dataframe(
                    filtered_df[["Filename", "Similarity Score", "Keyword Frequency"]]
                    .style.format({"Similarity Score": "{:.2%}", "Keyword Frequency": "{:,d}"})
                    .apply(highlight_rows, axis=1)
                )

                # Display selected candidates
                if not filtered_df.empty:
                    st.success(f"Selected Candidates (Similarity Score ≥ {similarity_threshold}% and Keyword Frequency ≥ {frequency_threshold}):")
                    for _, candidate in filtered_df.iterrows():
                        st.markdown(f"- **{candidate['Filename']}** (Score: {candidate['Similarity Score']:.2%}, Keyword Frequency: {candidate['Keyword Frequency']})")
                else:
                    st.warning(f"No candidates meet both the {similarity_threshold}% similarity threshold and the keyword frequency threshold of {frequency_threshold}.")

            # Call the function to display ranked CVs
            display_ranked_cvs()

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
