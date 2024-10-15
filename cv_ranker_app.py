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

@st.cache_data
def process_cv(file, keywords):
    try:
        text = extract_text_from_pdf(file)
        processed_text = preprocess_text(text)
        similarity_score = calculate_similarity(processed_text, keywords)
        return {
            "Filename": file.name,
            "Similarity Score": similarity_score,
            "Full Text": text
        }
    except Exception as e:
        return {
            "Filename": file.name,
            "Error": str(e)
        }

def main():
    # Custom CSS for improved UI with Inter font
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;700&display=swap');
    
    body {
        font-family: 'Inter', sans-serif;
        background-color: #F5F5F5;
        color: #333333;
    }
    .main {
        padding: 2rem;
    }
    .stButton > button {
        background-color: #1E88E5;
        color: white;
        font-weight: 500;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 4px;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #1565C0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    .stTextInput > div > div > input, .stTextArea > div > div > textarea {
        background-color: white;
        border: 1px solid #E0E0E0;
        border-radius: 4px;
        padding: 0.5rem;
        color: black !important;
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3 {
        color: #1E88E5;
        font-family: 'Inter', sans-serif;
    }
    .card {
        background-color: white;
        border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
    .stDataFrame {
        border: none;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    .stDataFrame th {
        background-color: #1E88E5;
        color: white;
    }
    .dataframe {
        font-size: 14px;
        font-family: 'Inter', sans-serif;
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
        border-bottom: 1px dotted #1E88E5;
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
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f0f0f0;
        color: #333;
        text-align: center;
        padding: 10px 0;
        font-size: 14px;
        font-family: 'Inter', sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("🌍 Aceli CV Parser and Ranker")
    st.markdown("### Streamline your recruitment process")

    # How to Use guide
    with st.expander("How to Use"):
        st.markdown("""
        1. Enter the job description in the text area provided.
        2. Input relevant keywords separated by commas in the designated field.
        3. Upload PDF CV files using the file uploader.
        4. Click the "Process and Rank CVs" button to analyze the uploaded files.
        5. Adjust the similarity score threshold using the slider to filter top candidates.
        6. Review the ranked results, selected candidates, and keyword frequency chart.
        7. Expand individual CV sections to view extracted text from each document.
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
            df = df.sort_values("Similarity Score", ascending=False).reset_index(drop=True)

            st.header("Ranked CVs")
            
            # Add a slider for selection threshold with tooltip
            threshold = st.slider(
                "Select similarity score threshold (%)", 
                0, 100, 60,
                help="Slide to set the minimum similarity score for candidate selection."
            )
            
            # Function to highlight rows based on threshold
            def highlight_selected(row):
                if row['Similarity Score'] >= threshold / 100:
                    return ['background-color: rgba(30, 136, 229, 0.2); font-weight: bold;'] * len(row)
                return [''] * len(row)

            # Display the DataFrame with highlighting
            st.dataframe(
                df[["Filename", "Similarity Score"]]
                .style.format({"Similarity Score": "{:.2%}"})
                .apply(highlight_selected, axis=1)
            )

            # Display selected candidates
            selected_candidates = df[df['Similarity Score'] >= threshold / 100]
            if not selected_candidates.empty:
                st.success(f"Selected Candidates (Similarity Score ≥ {threshold}%):")
                for _, candidate in selected_candidates.iterrows():
                    st.markdown(f"- **{candidate['Filename']}** (Score: {candidate['Similarity Score']:.2%})")
            else:
                st.warning(f"No candidates meet the {threshold}% similarity threshold.")

            st.header("Keyword Frequency")
            all_text = " ".join(r["Full Text"] for r in successful_results)
            word_freq = Counter(preprocess_text(all_text).split())
            keyword_freq = {word: freq for word, freq in word_freq.items() if word in keywords}
            st.bar_chart(keyword_freq)

            st.header("Extracted Text from CVs")
            for i, result in enumerate(successful_results):
                with st.expander(f"Show extracted text from {result['Filename']}"):
                    st.text_area(f"Extracted text (first 1000 characters)", 
                                 result['Full Text'][:1000], 
                                 height=200)
                    if len(result['Full Text']) > 1000:
                        st.info("Text truncated. Showing first 1000 characters.")

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
