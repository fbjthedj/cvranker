import streamlit as st
import pandas as pd
import re
from io import StringIO
from pdfminer.high_level import extract_text_to_fp
from pdfminer.layout import LAParams
import concurrent.futures
import google.generativeai as genai
from typing import Dict, List

# Custom CSS for Notion-like styling
def set_custom_style():
    st.markdown("""
        <style>
        /* Global Notion-like styling */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        /* Main container */
        .main {
            max-width: 1000px;
            margin: 0 auto;
            padding: 2rem;
            color: rgb(55, 53, 47);
        }
        
        /* Typography */
        h1, h2, h3, p {
            color: rgb(55, 53, 47);
        }
        
        h1 {
            font-weight: 700;
            font-size: 2.5em;
            line-height: 1.2;
            margin-bottom: 0.5em;
        }
        
        h2 {
            font-weight: 600;
            font-size: 1.5em;
            margin: 1.4em 0 0.5em;
        }
        
        h3 {
            font-weight: 600;
            font-size: 1.2em;
            margin: 1em 0 0.5em;
        }
        
        p {
            font-size: 1em;
            line-height: 1.5;
            margin-bottom: 0.5em;
        }
        
        /* Input fields */
        .stTextInput > div > div > input,
        .stTextArea > div > div > textarea {
            background: white;
            border: 1px solid rgba(55, 53, 47, 0.16);
            border-radius: 3px;
            padding: 0.5em;
            font-size: 1em;
        }
        
        /* Buttons */
        .stButton > button {
            background: rgb(35, 131, 226);
            color: white;
            border: none;
            border-radius: 3px;
            padding: 0.5em 1em;
            font-weight: 500;
        }
        
        /* File uploader */
        .uploadedFile {
            border: 1px solid rgba(55, 53, 47, 0.16);
            border-radius: 3px;
            padding: 1em;
            background: white;
        }
        
        /* Results cards */
        .result-card {
            border: 1px solid rgba(55, 53, 47, 0.16);
            border-radius: 3px;
            padding: 1em;
            margin-bottom: 1em;
            background: white;
        }
        
        /* Sidebar */
        .css-1d391kg {
            background: rgb(251, 251, 250);
            border-right: 1px solid rgba(55, 53, 47, 0.09);
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 1em;
            border-bottom: 1px solid rgba(55, 53, 47, 0.09);
        }
        
        .stTabs [data-baseweb="tab"] {
            padding: 0.5em 1em;
            color: rgb(55, 53, 47);
            font-weight: 500;
        }
        
        .stTabs [aria-selected="true"] {
            background: rgb(251, 251, 250);
            border-radius: 3px;
        }
        
        /* Progress bar */
        .stProgress > div > div > div {
            background: rgb(35, 131, 226);
        }
        
        /* Status indicators */
        .status {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 6px;
        }
        
        /* Info boxes */
        .info-box {
            background: rgb(251, 251, 250);
            border: 1px solid rgba(55, 53, 47, 0.09);
            border-radius: 3px;
            padding: 1em;
            margin: 1em 0;
        }
        
        /* Dividers */
        .divider {
            height: 1px;
            background: rgba(55, 53, 47, 0.09);
            margin: 2em 0;
        }
        
        /* Header container */
        .header-container {
            padding: 3rem 2rem 2rem 2rem;
            margin-bottom: 2rem;
            background: white;
            border-bottom: 1px solid rgba(55, 53, 47, 0.09);
        }
        
        /* Page title */
        .page-title {
            font-family: 'Inter', sans-serif;
            font-size: 2.5rem;
            font-weight: 700;
            color: rgb(55, 53, 47);
            margin-bottom: 0.5rem;
            line-height: 1.2;
        }
        
        /* Page subtitle */
        .page-subtitle {
            font-family: 'Inter', sans-serif;
            font-size: 1.1rem;
            color: rgba(55, 53, 47, 0.65);
            margin-top: 0.5rem;
            line-height: 1.5;
        }
        
        /* Content container */
        .content-container {
            padding: 0 2rem;
            max-width: 1200px;
            margin: 0 auto;
        }
        
        /* Streamlit container adjustments */
        .block-container {
            padding-top: 0 !important;
            margin-top: 0 !important;
        }
        
        .css-1544g2n {
            padding-top: 0 !important;
        }
        
        .css-1n76uvr {
            width: 100% !important;
        }
        </style>
    """, unsafe_allow_html=True)

def initialize_gemini(api_key: str) -> bool:
    """Initialize Gemini AI with the provided API key"""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        # Test the API key with a simple prompt
        response = model.generate_content("Test")
        return True
    except Exception as e:
        st.error(f"Error initializing Gemini AI: {str(e)}")
        return False

def extract_text_from_pdf(file):
    """Extract text content from PDF file"""
    output_string = StringIO()
    laparams = LAParams()
    extract_text_to_fp(file, output_string, laparams=laparams)
    return output_string.getvalue()

def preprocess_text(text):
    """Clean and normalize text"""
    # Convert to lowercase and remove special characters
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def calculate_keyword_similarity(cv_text, keywords):
    """
    Calculate the percentage of keywords found in the CV text
    Returns a value between 0 and 1 representing the match percentage
    """
    cv_text = cv_text.lower()
    matches = 0
    for keyword in keywords:
        # Check if the keyword or its variations exist in the CV
        if keyword.lower() in cv_text:
            matches += 1
    
    return matches / len(keywords) if keywords else 0

def get_matched_keywords(cv_text, keywords):
    """Return a list of keywords that were found in the CV"""
    cv_text = cv_text.lower()
    return [keyword for keyword in keywords if keyword.lower() in cv_text]

def analyze_cv_with_ai(cv_text: str, job_description: str) -> Dict:
    """
    Use Google Gemini to analyze CV suitability for the role
    """
    model = genai.GenerativeModel('gemini-pro')
    prompt = f"""
    You are an expert recruitment AI.
    Evaluate a candidate's CV against the provided job description to determine their suitability for the role. 
    Provide a  clear interview recommendation based on evidence extracted from the CV.
    Focus on assessing qualifications, skills, and relevant experiences outlined in the CV as they relate to the job description. 
    Provide specific examples to support your assessment and final recommendation.

    You must follow these scoring guidelines:
    - Suitability Score 80-100: Use "Strongly Recommend"
    - Suitability Score 60-79: Use "Recommend"
    - Suitability Score 40-59: Use "Consider"
    - Suitability Score 0-39: Use "Do Not Recommend"

    Provide your analysis in the following format:

    SUITABILITY_SCORE: [number between 0-100]
    
    STRENGTHS:
    - [strength 1]
    - [strength 2]
    - [strength 3]
    
    GAPS:
    - [gap 1]
    - [gap 2]
    - [gap 3]
    
    INTERVIEW_RECOMMENDATION: [Must match suitability score as per guidelines above]
    
    DETAILED_RECOMMENDATION:
    [2-3 sentence explanation of why you made this interview recommendation, including specific points from their CV]

    Here are the details to analyze:

    JOB DESCRIPTION:
    {job_description}

    CV CONTENT:
    {cv_text}
    """
    
    try:
        response = model.generate_content(prompt)
        response_text = response.text
        
        # Parse the response
        analysis = {}
        
        # Extract and validate suitability score
        score_match = re.search(r'SUITABILITY_SCORE:\s*(\d+)', response_text)
        if score_match:
            score = int(score_match.group(1))
            # Ensure score is within valid range
            score = max(0, min(100, score))
            analysis['suitability_score'] = score
            
            # Determine recommendation based on score
            if score >= 80:
                expected_recommendation = "Strongly Recommend"
            elif score >= 60:
                expected_recommendation = "Recommend"
            elif score >= 40:
                expected_recommendation = "Consider"
            else:
                expected_recommendation = "Do Not Recommend"
        else:
            score = 0
            expected_recommendation = "Do Not Recommend"
            analysis['suitability_score'] = score
        
        # Extract strengths
        strengths_section = re.search(r'STRENGTHS:(.*?)(?=GAPS:|$)', response_text, re.DOTALL)
        strengths = []
        if strengths_section:
            strengths = [s.strip('- ').strip() for s in strengths_section.group(1).strip().split('\n') if s.strip('- ').strip()]
        analysis['strengths'] = strengths[:3]  # Take up to 3 strengths
        
        # Extract gaps
        gaps_section = re.search(r'GAPS:(.*?)(?=INTERVIEW_RECOMMENDATION:|$)', response_text, re.DOTALL)
        gaps = []
        if gaps_section:
            gaps = [g.strip('- ').strip() for g in gaps_section.group(1).strip().split('\n') if g.strip('- ').strip()]
        analysis['gaps'] = gaps[:3]  # Take up to 3 gaps
        
        # Extract interview recommendation and ensure it matches the score
        interview_rec_match = re.search(r'INTERVIEW_RECOMMENDATION:\s*(.*?)(?=\n|$)', response_text)
        # Use the expected recommendation based on score
        analysis['interview_recommendation'] = expected_recommendation
        
        # Extract detailed recommendation
        detailed_rec_section = re.search(r'DETAILED_RECOMMENDATION:(.*?)$', response_text, re.DOTALL)
        analysis['detailed_recommendation'] = detailed_rec_section.group(1).strip() if detailed_rec_section else ""
        
        return analysis
        
    except Exception as e:
        st.error(f"Error in AI analysis: {str(e)}")
        return {
            "suitability_score": 0,
            "strengths": ["AI analysis failed"],
            "gaps": ["Unable to analyze"],
            "interview_recommendation": "Do Not Recommend",
            "detailed_recommendation": f"Error in AI analysis: {str(e)}"
        }

def calculate_composite_score(keyword_match_percentage: float, ai_score: int) -> float:
    """
    Calculate a composite score based on keyword matches and AI assessment
    Returns a score between 0 and 100
    
    Weights:
    - AI Assessment: 70%
    - Keyword Matching: 30%
    """
    ai_weight = 0.7
    keyword_weight = 0.3
    
    return (ai_score * ai_weight) + (keyword_match_percentage * 100 * keyword_weight)

def get_recommendation_from_score(composite_score: float) -> str:
    """
    Determine recommendation based on composite score
    """
    if composite_score >= 80:
        return "Strongly Recommend"
    elif composite_score >= 65:
        return "Recommend"
    elif composite_score >= 50:
        return "Consider"
    else:
        return "Do Not Recommend"

def process_cv(file, job_description):
    """Process individual CV file"""
    try:
        text = extract_text_from_pdf(file)
        
        # Get AI analysis
        ai_analysis = analyze_cv_with_ai(text, job_description)
        
        return {
            "Filename": file.name,
            "AI Score": ai_analysis["suitability_score"],
            "Recommendation": ai_analysis["interview_recommendation"],
            "Key Strengths": ai_analysis["strengths"],
            "Potential Gaps": ai_analysis["gaps"],
            "Detailed Analysis": ai_analysis["detailed_recommendation"]
        }
    except Exception as e:
        return {
            "Filename": file.name,
            "Error": str(e)
        }

def display_enhanced_results(results_df):
    if results_df.empty:
        st.warning("No results to display")
        return
    
    # Sort by AI Score
    sorted_df = results_df.sort_values('AI Score', ascending=False)
    
    # Summary Statistics
    st.markdown("<h3>Summary</h3>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Candidates", len(sorted_df))
    with col2:
        st.metric("Strongly Recommended", len(sorted_df[sorted_df['Recommendation'] == 'Strongly Recommend']))
    with col3:
        st.metric("Recommended", len(sorted_df[sorted_df['Recommendation'] == 'Recommend']))
    
    # Detailed Results
    st.markdown("<h3>Candidate Analysis</h3>", unsafe_allow_html=True)
    
    for _, row in sorted_df.iterrows():
        # Color coding for recommendations
        recommendation_colors = {
            "Strongly Recommend": "🟢",
            "Recommend": "🟡",
            "Consider": "🟠",
            "Do Not Recommend": "🔴"
        }
        rec_color = recommendation_colors.get(row['Recommendation'], "⚪")
        
        with st.expander(f"{rec_color} {row['Filename']} - AI Score: {row['AI Score']}"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"""
                    <div class='result-card'>
                        <h4>AI Assessment</h4>
                        <p>AI Score: {row['AI Score']}</p>
                        <p><strong>Recommendation: {row['Recommendation']}</strong></p>
                    </div>
                    
                    <div class='result-card'>
                        <h4>Detailed Analysis</h4>
                        <p>{row['Detailed Analysis']}</p>
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                    <div class='result-card'>
                        <h4>Key Strengths</h4>
                """, unsafe_allow_html=True)
                for strength in row['Key Strengths']:
                    st.markdown(f"• {strength}")
                st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown("""
                    <div class='result-card'>
                        <h4>Areas for Discussion</h4>
                """, unsafe_allow_html=True)
                for gap in row['Potential Gaps']:
                    st.markdown(f"• {gap}")
                st.markdown("</div>", unsafe_allow_html=True)

def main():
    set_custom_style()
    
    # Updated header with better spacing and structure
    st.markdown("""
        <div class="header-container">
            <div class="content-container">
                <div class="page-title">🌍 Aceli CV Analysis Tool</div>
                <div class="page-subtitle">
                    AI-powered candidate assessment
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar with Notion-like styling
    with st.sidebar:
        st.markdown("""
            <div class="block-container">
                <h3>⚙️ Settings</h3>
            </div>
        """, unsafe_allow_html=True)
        
        api_key = st.text_input(
            "Gemini API Key",
            type="password",
            help="Enter your Google Gemini API key",
            placeholder="Paste your API key here..."
        )
        
        if api_key:
            if initialize_gemini(api_key):
                st.markdown("""
                    <div class="info-box" style="background: rgb(221, 237, 234);">
                        <p style="color: rgb(68, 131, 97);">✓ API Connected</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div class="info-box" style="background: rgb(253, 230, 230);">
                        <p style="color: rgb(212, 76, 71);">✕ Invalid API Key</p>
                    </div>
                """, unsafe_allow_html=True)
                st.stop()
        else:
            st.markdown("""
                <div class="info-box" style="background: rgb(251, 251, 250);">
                    <p style="color: rgba(55, 53, 47, 0.65);">Please enter your API key to continue</p>
                </div>
            """, unsafe_allow_html=True)
            st.stop()
    
    # Main content with Notion-like tabs
    tabs = st.tabs(["📝 Input", "🔍 Analysis", "ℹ️ Help"])
    
    with tabs[0]:
        st.markdown("""
            <div class="block-container">
                <h2>Job Description</h2>
                <p style="color: rgba(55, 53, 47, 0.65);">Enter the complete job description below</p>
            </div>
        """, unsafe_allow_html=True)
        
        job_description = st.text_area(
            "",
            height=200,
            placeholder="Type or paste job description here..."
        )
        
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        st.markdown("""
            <div class="block-container">
                <h2>Upload CVs</h2>
                <p style="color: rgba(55, 53, 47, 0.65);">Select PDF files to analyze</p>
            </div>
        """, unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader(
            "",
            accept_multiple_files=True,
            type=['pdf']
        )
        
        if uploaded_files:
            st.markdown(f"""
                <div class="info-box">
                    <p>📎 {len(uploaded_files)} files ready for analysis</p>
                </div>
            """, unsafe_allow_html=True)
    
    with tabs[1]:
        if st.button("Start Analysis", type="primary"):
            if not all([uploaded_files, job_description]):
                st.error("Please provide both job description and CVs")
                return
            
            with st.spinner('Analyzing CVs with AI...'):
                results = []
                progress_bar = st.progress(0)
                
                # Process CVs with concurrent execution
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = [
                        executor.submit(process_cv, file, job_description) 
                        for file in uploaded_files
                    ]
                    for i, future in enumerate(concurrent.futures.as_completed(futures)):
                        result = future.result()
                        results.append(result)
                        progress_bar.progress((i + 1) / len(uploaded_files))
                
                # Create dataframe and display results
                df = pd.DataFrame([r for r in results if "Error" not in r])
                display_enhanced_results(df)
                
                # Display any errors
                errors = [r for r in results if "Error" in r]
                if errors:
                    st.error("Errors occurred while processing some files:")
                    for error in errors:
                        st.error(f"{error['Filename']}: {error['Error']}")
    
    with tabs[2]:
        st.markdown("""
            <h3>How to Use This Tool</h3>
            <div class='info-box'>
                <ol>
                    <li>Enter your Gemini API key in the sidebar</li>
                    <li>Paste the complete job description</li>
                    <li>Upload candidate CVs (PDF format)</li>
                    <li>Click "Start Analysis" to begin</li>
                </ol>
            </div>
            
            <h3>About the Analysis</h3>
            <div class='info-box'>
                <p>This tool uses AI to analyze CVs against the job description and provide interview recommendations.</p>
            </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
