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
        /* Main container styling */
        .main {
            padding: 2rem;
            max-width: 1200px;
            margin: 0 auto;
        }
        
        /* Headers styling */
        h1 {
            font-family: 'Inter', sans-serif;
            font-weight: 700;
            color: #37352f;
            margin-bottom: 1.5rem;
            font-size: 2.5rem !important;
        }
        
        h2, h3 {
            font-family: 'Inter', sans-serif;
            color: #37352f;
            font-weight: 600;
            margin-top: 2rem !important;
        }
        
        /* Text styling */
        p, li {
            font-family: 'Inter', sans-serif;
            color: #37352f;
            font-size: 1rem;
            line-height: 1.5;
        }
        
        /* Card-like containers */
        .stExpander {
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            margin-bottom: 1rem;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
        }
        
        /* Button styling */
        .stButton > button {
            background-color: #2ea44f;
            color: white;
            border: none;
            border-radius: 6px;
            padding: 0.5rem 1rem;
            font-weight: 500;
        }
        
        .stButton > button:hover {
            background-color: #2c974b;
        }
        
        /* Input field styling */
        .stTextInput > div > div > input,
        .stTextArea > div > div > textarea {
            border-radius: 6px;
            border: 1px solid #e0e0e0;
            padding: 0.5rem;
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            padding: 2rem 1rem;
        }
        
        /* Progress bar styling */
        .stProgress > div > div > div {
            background-color: #2ea44f;
        }
        
        /* Custom cards for results */
        .result-card {
            background-color: white;
            padding: 1.5rem;
            border-radius: 8px;
            border: 1px solid #e0e0e0;
            margin-bottom: 1rem;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
        }
        
        /* Status indicators */
        .status-indicator {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 8px;
        }
        
        /* Recommendation colors */
        .recommend-strong { background-color: #2ea44f; }
        .recommend-yes { background-color: #79b8ff; }
        .recommend-maybe { background-color: #ffab70; }
        .recommend-no { background-color: #f97583; }
        
        /* File upload area styling */
        .uploadedFile {
            border: 2px dashed #e0e0e0;
            border-radius: 8px;
            padding: 2rem;
            text-align: center;
            background-color: #fafafa;
        }
        
        /* Info boxes */
        .info-box {
            background-color: #f6f8fa;
            border: 1px solid #e0e0e0;
            border-radius: 6px;
            padding: 1rem;
            margin: 1rem 0;
        }
        
        /* Dataframe styling */
        .dataframe {
            border: none !important;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
        }
        
        .dataframe th {
            background-color: #f6f8fa;
            font-weight: 600;
        }
        
        /* Custom divider */
        .divider {
            height: 1px;
            background-color: #e0e0e0;
            margin: 2rem 0;
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
    As an expert recruitment AI, analyze this candidate's CV against the job description provided.
    Focus on determining their suitability for the role and provide a clear interview recommendation.

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

@st.cache_data
def process_cv(file, keywords, job_description):
    """Process individual CV file"""
    try:
        text = extract_text_from_pdf(file)
        processed_text = preprocess_text(text)
        match_percentage = calculate_keyword_similarity(processed_text, keywords)
        
        # Add AI analysis
        ai_analysis = analyze_cv_with_ai(text, job_description)
        
        return {
            "Filename": file.name,
            "Match Percentage": match_percentage,
            "Matched Keywords": get_matched_keywords(processed_text, keywords),
            "AI Suitability Score": ai_analysis["suitability_score"],
            "Key Strengths": ai_analysis["strengths"],
            "Potential Gaps": ai_analysis["gaps"],
            "interview_recommendation": ai_analysis["interview_recommendation"],
            "detailed_recommendation": ai_analysis["detailed_recommendation"],
            "Full Text": text
        }
    except Exception as e:
        return {
            "Filename": file.name,
            "Error": str(e)
        }

def process_cvs_with_progress(uploaded_files, keywords, job_description):
    """Process CVs with a progress bar"""
    results = []
    progress_bar = st.progress(0)
    
    # Process CVs with concurrent execution
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_cv, file, keywords, job_description) 
            for file in uploaded_files
        ]
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            result = future.result()
            results.append(result)
            progress_bar.progress((i + 1) / len(uploaded_files))
    
    return pd.DataFrame([r for r in results if "Error" not in r])

def display_enhanced_results(results_df, threshold):
    if results_df.empty:
        st.warning("No results to display")
        return
    
    # Filter and sort results
    filtered_df = results_df[results_df['Match Percentage'] >= threshold/100].copy()
    filtered_df['Match Percentage'] *= 100
    filtered_df = filtered_df.sort_values(['AI Suitability Score', 'Match Percentage'], 
                                        ascending=[False, False])
    
    if filtered_df.empty:
        st.warning(f"No candidates meet the {threshold}% threshold")
        return
    
    # Summary Statistics
    st.markdown("<h3>Summary</h3>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Candidates", len(filtered_df))
    with col2:
        st.metric("Avg Match %", f"{filtered_df['Match Percentage'].mean():.1f}%")
    with col3:
        st.metric("Avg AI Score", f"{filtered_df['AI Suitability Score'].mean():.1f}")
    
    # Detailed Results
    st.markdown("<h3>Candidate Analysis</h3>", unsafe_allow_html=True)
    
    for _, row in filtered_df.iterrows():
        with st.expander(f"📄 {row['Filename']}"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"""
                    <div class='result-card'>
                        <h4>Match Analysis</h4>
                        <p>Keyword Match: {row['Match Percentage']:.1f}%</p>
                        <p>AI Suitability: {row['AI Suitability Score']}</p>
                        <p>Recommendation: {row['interview_recommendation']}</p>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                    <div class='result-card'>
                        <h4>AI Insights</h4>
                        <p>{row['detailed_recommendation']}</p>
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                    <div class='result-card'>
                        <h4>Strengths</h4>
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
    
    # App Header
    st.markdown("""
        <div style='text-align: center; padding: 2rem 0;'>
            <h1>📄 CV Analysis Assistant</h1>
            <p style='font-size: 1.2rem; color: #666;'>
                Powered by AI to help you find the best candidates
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar Configuration
    with st.sidebar:
        st.markdown("""
            <div style='padding: 1rem 0;'>
                <h3 style='margin: 0;'>⚙️ Configuration</h3>
            </div>
        """, unsafe_allow_html=True)
        
        api_key = st.text_input(
            "Gemini API Key",
            type="password",
            help="Enter your Google Gemini API key"
        )
        
        if api_key:
            if initialize_gemini(api_key):
                st.success("✅ API Connected")
            else:
                st.error("❌ Invalid API Key")
                st.stop()
        else:
            st.warning("⚠️ API Key Required")
            st.stop()
    
    # Main Content Area
    tabs = st.tabs(["📝 Input", "🔍 Analysis", "ℹ️ Help"])
    
    with tabs[0]:
        st.markdown("<h3>Job Details</h3>", unsafe_allow_html=True)
        
        # Job Description
        st.markdown("""
            <div class='info-box'>
                <p style='margin: 0;'>📋 Enter the job description below</p>
            </div>
        """, unsafe_allow_html=True)
        
        job_description = st.text_area(
            "",
            height=200,
            placeholder="Paste the job description here..."
        )
        
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        
        # Keywords
        st.markdown("<h3>Keywords</h3>", unsafe_allow_html=True)
        keywords_input = st.text_area(
            "",
            placeholder="Enter keywords separated by commas...",
            help="These keywords will be used to match against CVs"
        )
        
        if keywords_input:
            keywords = [k.strip() for k in keywords_input.split(',') if k.strip()]
            st.markdown(f"""
                <div class='info-box'>
                    <p>🎯 {len(keywords)} keywords identified</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            keywords = []
        
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        
        # File Upload
        st.markdown("<h3>Upload CVs</h3>", unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "",
            accept_multiple_files=True,
            type=['pdf']
        )
        
        if uploaded_files:
            st.markdown(f"""
                <div class='info-box'>
                    <p>📎 {len(uploaded_files)} files uploaded</p>
                </div>
            """, unsafe_allow_html=True)
        
        # Analysis Settings
        st.markdown("<h3>Analysis Settings</h3>", unsafe_allow_html=True)
        match_threshold = st.slider(
            "Minimum Match Threshold",
            0, 100, 10,
            format="%d%%",
            help="Set the minimum keyword match percentage required"
        )
    
    with tabs[1]:
        if st.button("Start Analysis", type="primary"):
            if not all([keywords, uploaded_files, job_description]):
                st.error("Please fill in all required information first")
                return
            
            with st.spinner('Analyzing CVs...'):
                # Your existing analysis code here
                results = process_cvs_with_progress(uploaded_files, keywords, job_description)
                display_enhanced_results(results, match_threshold)
    
    with tabs[2]:
        st.markdown("""
            <h3>How to Use This Tool</h3>
            <div class='info-box'>
                <ol>
                    <li>Enter your Gemini API key in the sidebar</li>
                    <li>Paste the complete job description</li>
                    <li>Enter relevant keywords for the position</li>
                    <li>Upload candidate CVs (PDF format)</li>
                    <li>Set your minimum match threshold</li>
                    <li>Click "Start Analysis" to begin</li>
                </ol>
            </div>
            
            <h3>About the Analysis</h3>
            <div class='info-box'>
                <p>This tool combines keyword matching with AI analysis to:</p>
                <ul>
                    <li>Match CV content against your keywords</li>
                    <li>Evaluate candidate suitability</li>
                    <li>Identify key strengths and gaps</li>
                    <li>Provide interview recommendations</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
