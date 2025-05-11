import streamlit as st  
import pandas as pd
import re
import time
import json
from io import StringIO
from pdfminer.high_level import extract_text_to_fp
from pdfminer.layout import LAParams
import concurrent.futures
import openai
from typing import Dict, List

# Custom CSS to force light mode styling
def set_custom_style():
    st.markdown("""
        <style>
        /* Base styles */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        /* Force light mode */
        html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"], 
        [data-testid="stToolbar"], [data-testid="stSidebar"], [data-testid="stSidebarUserContent"],
        [data-testid="collapsedControl"], .main, .main-container {
            background-color: white !important;
            color: #0f172a !important;
        }
        
        .stButton > button:focus:not(:active) {
            border-color: #A9DBB8 !important;
            color: #1a1a1a !important;
        }
        
        /* Fix for dark text on inputs */
        textarea, input, .stTextInput > div > div > input, .stTextArea > div > div > textarea {
            color: #0f172a !important;
            background-color: white !important;
            border-color: #e2e8f0 !important;
        }
        
        /* Typography */
        * {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        h1, h2, h3, h4, h5, h6, p, li, label, [data-testid="stSidebarUserContent"] {
            color: #1a1a1a !important;
        }
        
        h1 {
            font-size: 34px;
            font-weight: 700;
            line-height: 1.2;
            margin-bottom: 16px;
            color: #0f172a !important;
        }
        
        h2 {
            font-size: 24px;
            font-weight: 600;
            margin: 24px 0 16px 0;
            color: #1e293b !important;
        }
        
        h3 {
            font-size: 20px;
            font-weight: 600;
            margin: 20px 0 12px 0;
            color: #334155 !important;
        }
        
        p, li, span, label, .stMarkdown, .stMarkdown * {
            font-size: 16px;
            line-height: 1.5;
            color: #475569 !important;
        }
        
        /* Fix specifically for checkbox labels */
        [data-testid="stCheckbox"] label p {
            color: #1a1a1a !important;
        }
        
        /* Metrics and other components */
        [data-testid="stMetricValue"] {
            color: #1a1a1a !important;
            background-color: transparent !important;
        }
        
        [data-testid="stMetricLabel"] {
            color: #64748b !important;
        }
        
        /* Input fields */
        .stTextInput > div > div > input,
        .stTextArea > div > div > textarea {
            border: 1px solid #e2e8f0 !important;
            border-radius: 6px !important;
            padding: 10px !important;
            font-size: 16px !important;
            background: white !important;
            color: #1a1a1a !important;
        }
        
        /* Buttons */
        .stButton > button {
            background-color: #A9DBB8 !important;
            color: #1a1a1a !important;
            border: none !important;
            border-radius: 4px !important;
            padding: 10px 20px !important;
            font-weight: 500 !important;
            font-size: 16px !important;
            cursor: pointer !important;
        }
        
        .stButton > button:hover {
            background-color: #98c9a7 !important;
            color: #1a1a1a !important;
        }
        
        /* File uploader */
        .uploadedFile {
            border: 1px solid #e2e8f0 !important;
            border-radius: 6px !important;
            padding: 16px !important;
            background: white !important;
            color: #1a1a1a !important;
        }
        
        /* File uploader text */
        [data-testid="stFileUploader"] label span {
            color: #1a1a1a !important;
        }
        
        /* File uploader drop area */
        [data-testid="stFileUploadDropzone"] {
            background-color: #f8fafc !important;
            border-color: #e2e8f0 !important;
            color: #1a1a1a !important;
        }
        
        /* Results cards */
        .result-card {
            border: 1px solid #e2e8f0 !important;
            border-radius: 8px !important;
            padding: 20px !important;
            margin-bottom: 16px !important;
            background: white !important;
            color: #1a1a1a !important;
        }
        
        /* Expander */
        [data-testid="stExpander"] {
            border: 1px solid #e2e8f0 !important;
            background-color: white !important;
        }
        
        /* Expander header */
        [data-testid="stExpander"] details summary p {
            color: #1a1a1a !important;
        }
        
        /* Sidebar */
        [data-testid="stSidebar"] {
            background-color: #f8fafc !important;
            border-right: 1px solid #e2e8f0 !important;
        }
        
        [data-testid="stSidebar"] [data-testid="stMarkdown"] p {
            color: #1a1a1a !important;
        }
        
        /* Make sure all text and headers in sidebar are visible */
        [data-testid="stSidebar"] div, [data-testid="stSidebar"] p, 
        [data-testid="stSidebar"] span, [data-testid="stSidebar"] label, 
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, 
        [data-testid="stSidebar"] h3 {
            color: #1a1a1a !important;
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
            border-bottom: 1px solid #e2e8f0 !important;
            padding: 0 4px;
            background-color: white !important;
        }
        
        .stTabs [data-baseweb="tab"] {
            padding: 10px 16px;
            color: #64748b !important;
            font-weight: 500;
            font-size: 15px;
            border-radius: 6px 6px 0 0;
            background-color: white !important;
        }
        
        .stTabs [aria-selected="true"] {
            color: #0f172a !important;
            font-weight: 600;
            background-color: white !important;
            border-bottom: 2px solid #0f172a !important;
        }
        
        /* Tab content area */
        .stTabs [data-baseweb="tab-panel"] {
            background-color: white !important;
        }
        
        /* Progress bar */
        .stProgress > div > div > div {
            background-color: #0f172a !important;
        }
        
        /* Info boxes */
        .info-box {
            background: #f8fafc !important;
            border: 1px solid #e2e8f0 !important;
            border-radius: 6px !important;
            padding: 16px !important;
            margin: 16px 0 !important;
            color: #1a1a1a !important;
        }
        
        /* Info/warning/error/success boxes */
        [data-testid="stInfo"], [data-testid="stWarning"], 
        [data-testid="stError"], [data-testid="stSuccess"] {
            color: #1a1a1a !important;
        }
        
        /* Dividers */
        .divider {
            height: 1px;
            background: #e2e8f0 !important;
            margin: 24px 0;
        }
        
        /* Columns & container background fixes */
        [data-testid="column"] {
            background-color: white !important;
        }
        
        [data-testid="stVerticalBlock"] {
            background-color: white !important;
        }
        
        /* Header container */
        .header-container {
            padding: 48px 32px 40px 32px;
            background: white !important;
            border-bottom: 1px solid #e2e8f0 !important;
            margin-bottom: 32px;
        }
        
        /* Content container */
        .content-container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 32px;
            background: white !important;
        }
        
        /* Header typography */
        .header-container h1 {
            font-size: 40px;
            font-weight: 700;
            line-height: 1.2;
            color: #0f172a !important;
            margin-bottom: 8px;
            letter-spacing: -0.02em;
        }
        
        .header-container p {
            font-size: 20px;
            line-height: 1.4;
            color: #64748b !important;
            margin-top: 8px;
            font-weight: 400;
        }
        
        /* Status indicators */
        .recommendation-status {
            font-weight: 500;
            margin-right: 8px;
            padding: 4px 8px;
            border-radius: 4px;
        }
        
        .status-strongly-recommend {
            background: #dcfce7 !important;
            color: #166534 !important;
        }
        
        .status-recommend {
            background: #fef9c3 !important;
            color: #854d0e !important;
        }
        
        .status-consider {
            background: #ffedd5 !important;
            color: #9a3412 !important;
        }
        
        .status-do-not-recommend {
            background: #fee2e2 !important;
            color: #991b1b !important;
        }
        
        /* Section headers */
        .section-header {
            font-size: 18px;
            font-weight: 600;
            color: #0f172a !important;
            margin: 24px 0 16px 0;
            padding-bottom: 8px;
            border-bottom: 1px solid #e2e8f0 !important;
        }
        
        /* Select box */
        [data-testid="stSelectbox"] {
            color: #1a1a1a !important;
        }
        
        [data-testid="stSelectbox"] > div > div {
            background-color: white !important;
            color: #1a1a1a !important;
        }
        
        /* Fix dark background in select dropdown */
        div[data-baseweb="select"] > div,
        div[data-baseweb="select"] span,
        div[data-baseweb="select"] input::placeholder {
            color: #1a1a1a !important;
            background-color: white !important;
        }
        
        /* Fix metric colors */
        [data-testid="stMetricValue"] > div,
        [data-testid="stMetricValue"] > div > div,
        [data-testid="stMetricLabel"] > div,
        [data-testid="stMetricLabel"] > div > div {
            color: #1a1a1a !important;
        }
        
        /* Fix error/success/info/warning messages */
        .stAlert {
            background-color: white !important;
            color: #1a1a1a !important;
        }
        
        .stAlert a {
            color: #2563eb !important;
        }
        </style>
    """, unsafe_allow_html=True)

def initialize_openai(api_key: str) -> bool:
    """Initialize OpenAI with the provided API key, only if changed."""
    try:
        # Only re-initialize if the key is new
        if (
            'openai_client' in st.session_state and
            'openai_api_key' in st.session_state and
            st.session_state.openai_api_key == api_key
        ):
            return True  # Already initialized with this key
        # Configure the client
        client = openai.OpenAI(api_key=api_key)
        # Test with a minimal request
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"}
            ],
            max_tokens=10
        )
        # If we get here without exception, the API key is valid
        st.session_state.openai_client = client
        st.session_state.openai_api_key = api_key
        st.session_state.openai_connected = True
        st.session_state.openai_error = None
        return True
    except Exception as e:
        st.session_state.openai_client = None
        st.session_state.openai_connected = False
        st.session_state.openai_error = str(e)
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

def analyze_cv_with_openai(cv_text: str, job_description: str) -> Dict:
    """
    Use OpenAI to analyze CV suitability for the role
    """
    try:
        # Check for OpenAI client
        if 'openai_client' not in st.session_state or st.session_state.openai_client is None:
            return {
                "suitability_score": 0,
                "strengths": ["Unable to analyze CV - OpenAI client not initialized"],
                "gaps": ["Please check your API key and connection."],
                "interview_recommendation": "Do Not Recommend",
                "detailed_recommendation": "OpenAI client is not initialized. Please enter a valid API key and connect."
            }
        client = st.session_state.openai_client
        # Truncate inputs if needed to reduce token consumption
        if st.session_state.get('truncate_input', True):
            max_cv_length = 5000  # characters
            max_job_desc_length = 2000  # characters
            if len(cv_text) > max_cv_length:
                cv_text = cv_text[:max_cv_length] + "... [truncated to save tokens]"
            if len(job_description) > max_job_desc_length:
                job_description = job_description[:max_job_desc_length] + "... [truncated to save tokens]"
        # Choose prompt based on simplify setting
        if st.session_state.get('simplify_prompt', True):
            prompt = f"""
            As a recruitment expert, assess this CV against the job requirements.
            Score the match from 0-100, and list key strengths and gaps.
            Format your response as a JSON object with these keys:
            - suitability_score: number between 0-100
            - strengths: array of 3 strings
            - gaps: array of 3 strings
            - interview_recommendation: string, one of "Strongly Recommend" (80-100), "Recommend" (60-79), "Consider" (40-59), or "Do Not Recommend" (0-39)
            - detailed_recommendation: string explaining rationale in 2-3 sentences
            JOB DESCRIPTION:
            {job_description}
            CV:
            {cv_text}
            """
        else:
            prompt = f"""
            You are an expert recruitment AI. 
            Your task is to assess a candidate's Curriculum Vitae (CV) against a provided job description to determine their suitability for the role.
            Steps for Evaluation:
            1. Analyze the job description to identify key qualifications, skills, and experiences required
            2. Review the CV to identify matching qualifications, skills, and experiences
            3. Identify potential strengths and gaps
            4. Provide a clear recommendation based on the candidate's suitability
            Format your response as a JSON object with these keys:
            - suitability_score: number between 0-100 where:
              * 80-100: Strong match with requirements, highly qualified
              * 60-79: Good match, meets key requirements
              * 40-59: Partial match, some gaps in requirements
              * 0-39: Poor match, significant gaps or missing requirements
            - strengths: array of 3 strings, each describing a key strength with specific evidence from CV
            - gaps: array of 3 strings, each describing a gap with explanation
            - interview_recommendation: string, one of "Strongly Recommend" (80-100), "Recommend" (60-79), "Consider" (40-59), or "Do Not Recommend" (0-39)
            - detailed_recommendation: string explaining recommendation in 2-3 sentences with specific evidence
            JOB DESCRIPTION:
            {job_description}
            CV:
            {cv_text}
            """
        model = st.session_state.get('selected_model', 'gpt-3.5-turbo')
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert recruitment assistant that analyzes CVs."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        response_text = response.choices[0].message.content
        analysis = json.loads(response_text)
        # Ensure score is within range
        if 'suitability_score' in analysis:
            analysis['suitability_score'] = max(0, min(100, analysis['suitability_score']))
        else:
            analysis['suitability_score'] = 0
        # Ensure we have all required keys with defaults
        if 'strengths' not in analysis or not analysis['strengths']:
            analysis['strengths'] = ["No clear strengths identified"]
        if 'gaps' not in analysis or not analysis['gaps']:
            analysis['gaps'] = ["Unable to identify specific gaps"]
        if 'interview_recommendation' not in analysis:
            score = analysis['suitability_score']
            if score >= 80:
                analysis['interview_recommendation'] = "Strongly Recommend"
            elif score >= 60:
                analysis['interview_recommendation'] = "Recommend"
            elif score >= 40:
                analysis['interview_recommendation'] = "Consider"
            else:
                analysis['interview_recommendation'] = "Do Not Recommend"
        if 'detailed_recommendation' not in analysis:
            analysis['detailed_recommendation'] = "No detailed analysis provided"
        if len(analysis['strengths']) > 3:
            analysis['strengths'] = analysis['strengths'][:3]
        if len(analysis['gaps']) > 3:
            analysis['gaps'] = analysis['gaps'][:3]
        return analysis
    except Exception as e:
        error_message = str(e)
        if "rate limit" in error_message.lower():
            error_message = "Rate limit exceeded. Please try again later or check your OpenAI plan."
        return {
            "suitability_score": 0,
            "strengths": ["Unable to analyze CV - API error"],
            "gaps": ["Unable to analyze CV - Please try again later"],
            "interview_recommendation": "Do Not Recommend",
            "detailed_recommendation": f"Error in analysis: {error_message}"
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
        # Extract text from PDF
        try:
            text = extract_text_from_pdf(file)
        except Exception as e:
            return {
                "Filename": getattr(file, 'name', 'Unknown'),
                "Error": f"PDF extraction error: {str(e)}"
            }
        # Check if we need to truncate the CV text to save tokens
        if st.session_state.get('truncate_input', True) and len(text) > 5000:
            text = text[:5000] + "... [truncated to save tokens]"
        # Get AI analysis
        ai_analysis = analyze_cv_with_openai(text, job_description)
        if 'Error' in ai_analysis:
            return {
                "Filename": getattr(file, 'name', 'Unknown'),
                "Error": ai_analysis['Error']
            }
        return {
            "Filename": getattr(file, 'name', 'Unknown'),
            "AI Score": ai_analysis["suitability_score"],
            "Recommendation": ai_analysis["interview_recommendation"],
            "Key Strengths": ai_analysis["strengths"],
            "Potential Gaps": ai_analysis["gaps"],
            "Detailed Analysis": ai_analysis["detailed_recommendation"]
        }
    except Exception as e:
        error_message = str(e)
        if "rate limit" in error_message.lower():
            error_message = "Rate limit exceeded. Try processing fewer CVs or waiting."
        return {
            "Filename": getattr(file, 'name', 'Unknown'),
            "Error": error_message
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
    # Initialize session state
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = 'gpt-3.5-turbo'
    if 'openai_connected' not in st.session_state:
        st.session_state.openai_connected = False
    if 'openai_error' not in st.session_state:
        st.session_state.openai_error = None
    if 'openai_api_key' not in st.session_state:
        st.session_state.openai_api_key = ''
    # Force light mode
    st.set_page_config(
        page_title="Aceli CV Analysis Tool",
        page_icon="🌍",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': None,
            'Report a bug': None,
            'About': None
        }
    )
    set_custom_style()
    st.markdown("""
        <div class="header-container">
            <div class="content-container">
                <h1 style="font-size: 40px; margin-bottom: 8px;">🌍 Aceli CV Analysis Tool</h1>
                <p style="font-size: 20px; color: #64748b; margin-top: 8px;">
                    AI-powered candidate assessment
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    with st.sidebar:
        st.markdown("""
            <div style="margin-bottom: 20px;">
                <h3 style="color: #1a1a1a !important;">⚙️ Settings</h3>
            </div>
        """, unsafe_allow_html=True)
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Enter your OpenAI API key",
            placeholder="Paste your API key here...",
            value=st.session_state.openai_api_key
        )
        debug_mode = st.checkbox("Debug Mode", value=False, help="Show detailed error messages")
        st.markdown("<p style='color: #1a1a1a !important;'>Model Selection:</p>", unsafe_allow_html=True)
        model_options = [
            "gpt-3.5-turbo",
            "gpt-4",
            "gpt-4-turbo",
        ]
        selected_model = st.selectbox(
            "OpenAI Model",
            options=model_options,
            index=model_options.index(st.session_state.selected_model) if st.session_state.selected_model in model_options else 0,
            help="Select a model (gpt-3.5-turbo is faster and cheaper)"
        )
        st.session_state.selected_model = selected_model
        st.markdown("<p style='color: #1a1a1a !important;'>Token Management:</p>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            truncate_input = st.checkbox("Truncate Inputs", value=st.session_state.get('truncate_input', True), help="Reduce input length to save tokens")
        with col2:
            simplify_prompt = st.checkbox("Simplify Prompts", value=st.session_state.get('simplify_prompt', True), help="Use simpler prompts that use fewer tokens")
        st.session_state.truncate_input = truncate_input
        st.session_state.simplify_prompt = simplify_prompt
        # Only try to connect if API key is provided and changed
        if api_key:
            if api_key != st.session_state.openai_api_key or not st.session_state.openai_connected:
                with st.spinner('Connecting to OpenAI API...'):
                    api_connected = initialize_openai(api_key)
            else:
                api_connected = st.session_state.openai_connected
            if api_connected:
                st.markdown(f"""
                    <div class="info-box" style="background: rgb(221, 237, 234);">
                        <p style="color: rgb(68, 131, 97);">✓ API Connected using model: {st.session_state.selected_model}</p>
                    </div>
                """, unsafe_allow_html=True)
                model_costs = {
                    "gpt-3.5-turbo": "$0.0015 / 1K input tokens, $0.002 / 1K output tokens",
                    "gpt-4": "$0.03 / 1K input tokens, $0.06 / 1K output tokens",
                    "gpt-4-turbo": "$0.01 / 1K input tokens, $0.03 / 1K output tokens"
                }
                st.markdown(f"""
                    <div class="info-box" style="background: rgb(255, 249, 219);">
                        <p style="color: rgb(151, 90, 22);"><strong>Cost Information:</strong></p>
                        <p style="color: rgb(151, 90, 22); font-size: 14px;">Current model: {st.session_state.selected_model}</p>
                        <p style="color: rgb(151, 90, 22); font-size: 14px;">Pricing: {model_costs.get(st.session_state.selected_model, "Price varies")}</p>
                        <p style="color: rgb(151, 90, 22); font-size: 14px;">Analyze fewer CVs at once to control costs</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                error_msg = st.session_state.openai_error or "Invalid API Key or connection issue"
                if debug_mode:
                    st.markdown(f"<div class=\"info-box\" style=\"background: rgb(253, 230, 230);\"><p style=\"color: rgb(212, 76, 71);\">✕ {error_msg}</p></div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class=\"info-box\" style=\"background: rgb(253, 230, 230);\"><p style=\"color: rgb(212, 76, 71);\">✕ {error_msg}</p></div>", unsafe_allow_html=True)
                    st.stop()
        else:
            st.markdown("""
                <div class="info-box" style="background: rgb(251, 251, 250);">
                    <p style="color: #1a1a1a;">Please enter your OpenAI API key to continue</p>
                </div>
            """, unsafe_allow_html=True)
            st.stop()
    tabs = st.tabs(["📝 Input", "🔍 Analysis", "ℹ️ Help"])
    with tabs[0]:
        st.markdown("""
            <div class="block-container">
                <h2 style="color: #1a1a1a !important;">Job Description</h2>
                <p style="color: #475569 !important;">Enter the complete job description below</p>
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
                <h2 style="color: #1a1a1a !important;">Upload CVs</h2>
                <p style="color: #475569 !important;">Select PDF files to analyze</p>
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
                    <p style="color: #1a1a1a !important;">📎 {len(uploaded_files)} files ready for analysis</p>
                </div>
            """, unsafe_allow_html=True)
    with tabs[1]:
        # Only enable button if all requirements are met
        can_analyze = (
            st.session_state.openai_connected and
            uploaded_files and
            job_description and
            isinstance(uploaded_files, list) and
            len(uploaded_files) > 0 and
            isinstance(job_description, str) and
            len(job_description.strip()) > 0
        )
        if not can_analyze:
            st.button("Start Analysis", type="primary", disabled=True, help="Please provide API key, job description, and at least one CV PDF.")
        else:
            if st.button("Start Analysis", type="primary"):
                with st.spinner('Analyzing CVs with AI...'):
                    results = []
                    progress_bar = st.progress(0)
                    max_workers = min(2, len(uploaded_files)) if uploaded_files else 1
                    files_to_process = uploaded_files.copy()
                    batch_size = 2
                    for i in range(0, len(files_to_process), batch_size):
                        batch = files_to_process[i:i+batch_size]
                        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                            futures = [
                                executor.submit(process_cv, file, job_description) 
                                for file in batch
                            ]
                            for j, future in enumerate(concurrent.futures.as_completed(futures)):
                                try:
                                    result = future.result()
                                    results.append(result)
                                except Exception as e:
                                    if debug_mode:
                                        st.error(f"Error processing file: {str(e)}")
                                    results.append({
                                        "Filename": f"File {i+j+1}",
                                        "Error": str(e)
                                    })
                                progress = (i + j + 1) / len(files_to_process)
                                progress_bar.progress(progress)
                        if i + batch_size < len(files_to_process):
                            with st.spinner(f'Processed {i+batch_size}/{len(files_to_process)} CVs. Please wait...'):
                                time.sleep(1)
                    st.success("✅ Analysis complete!")
                    df = pd.DataFrame([r for r in results if "Error" not in r])
                    if not df.empty:
                        display_enhanced_results(df)
                    errors = [r for r in results if "Error" in r]
                    if errors:
                        st.error("Errors occurred while processing some files:")
                        for error in errors:
                            st.error(f"{error['Filename']}: {error['Error']}")
    with tabs[2]:
        st.markdown("""
            <h3 style="color: #1a1a1a !important;">How to Use This Tool</h3>
            <div class='info-box'>
                <ol>
                    <li>Enter your OpenAI API key in the sidebar</li>
                    <li>Paste the complete job description</li>
                    <li>Upload candidate CVs (PDF format)</li>
                    <li>Click "Start Analysis" to begin</li>
                </ol>
            </div>
            <h3 style="color: #1a1a1a !important;">About the Analysis</h3>
            <div class='info-box'>
                <p>This tool uses OpenAI's language models to analyze CVs against the job description and provide interview recommendations.</p>
                <p>The analysis includes a suitability score, key strengths and potential gaps, and a detailed recommendation.</p>
            </div>
            <h3 style="color: #1a1a1a !important;">API Key Information</h3>
            <div class='info-box'>
                <p><strong>OpenAI API Keys:</strong></p>
                <ol>
                    <li>Get an API key from <a href="https://platform.openai.com/account/api-keys" target="_blank">OpenAI's platform</a></li>
                    <li>You need a paid account with OpenAI to use this tool</li>
                    <li>API usage is billed based on your OpenAI plan</li>
                    <li>Your API key is not stored and is only used for this session</li>
                </ol>
            </div>
            <h3 style="color: #1a1a1a !important;">Troubleshooting</h3>
            <div class='info-box'>
                <p><strong>API Key Issues:</strong></p>
                <ul>
                    <li>Ensure your OpenAI API key is active and has proper permissions</li>
                    <li>Check that you have billing enabled on your OpenAI account</li>
                    <li>If you hit rate limits, try selecting a different model or processing fewer CVs</li>
                    <li>Enable the debug mode in the sidebar for more detailed error messages</li>
                </ul>
                <p><strong>PDF Processing Issues:</strong></p>
                <ul>
                    <li>Ensure PDFs are not password-protected</li>
                    <li>Try with text-based PDFs rather than scanned documents</li>
                    <li>Large files may take longer to process</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
if __name__ == "__main__":
    main()
