import streamlit as st
import pandas as pd
import re
from io import StringIO
from pdfminer.high_level import extract_text_to_fp
from pdfminer.layout import LAParams
import concurrent.futures
import google.generativeai as genai
from typing import Dict, List

# Custom CSS for light mode styling
def set_custom_style():
    st.markdown("""
        <style>
        /* Base styles */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        * {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        /* Typography */
        h1, h2, h3, p {
            color: #1a1a1a;
        }
        
        h1 {
            font-size: 34px;
            font-weight: 700;
            line-height: 1.2;
            margin-bottom: 16px;
            color: #0f172a;
        }
        
        h2 {
            font-size: 24px;
            font-weight: 600;
            margin: 24px 0 16px 0;
            color: #1e293b;
        }
        
        h3 {
            font-size: 20px;
            font-weight: 600;
            margin: 20px 0 12px 0;
            color: #334155;
        }
        
        p {
            font-size: 16px;
            line-height: 1.5;
            margin-bottom: 8px;
            color: #475569;
        }
        
        /* Input fields */
        .stTextInput > div > div > input,
        .stTextArea > div > div > textarea {
            border: 1px solid #e2e8f0;
            border-radius: 6px;
            padding: 10px;
            font-size: 16px;
            background: white;
            color: #0f172a;
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
            color: #1a1a1a !important;
            background-color: #98c9a7 !important;
        }
        
        /* File uploader */
        .uploadedFile {
            border: 1px solid #e2e8f0;
            border-radius: 6px;
            padding: 16px;
            background: white;
            color: #1a1a1a;
        }
        
        /* Results cards */
        .result-card {
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 16px;
            background: white;
            color: #1a1a1a;
        }
        
        /* Sidebar */
        .css-1d391kg, .css-1cypcdb, [data-testid="stSidebar"] {
            background: #f8fafc;
            border-right: 1px solid #e2e8f0;
        }
        
        /* Make sure all text in sidebar is visible */
        [data-testid="stSidebar"] p, [data-testid="stSidebar"] h1, 
        [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] label {
            color: #1a1a1a !important;
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
            border-bottom: 1px solid #e2e8f0;
            padding: 0 4px;
        }
        
        .stTabs [data-baseweb="tab"] {
            padding: 10px 16px;
            color: #64748b;
            font-weight: 500;
            font-size: 15px;
            border-radius: 6px 6px 0 0;
        }
        
        .stTabs [aria-selected="true"] {
            color: #0f172a;
            font-weight: 600;
            background: white;
            border-bottom: 2px solid #0f172a;
        }
        
        /* Progress bar */
        .stProgress > div > div > div {
            background: #0f172a;
        }
        
        /* Info boxes */
        .info-box {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 6px;
            padding: 16px;
            margin: 16px 0;
            color: #1a1a1a;
        }
        
        /* Dividers */
        .divider {
            height: 1px;
            background: #e2e8f0;
            margin: 24px 0;
        }
        
        /* Header container */
        .header-container {
            padding: 48px 32px 40px 32px;
            background: white;
            border-bottom: 1px solid #e2e8f0;
            margin-bottom: 32px;
        }
        
        /* Content container */
        .content-container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 32px;
        }
        
        /* Header typography */
        .header-container h1 {
            font-size: 40px;
            font-weight: 700;
            line-height: 1.2;
            color: #0f172a;
            margin-bottom: 8px;
            letter-spacing: -0.02em;
        }
        
        .header-container p {
            font-size: 20px;
            line-height: 1.4;
            color: #64748b;
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
            background: #dcfce7;
            color: #166534;
        }
        
        .status-recommend {
            background: #fef9c3;
            color: #854d0e;
        }
        
        .status-consider {
            background: #ffedd5;
            color: #9a3412;
        }
        
        .status-do-not-recommend {
            background: #fee2e2;
            color: #991b1b;
        }
        
        /* Section headers */
        .section-header {
            font-size: 18px;
            font-weight: 600;
            color: #0f172a;
            margin: 24px 0 16px 0;
            padding-bottom: 8px;
            border-bottom: 1px solid #e2e8f0;
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
        
        /* Force light mode for all Streamlit elements */
        .st-emotion-cache-ue6h4q, .st-emotion-cache-ue6h4q *, 
        .st-emotion-cache-16txtl3, .st-emotion-cache-16txtl3 *,
        .st-emotion-cache-1wrcr25, .stApp, .stApp * {
            color-scheme: light !important;
        }
        
        /* Ensure all text inputs have dark text on light background */
        input, textarea, .stTextInput input, .stTextArea textarea {
            color: #1a1a1a !important;
            background-color: white !important;
        }
        
        /* Override any theme variables that might cause dark mode */
        :root {
            --background-color: white;
            --text-color: #1a1a1a;
            --secondary-background-color: #f8fafc;
        }
        </style>
    """, unsafe_allow_html=True)

def initialize_gemini(api_key: str) -> bool:
    """Initialize Gemini AI with the provided API key"""
    try:
        # Configure the API key
        genai.configure(api_key=api_key)
        
        # Create a model instance
        model = genai.GenerativeModel('gemini-pro')
        
        # Test with a simple prompt
        response = model.generate_content("Hello")
        
        # Check if response was successful
        if response and hasattr(response, 'text'):
            return True
        return False
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
    try:
        model = genai.GenerativeModel('gemini-pro')
        prompt = f"""
        You are an expert recruitment AI. 
        Your task is to assess a candidate's Curriculum Vitae (CV) against a provided job description to determine their suitability for the role.
        Steps for Evaluation:
        Job Description Analysis
        Analyze the job description to identify and extract key qualifications (e.g., education, certifications), essential skills (technical and soft skills), and crucial experiences required for the role.
        Prioritize these elements based on their importance for success in the position.
        CV Assessment
        Review the CV to identify qualifications, skills, and experiences that align with the job requirements.
        Quantify accomplishments whenever possible, focusing on metrics, results, or specific impacts (e.g., "increased social media engagement by 20% within three months" instead of "managed social media").
        Identify potential red flags, such as lack of required experience, career history inconsistencies, or skills gaps relative to the job description.
        Comparative Analysis
        Compare the job requirements with the CV findings to assess the candidate's suitability.
        Evaluate the depth of experience by determining whether the candidate has demonstrated mastery of skills through relevant projects and achievements.
        Evidence-Based Recommendation
        Provide a clear recommendation on whether the candidate should proceed to an interview, supported by specific evidence from the CV.
        If recommending an interview, highlight strengths and suggest areas to explore key skills further.
        If rejecting the candidate, clearly explain the decision based on specific gaps in qualifications, skills, or experience critical to the role.

        Provide your analysis in the following strict format:

        SUITABILITY_SCORE: [Score between 0-100, where:
        - 80-100: Strong match with requirements, highly qualified
        - 60-79: Good match, meets key requirements
        - 40-59: Partial match, some gaps in requirements
        - 0-39: Poor match, significant gaps or missing requirements]
        
        STRENGTHS:
        - [Key strength 1 with specific evidence from CV]
        - [Key strength 2 with specific evidence from CV]
        - [Key strength 3 with specific evidence from CV]
        
        GAPS:
        - [Gap 1 with explanation]
        - [Gap 2 with explanation]
        - [Gap 3 with explanation]
        
        RECOMMENDATION: [Must be one of:
        - "Strongly Recommend" (for scores 80-100)
        - "Recommend" (for scores 60-79)
        - "Consider" (for scores 40-59)
        - "Do Not Recommend" (for scores 0-39)]
        
        DETAILED_ANALYSIS:
        [Provide 2-3 sentences explaining the recommendation, citing specific evidence from the CV and how it relates to the job requirements]

        Here are the details to analyze:

        JOB DESCRIPTION:
        {job_description}

        CV CONTENT:
        {cv_text}
        """
        
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
        analysis['strengths'] = strengths[:3] if strengths else ["No clear strengths identified"]
        
        # Extract gaps
        gaps_section = re.search(r'GAPS:(.*?)(?=RECOMMENDATION:|$)', response_text, re.DOTALL)
        gaps = []
        if gaps_section:
            gaps = [g.strip('- ').strip() for g in gaps_section.group(1).strip().split('\n') if g.strip('- ').strip()]
        analysis['gaps'] = gaps[:3] if gaps else ["Unable to identify specific gaps"]
        
        # Extract detailed analysis
        detailed_section = re.search(r'DETAILED_ANALYSIS:(.*?)$', response_text, re.DOTALL)
        analysis['detailed_recommendation'] = detailed_section.group(1).strip() if detailed_section else "No detailed analysis provided"
        
        # Force recommendation to match score
        analysis['interview_recommendation'] = expected_recommendation
        
        return analysis
        
    except Exception as e:
        st.error(f"Error in AI analysis: {str(e)}")
        return {
            "suitability_score": 0,
            "strengths": ["Unable to analyze CV"],
            "gaps": ["Unable to analyze CV"],
            "interview_recommendation": "Do Not Recommend",
            "detailed_recommendation": f"Error in analysis: {str(e)}"
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
    # Force light mode
    st.set_page_config(
        page_title="Aceli CV Analysis Tool",
        page_icon="🌍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply custom styling
    set_custom_style()
    
    # Updated header with proper sizing
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
    
    # Add debugging option for API connection
    debug_mode = False
    
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
        
        # Add debug checkbox
        debug_mode = st.checkbox("Debug Mode", value=False, help="Show detailed error messages")
        
        if api_key:
            api_connected = initialize_gemini(api_key)
            if api_connected:
                st.markdown("""
                    <div class="info-box" style="background: rgb(221, 237, 234);">
                        <p style="color: rgb(68, 131, 97);">✓ API Connected</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                error_msg = "Invalid API Key or connection issue"
                if debug_mode:
                    try:
                        # Test with specific error handling
                        genai.configure(api_key=api_key)
                        model = genai.GenerativeModel('gemini-pro')
                        test_response = model.generate_content("Test connection")
                        error_msg = f"API responded but validation failed: {str(test_response)}"
                    except Exception as e:
                        error_msg = f"API Error: {str(e)}"
                
                st.markdown(f"""
                    <div class="info-box" style="background: rgb(253, 230, 230);">
                        <p style="color: rgb(212, 76, 71);">✕ {error_msg}</p>
                    </div>
                """, unsafe_allow_html=True)
                
                # If in debug mode, don't stop the application
                if not debug_mode:
                    st.stop()
        else:
            st.markdown("""
                <div class="info-box" style="background: rgb(251, 251, 250);">
                    <p style="color: rgba(55, 53, 47, 0.65);">Please enter your API key to continue</p>
                </div>
            """, unsafe_allow_html=True)
            if not debug_mode:
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
            if not all([uploaded_files, job_description]) and not debug_mode:
                st.error("Please provide both job description and CVs")
                return
            
            with st.spinner('Analyzing CVs with AI...'):
                # Display more info in debug mode
                if debug_mode and not api_key:
                    st.warning("Running in debug mode without API key")
                
                results = []
                progress_bar = st.progress(0)
                
                # Process CVs with concurrent execution
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = [
                        executor.submit(process_cv, file, job_description) 
                        for file in uploaded_files
                    ]
                    for i, future in enumerate(concurrent.futures.as_completed(futures)):
                        try:
                            result = future.result()
                            results.append(result)
                        except Exception as e:
                            if debug_mode:
                                st.error(f"Error processing file: {str(e)}")
                            results.append({
                                "Filename": f"File {i+1}",
                                "Error": str(e)
                            })
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
                <p>This tool uses Google Gemini AI to analyze CVs against the job description and provide interview recommendations.</p>
                <p>The AI scores candidates based on their match to the job requirements and provides specific strengths and areas to discuss during interviews.</p>
            </div>
            
            <h3>Troubleshooting</h3>
            <div class='info-box'>
                <p><strong>API Key Issues:</strong></p>
                <ul>
                    <li>Ensure your Gemini API key is active and has proper permissions</li>
                    <li>Check that you have billing enabled on your Google Cloud account</li>
                    <li>Verify quotas and usage limits in your Google AI Studio dashboard</li>
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
