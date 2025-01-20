import streamlit as st
import pandas as pd
import re
from io import StringIO
from pdfminer.high_level import extract_text_to_fp
from pdfminer.layout import LAParams
import concurrent.futures
import google.generativeai as genai
from typing import Dict, List

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

def display_results(df, match_threshold):
    """Display analysis results"""
    # Filter CVs based on the match threshold
    filtered_df = df[df['Match Percentage'] >= match_threshold / 100].copy()
    filtered_df['Match Percentage'] = filtered_df['Match Percentage'] * 100  # Convert to percentage for display
    filtered_df = filtered_df.sort_values(["Match Percentage", "AI Suitability Score"], 
                                        ascending=[False, False]).reset_index(drop=True)
    
    st.header("Ranked CVs")
    
    # Display results
    if not filtered_df.empty:
        for _, row in filtered_df.iterrows():
            # Color code the interview recommendation
            rec_color = {
                "Strongly Recommend": "🟢",
                "Recommend": "🟡",
                "Consider": "🟠",
                "Do Not Recommend": "🔴",
                "Analysis Failed": "⚪"
            }.get(row['interview_recommendation'], "⚪")
            
            with st.expander(
                f"{rec_color} {row['Filename']} - Match: {row['Match Percentage']:.1f}% | AI Score: {row['AI Suitability Score']}"
            ):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("📝 **Matched Keywords:**")
                    st.write(", ".join(row['Matched Keywords']))
                    
                    st.write("💪 **Key Strengths:**")
                    for strength in row['Key Strengths']:
                        st.write(f"• {strength}")
                
                with col2:
                    st.write("🎯 **Potential Gaps:**")
                    for gap in row['Potential Gaps']:
                        st.write(f"• {gap}")
                    
                    st.write("🤖 **Interview Recommendation:**")
                    st.write(f"**{row['interview_recommendation']}**")
                    st.write(row['detailed_recommendation'])
        
        # Display summary dataframe with interview recommendations
        summary_df = filtered_df[["Filename", "Match Percentage", "AI Suitability Score", "interview_recommendation"]]
        summary_df = summary_df.rename(columns={
            "interview_recommendation": "Interview Recommendation"
        })
        st.dataframe(summary_df)
        
        # Display interview recommendations summary
        st.subheader("Interview Recommendations Summary")
        recommendations = filtered_df['interview_recommendation'].value_counts()
        st.write("Number of candidates by recommendation:")
        for rec, count in recommendations.items():
            st.write(f"- {rec}: {count}")
            
    else:
        st.warning(f"No CVs meet the minimum match threshold of {match_threshold}%")
        if not df.empty:
            st.info(f"Highest match percentage in uploaded CVs: {(df['Match Percentage'].max() * 100):.1f}%")
def main():
    st.title("🎯 AI-Powered CV Matcher and Analyzer")
    
    # Instructions
    with st.expander("How to Use"):
        st.markdown("""
        1. Enter your Google Gemini API key
        2. Enter the job description
        3. Enter your keywords (separated by commas)
        4. Upload the CVs you want to analyze (PDF format)
        5. Set your desired match threshold percentage
        6. Review the results including AI analysis
        
        To get a Google Gemini API key:
        1. Go to https://makersuite.google.com/app/apikey
        2. Create or select a project
        3. Generate an API key
        """)
    
    # API Key input in sidebar
    with st.sidebar:
        st.header("API Configuration")
        api_key = st.text_input(
            "Enter your Google Gemini API Key:",
            type="password",
            help="Get your API key from https://makersuite.google.com/app/apikey"
        )
        
        if api_key:
            if initialize_gemini(api_key):
                st.success("API key validated successfully!")
            else:
                st.error("Invalid API key. Please check and try again.")
                st.stop()
        else:
            st.warning("Please enter your Google Gemini API key to enable AI analysis.")
            st.stop()
    
    # Job Description input
    job_description = st.text_area(
        "Enter Job Description:",
        help="Paste the full job description here for AI analysis",
        height=200
    )
    
    # Keyword input
    keywords_input = st.text_area(
        "Enter keywords (separated by commas):", 
        help="Enter the keywords you want to search for in the CVs"
    )
    
    if keywords_input:
        keywords = [k.strip() for k in keywords_input.split(',') if k.strip()]
        st.info(f"Number of keywords entered: {len(keywords)}")
    else:
        keywords = []
    
    # File upload
    uploaded_files = st.file_uploader(
        "Upload CVs (PDF format)", 
        accept_multiple_files=True, 
        type=['pdf']
    )
    
    if uploaded_files:
        st.info(f"Number of files uploaded: {len(uploaded_files)}")
    
    # Match threshold slider
    match_threshold = st.slider(
        "Minimum Keyword Match Threshold (%)", 
        min_value=0, 
        max_value=100, 
        value=10,
        help="CVs must match at least this percentage of keywords to be included in results"
    )
    
    if st.button("Analyze CVs") and keywords and uploaded_files and job_description:
        with st.spinner('Analyzing CVs with AI...'):
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
            
            # Create dataframe and display results
            df = pd.DataFrame([r for r in results if "Error" not in r])
            display_results(df, match_threshold)
            
            # Display any errors
            errors = [r for r in results if "Error" in r]
            if errors:
                st.error("Errors occurred while processing some files:")
                for error in errors:
                    st.error(f"{error['Filename']}: {error['Error']}")

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
