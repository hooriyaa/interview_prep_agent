import streamlit as st
import os
import asyncio  
import re
from dotenv import load_dotenv
from fpdf import FPDF
import pdfplumber
import google.generativeai as genai
from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI

# ✅ Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ✅ Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-pro")

# ✅ Initialize OpenAI SDK for Gemini
provider = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai",
)

# ✅ Configure the language model
model = OpenAIChatCompletionsModel(model="gemini-2.0-flash", openai_client=provider)

# ✅ AI Agent using Gemini API
agent = Agent(
    name="Job Interview Prep Agent",
    instructions="Assist users in preparing for job interviews by providing sample questions, conducting mock interviews, offering feedback, and analyzing resumes. If someone asks an irrelevant question, respond with: 'Hooriya's agent is here just for Interview Prep, I can't answer anything else, sorry.'",
    model=model,
)

# ✅ Initialize session state
if "questions" not in st.session_state:
    st.session_state.questions = ""

if "feedback" not in st.session_state:
    st.session_state.feedback = ""

# ✅ Function to generate sample interview questions
def generate_sample_questions(job_title: str) -> str:
    try:
        prompt = f"Generate 10 common interview questions for a {job_title} position, along with ideal sample answers."
        result = asyncio.run(Runner.run(agent, prompt))  
        return result.final_output
    except Exception as e:
        return f"Error generating questions: {e}"

# ✅ Function to conduct mock interview and provide feedback
def mock_interview_response(user_answer: str) -> str:
    try:
        prompt = f"""
        You are an AI Interviewer. Evaluate the following candidate response based on:
        - Clarity and structure
        - Technical correctness
        - Confidence and tone
        - Grammar and fluency
        - Overall effectiveness

        Then provide:
        1. **Score out of 10** 📊
        2. **Constructive feedback** 📝
        3. **Suggested improvements** 🚀
        4. **A relevant follow-up question** ❓

        Candidate's Answer:
        "{user_answer}"
        """
        result = asyncio.run(Runner.run(agent, prompt))  
        return result.final_output
    except Exception as e:
        return f"Error analyzing response: {e}"

# ✅ Resume analysis using PDFPlumber
def analyze_resume(uploaded_file) -> str:
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            text = "".join(page.extract_text() for page in pdf.pages if page.extract_text())
        prompt = f"Analyze this resume and provide suggestions for improvement:\n{text}"
        result = asyncio.run(Runner.run(agent, prompt))  
        return result.final_output
    except Exception as e:
        return f"Error analyzing resume: {e}"

# ✅ Function to clean text (remove emojis)
def clean_text(text):
    return re.sub(r'[^\x00-\x7F]+', '', text)  # Non-ASCII characters (like emojis) remove karega

# ✅ Generate a PDF report with FPDF
def generate_pdf_report(content: str, filename: str):
    try:
        pdf = FPDF()
        pdf.add_page()

        # ✅ Use Helvetica (Unicode supported font)
        pdf.set_font("helvetica", size=12)  

        # ✅ Clean text before saving (remove emojis if needed)
        content = clean_text(content)

        pdf.multi_cell(0, 10, content)
        pdf.output(filename, "F")
    except Exception as e:
        st.error(f"Error generating PDF: {e}")

# ✅ Streamlit App UI
st.set_page_config(page_title="Job Interview Prep Agent", page_icon="🧠")
st.title("🧠 Job Interview Prep Agent")
st.markdown("Prepare effectively for your job interviews with personalized assistance.")

# ✅ Sidebar - Resume Upload
st.sidebar.header("📄 Upload Your Resume")
resume_file = st.sidebar.file_uploader("Upload Resume (PDF)", type="pdf")
if resume_file:
    with st.spinner("Analyzing resume..."):
        resume_feedback = analyze_resume(resume_file)
    st.sidebar.success("Resume analysis complete.")
    st.sidebar.download_button("Download Resume Feedback", resume_feedback, file_name="resume_feedback.txt")

# ✅ Reset button to clear session state
if st.sidebar.button("🔄 Reset All Data"):
    st.session_state.questions = ""
    st.session_state.feedback = ""
    st.sidebar.success("Data has been reset!")

# ✅ Main Tabs
tab1, tab2, tab3 = st.tabs(["Sample Questions", "Mock Interview", "Tips & Resources"])

# ✅ Sample Questions Tab
with tab1:
    st.header("📋 Generate Sample Questions")
    job_title = st.text_input("Enter the job title:", placeholder="e.g., Software Engineer")
    
    if st.button("Generate Questions"):
        with st.spinner("Generating sample questions..."):
            st.session_state.questions = generate_sample_questions(job_title)
        st.markdown(st.session_state.questions)
    elif st.session_state.questions:
        st.markdown(st.session_state.questions)

# ✅ Mock Interview Tab
with tab2:
    st.header("🎤 Mock Interview")
    user_answer = st.text_area("Your Answer:", height=150, placeholder="Type your response here...")
    
    if st.button("Submit Answer"):
        with st.spinner("Analyzing your response..."):
            feedback_response = mock_interview_response(user_answer)
            st.session_state.feedback = feedback_response
        st.subheader("📝 Feedback on Your Answer:")
        st.markdown(st.session_state.feedback)

# ✅ Tips & Resources Tab
with tab3:
    st.header("📚 Tips & Resources")
    st.markdown("""
    - Practice answering behavioral questions using the STAR method (Situation, Task, Action, Result).
    - Research the company and role thoroughly before your interview.
    - Prepare questions to ask the interviewer.
    - Dress appropriately and maintain positive body language during the interview.
    - Get feedback and refine your answers over time.
    """)

# ✅ Download Interview Report
if st.button("Download Full Interview Report"):
    try:
        report_content = f"Interview Preparation Summary:\n\nSample Questions:\n{st.session_state.questions or 'No questions generated yet.'}\n\nFeedback:\n{st.session_state.feedback or 'No feedback generated yet.'}"
        filename = "interview_report.pdf"
        generate_pdf_report(report_content, filename)
        with open(filename, "rb") as file:
            st.download_button("📥 Download Report PDF", file, file_name=filename)
    except Exception as e:
        st.error(f"Error downloading report: {e}")

st.markdown("---")  
st.markdown(
    "<div style='text-align: center; font-size: 16px; font-weight: bold'>"
    "Created by ❤️ Hooriya M. Fareed</div>",
    unsafe_allow_html=True
)
