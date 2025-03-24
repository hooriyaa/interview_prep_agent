import streamlit as st
import os
import asyncio  
import speech_recognition as sr
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
    prompt = f"Generate 10 common interview questions for a {job_title} position, along with ideal sample answers."
    
    # ✅ Use asyncio.run() to handle async functions in Streamlit
    result = asyncio.run(Runner.run(agent, prompt))  
    return result.final_output

# ✅ Function to conduct mock interview and provide feedback
def mock_interview_response(user_answer: str) -> str:
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
    
    # ✅ Use asyncio.run() to handle async functions
    result = asyncio.run(Runner.run(agent, prompt))  
    return result.final_output

# ✅ Voice-to-text conversion using SpeechRecognition
def transcribe_audio() -> str:
    try:
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            st.info("Listening... Please speak your answer clearly.")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=70)  

        try:
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            return "Sorry, I couldn't understand your response. Please try again."
        except sr.RequestError as e:
            return f"Could not process audio; {e}"

    except Exception as e:
        return f"Voice input error: {e}"

# ✅ Resume analysis using PDFPlumber
def analyze_resume(uploaded_file) -> str:
    with pdfplumber.open(uploaded_file) as pdf:
        text = "".join(page.extract_text() for page in pdf.pages if page.extract_text())
    prompt = f"Analyze this resume and provide suggestions for improvement:\n{text}"
    
    # ✅ Use asyncio.run() for async execution
    result = asyncio.run(Runner.run(agent, prompt))  
    return result.final_output

# ✅ Generate a PDF report with FPDF
def generate_pdf_report(content: str, filename: str):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, content)
    pdf.output(filename)

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
    
    interview_mode = st.radio("Select Input Mode:", ("Text Input", "Voice Input"))

    if interview_mode == "Text Input":
        user_answer = st.text_area("Your Answer:", height=150, placeholder="Type your response here...")
        
        if st.button("Submit Text Answer"):
            with st.spinner("Analyzing your response..."):
                feedback_response = mock_interview_response(user_answer)
                st.session_state.feedback = feedback_response

            st.subheader("📝 Feedback on Your Answer:")
            st.markdown(st.session_state.feedback)
            
    else:
        if st.button("Start Voice Interview"):
            with st.spinner("Listening..."):
                user_answer = transcribe_audio()

            st.text(f"You said: {user_answer}")

            if "Voice input error" in user_answer:
                st.error(user_answer)
            else:
                with st.spinner("Analyzing your spoken answer..."):
                    feedback_response = mock_interview_response(user_answer)
                    st.session_state.feedback = feedback_response

                st.subheader("📝 Feedback on Your Answer:")
                st.markdown(st.session_state.feedback)

    if st.session_state.feedback and "❓" in st.session_state.feedback:
        follow_up_question = st.session_state.feedback.split("❓")[-1]
        st.subheader("🔄 Follow-up Question:")
        st.markdown(f"**{follow_up_question}**")

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

if st.button("Download Full Interview Report"):
    report_content = f"Interview Preparation Summary:\n\nSample Questions:\n{st.session_state.questions or 'No questions generated yet.'}\n\nFeedback:\n{st.session_state.feedback or 'No feedback generated yet.'}"
    generate_pdf_report(report_content, "interview_report.pdf")
    with open("interview_report.pdf", "rb") as file:
        st.download_button("Download Report PDF", file, file_name="interview_report.pdf")

st.markdown("---")  
st.markdown(
    "<div style='text-align: center; font-size: 16px; font-weight: bold'>"
    "Created by ❤️ Hooriya M. Fareed</div>",
    unsafe_allow_html=True
)
