import streamlit as st
import google.generativeai as genai
import os
import PyPDF2
import docx
import re
from io import BytesIO
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini API
API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(model_name="gemini-2.0-flash")


# Set page config
st.set_page_config(
    page_title="TalentScout Hiring Assistant",
    page_icon="👔",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "resume_data" not in st.session_state:
    st.session_state.resume_data = None
if "interview_stage" not in st.session_state:
    st.session_state.interview_stage = "upload"
if "question_count" not in st.session_state:
    st.session_state.question_count = 0
if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = None
if "index" not in st.session_state:  # Changed from chroma_client to index
    st.session_state.index = None
if "question_embeddings" not in st.session_state:  # Added to store embeddings
    st.session_state.question_embeddings = None
if "question_data" not in st.session_state:  # Added to store question metadata
    st.session_state.question_data = None
if "exit_keywords" not in st.session_state:
    st.session_state.exit_keywords = ["exit", "quit", "thank you"]

# Define helper functions
def extract_text_from_pdf(file_bytes):
    """Extract text from a PDF file"""
    with BytesIO(file_bytes) as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text

def extract_text_from_docx(file_bytes):
    """Extract text from a DOCX file"""
    with BytesIO(file_bytes) as docx_file:
        doc = docx.Document(docx_file)
        return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_resume(uploaded_file):
    """Extract text based on file type"""
    file_bytes = uploaded_file.getvalue()
    file_type = uploaded_file.type
    
    if "pdf" in file_type:
        return extract_text_from_pdf(file_bytes)
    elif "docx" in file_type or "doc" in file_type:
        return extract_text_from_docx(file_bytes)
    else:
        return None

def parse_resume(uploaded_file):
    """Parse resume and extract key information"""
    text = extract_text_from_resume(uploaded_file)
    if not text:
        return None
    
    # Use Gemini to extract structured data from resume
    model = genai.GenerativeModel('gemini-2.0-flash')
    prompt = f"""
    Extract the following information from this resume:
    
    Resume text:
    {text}
    
    Extract and return the following information in JSON format:
    - name: The candidate's full name
    - email: The candidate's email address
    - phone: The candidate's phone number
    - years_of_experience: Total years of experience (number only)
    - current_role: Current or most recent job title
    - skills: List of technical skills (focus on programming languages, frameworks, tools)
    - education: Highest degree and institution
    - experience: List of past roles with company names (most recent 3)
    
    Return ONLY the JSON without any other text or explanation.
    """
    
    try:
        response = model.generate_content(prompt)
        # Use regex to extract JSON from the response
        json_match = re.search(r'```json\s*(.*?)\s*```', response.text, re.DOTALL)
        if json_match:
            import json
            return json.loads(json_match.group(1))
        else:
            import json
            return json.loads(response.text)
    except Exception as e:
        st.error(f"Error parsing resume: {str(e)}")
        return {
            "name": "Unknown",
            "email": "Not found",
            "phone": "Not found",
            "years_of_experience": "Unknown",
            "current_role": "Not found",
            "skills": ["Not found"],
            "education": "Not found",
            "experience": ["Not found"]
        }

def initialize_rag_system():
    """Initialize the RAG system with technical questions using FAISS"""
    # Initialize embedding model
    if st.session_state.embedding_model is None:
        with st.spinner("Loading embedding model..."):
            st.session_state.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Technical questions bank
    technical_questions = {
        "Python": [
            "What are the key differences between Python 2 and Python 3?",
            "Explain list comprehensions in Python with an example.",
            "How do you handle exceptions in Python?",
            "What are decorators in Python and how do you use them?",
            "Explain the GIL (Global Interpreter Lock) in Python.",
            "Write a function to check if a string is a palindrome."
        ],
        "JavaScript": [
            "What is the difference between '==' and '===' in JavaScript?",
            "Explain closures in JavaScript.",
            "What is the event loop in JavaScript?",
            "Describe the differences between var, let, and const.",
            "What are Promises and how do they work?",
            "Explain async/await in JavaScript."
        ],
        "React": [
            "What is JSX in React?",
            "Explain the component lifecycle in React.",
            "What are hooks in React and how do they work?",
            "Describe the differences between state and props.",
            "How does React handle routing?",
            "Explain the context API in React."
        ],
        "SQL": [
            "What is the difference between INNER JOIN and LEFT JOIN?",
            "Explain database normalization and its forms.",
            "How do you optimize a slow SQL query?",
            "What are indexes and how do they work?",
            "Write a query to find duplicate records in a table.",
            "Explain transactions in SQL."
        ],
        "Machine Learning": [
            "What is the difference between supervised and unsupervised learning?",
            "Explain overfitting and how to prevent it.",
            "What is the bias-variance tradeoff?",
            "Describe the steps in a machine learning pipeline.",
            "Explain the concept of regularization in machine learning.",
            "What are gradient descent and its variants?"
        ]
    }
    
    # Create a list of questions with metadata
    questions = []
    question_texts = []
    
    for skill, skill_questions in technical_questions.items():
        for q in skill_questions:
            questions.append({
                "text": q,
                "skill": skill,
                "difficulty": "medium"
            })
            question_texts.append(q)
    
    # Create embeddings for all questions
    embeddings = st.session_state.embedding_model.encode(question_texts)
    
    # Create and populate FAISS index
    dimension = embeddings.shape[1]  # Get dimension from embeddings
    index = faiss.IndexFlatL2(dimension)  # Using L2 distance
    faiss.normalize_L2(embeddings)  # Normalize embeddings
    index.add(embeddings)  # Add embeddings to index
    
    # Store in session state
    st.session_state.index = index
    st.session_state.question_embeddings = embeddings
    st.session_state.question_data = questions

def retrieve_relevant_questions(skills, n=5):
    """Retrieve relevant questions for the candidate's skills using FAISS"""
    if not skills or st.session_state.index is None:
        return []
    
    all_questions = []
    
    # Prioritize first 3 skills
    priority_skills = skills[:3]
    
    for skill in priority_skills:
        # Convert skill to embedding
        skill_embedding = st.session_state.embedding_model.encode([skill])
        faiss.normalize_L2(skill_embedding)
        
        # Search for relevant questions
        k = 2  # Get 2 questions per skill
        D, I = st.session_state.index.search(skill_embedding, k)
        
        # Get the questions
        for idx in I[0]:
            question = st.session_state.question_data[idx]["text"]
            all_questions.append(question)
    
    # If we don't have enough questions, add generic technical questions
    if len(all_questions) < n:
        generic_questions = [
            "Tell me about a challenging technical problem you solved recently.",
            "How do you approach learning new technologies?",
            "Describe your ideal work environment.",
            "How do you ensure code quality in your projects?",
            "What's your approach to debugging complex issues?"
        ]
        all_questions.extend(generic_questions[:n-len(all_questions)])
    
    return all_questions[:n]

def create_prompt(stage, resume_data=None, user_input=None, conversation_history=None, context=None):
    """Create a prompt for Gemini based on the current interview stage."""
    
    system_prompt = """
    You are TalentScout's AI Hiring Assistant, designed to conduct initial screening interviews for technology positions.
    Your goal is to evaluate candidates based on:
    1. Technical skills matching the position requirements
    2. Experience level and relevance
    3. Problem-solving abilities
    4. Communication skills
    
    Guidelines:
    - Maintain a professional but friendly tone
    - Ask one question at a time
    - Progress from basic to more complex technical questions
    - Adapt questions based on candidate responses
    - Follow up on unclear or incomplete answers
    - Don't reveal assessment criteria to candidates
    - Keep responses concise but informative
    """
    
    if stage == "start_interview":
        # Initial greeting after resume upload
        prompt = f"""
        {system_prompt}
        
        You've analyzed the candidate's resume with the following information:
        - Name: {resume_data.get('name', 'the candidate')}
        - Years of Experience: {resume_data.get('years_of_experience', 'Not specified')}
        - Skills: {', '.join(resume_data.get('skills', ['Not specified']))}
        - Current/Recent Role: {resume_data.get('current_role', 'Not specified')}
        
        Start the interview with a friendly introduction. Confirm some basic information from their resume.
        Then ask your first relevant technical question based on their most prominent skill.
        """
        
    elif stage == "interview":
        # Regular interview questions
        skills_str = ', '.join(resume_data.get('skills', ['Not specified'])) if resume_data else 'Not specified'
        
        # Create conversation history string without using \n in f-string
        conversation_str = ""
        if conversation_history:
            for msg in conversation_history[:-1]:
                conversation_str += f"{msg['role'].title()}: {msg['content']}\n"
        
        prompt = f"""
        {system_prompt}
        
        Resume Information:
        - Skills: {skills_str}
        - Experience: {resume_data.get('years_of_experience', 'Not specified')} years
        - Current/Recent Role: {resume_data.get('current_role', 'Not specified')}
        
        Relevant context from resume and knowledge base:
        {context}
        
        Previous conversation:
        {conversation_str}
        
        Candidate's last response: {user_input}
        
        Based on this information:
        1. Evaluate the candidate's last response
        2. Determine what to assess next (deeper on same topic or move to a new skill)
        3. Ask a relevant follow-up question OR move to a new important skill to assess
        
        Respond as the hiring assistant. Do not reveal your internal assessment process.
        """
        
    elif stage == "wrap_up":
        # Concluding the interview
        # Create conversation history string without using \n in f-string
        conversation_str = ""
        if conversation_history:
            for msg in conversation_history:
                conversation_str += f"{msg['role'].title()}: {msg['content']}\n"
        
        prompt = f"""
        {system_prompt}
        
        The interview is concluding. Based on the entire conversation:
        
        {conversation_str}
        
        Create a professional and friendly closing message that:
        1. Thanks the candidate for their time
        2. Explains the next steps in the interview process
        3. Gives them a timeframe for when they might hear back
        
        Do not provide an assessment of their performance.
        """
    
    return prompt

def generate_response(stage, resume_data=None, user_input=None, context=None, conversation_history=None):
    """Generate a response using Gemini"""
    # Set up Gemini
    model = genai.GenerativeModel(model_name="gemini-2.0-flash")
    
    # Create the appropriate prompt
    prompt = create_prompt(stage, resume_data, user_input, conversation_history, context)
    
    # Generate response
    response = model.generate_content(prompt)
    
    return response.text


# Main UI layout
st.title("TalentScout Hiring Assistant")

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/150x150.png?text=TalentScout", width=150)
    st.markdown("## TalentScout")
    st.markdown("AI-powered technical screening assistant")
    st.divider()
    
    if st.session_state.resume_data:
        st.markdown("### Candidate Information")
        st.markdown(f"**Name:** {st.session_state.resume_data.get('name', 'Not specified')}")
        st.markdown(f"**Current Role:** {st.session_state.resume_data.get('current_role', 'Not specified')}")
        st.markdown(f"**Experience:** {st.session_state.resume_data.get('years_of_experience', 'Not specified')} years")
        
        st.markdown("### Technical Skills")
        skills = st.session_state.resume_data.get('skills', [])
        if skills and skills[0] != "Not found":
            for skill in skills:
                st.markdown(f"- {skill}")
        else:
            st.markdown("No skills detected")
    
    if st.button("Reset Interview"):
        st.session_state.messages = []
        st.session_state.resume_data = None
        st.session_state.interview_stage = "upload"
        st.session_state.question_count = 0
        st.rerun()

# Main area
if st.session_state.interview_stage == "upload":
    st.write("## Welcome to TalentScout Technical Screening")
    st.write("Please upload your resume to begin the technical screening interview.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Upload your resume (PDF or DOCX)", type=["pdf", "docx"])
    
    with col2:
        if uploaded_file:
            st.write("Ready to begin!")
            if st.button("Start Interview", type="primary"):
                with st.spinner("Analyzing your resume..."):
                    # Parse resume
                    resume_data = parse_resume(uploaded_file)
                    st.session_state.resume_data = resume_data
                    
                    # Initialize RAG system
                    initialize_rag_system()
                    
                    # Update interview stage
                    st.session_state.interview_stage = "interview"
                    
                    # Generate initial message
                    initial_message = generate_response("start_interview", resume_data=resume_data)
                    st.session_state.messages.append({"role": "assistant", "content": initial_message})
                    st.session_state.question_count = 1
                    
                    st.rerun()
    
    # Display sample resume format
    with st.expander("Tips for better resume parsing"):
        st.write("""
        For best results, make sure your resume:
        - Clearly lists your technical skills
        - Includes years of experience
        - Details your current or most recent role
        - Has your contact information
        - Is in PDF or DOCX format
        - Can stop the interview with thank you or exit or quit
        """)

# In the interview stage section, modify to:
# In the interview stage section, modify to:
elif st.session_state.interview_stage == "interview":
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Show progress
    progress = min(st.session_state.question_count / 5 * 100, 100)
    st.progress(progress / 100, f"Interview Progress: {int(progress)}%")
    
    # Get user input
    user_input = st.chat_input("Your response:")
    if user_input:
        # Check for exit keywords
        if any(keyword in user_input.lower() for keyword in st.session_state.exit_keywords):
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.session_state.interview_stage = "wrap_up"
            wrap_up_message = generate_response(
                "wrap_up", 
                resume_data=st.session_state.resume_data,
                conversation_history=st.session_state.messages
            )
            st.session_state.messages.append({"role": "assistant", "content": wrap_up_message})
            st.rerun()
        
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Generate context for RAG
        context = ""
        if st.session_state.resume_data and "skills" in st.session_state.resume_data:
            # Get relevant questions based on skills
            relevant_questions = retrieve_relevant_questions(st.session_state.resume_data.get("skills", []))
            if relevant_questions:
                context = "Relevant technical questions to consider:\n" + "\n".join([f"- {q}" for q in relevant_questions])
        
        # Generate assistant response based on the current question count
        with st.spinner("Thinking..."):
            if st.session_state.question_count < 5:
                # Regular interview question
                response = generate_response(
                    "interview",
                    resume_data=st.session_state.resume_data,
                    user_input=user_input,
                    context=context,
                    conversation_history=st.session_state.messages
                )
                
                # Add assistant response to chat
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                # Increment question count AFTER generating and adding the question
                st.session_state.question_count += 1
            else:
                # This was the response to the 5th question, transition to wrap-up
                st.session_state.interview_stage = "wrap_up"
                wrap_up_message = generate_response(
                    "wrap_up", 
                    resume_data=st.session_state.resume_data,
                    conversation_history=st.session_state.messages
                )
                st.session_state.messages.append({"role": "assistant", "content": wrap_up_message})
        
        st.rerun()

# Add this section to handle the wrap-up stage properly
elif st.session_state.interview_stage == "wrap_up":
    # Display all chat messages including the wrap-up message
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Show completed progress bar
    st.progress(1.0, "Interview Complete")
    
    # Add a message explaining what happens next
    st.info("The technical screening interview is now complete. Thank you for your participation!")
    
    # Add a button to start a new interview if desired
    if st.button("Start New Interview"):
        st.session_state.messages = []
        st.session_state.resume_data = None
        st.session_state.interview_stage = "upload"
        st.session_state.question_count = 0
        st.rerun()
