# TalentScout Hiring Assistant

An AI-powered technical screening assistant that conducts automated initial interviews for technology positions.

## Overview

TalentScout Hiring Assistant is a Streamlit application that uses Google's Generative AI (Gemini) to conduct technical screening interviews with candidates. The system analyzes resumes, creates personalized interview questions, and provides a conversational interface for candidates.

## Features

- **Resume Analysis**: Automatically extracts skills, experience, and other key information from PDF and DOCX resumes
- **AI-Driven Interviews**: Uses Gemini AI to conduct natural conversations and technical assessments
- **Skill-Based Questions**: Matches candidate skills to relevant technical questions using a vector similarity search
- **Adaptive Questioning**: Adjusts follow-up questions based on candidate responses
- **Progress Tracking**: Shows interview progress and manages the full interview lifecycle

## Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/talentscout.git
cd talentscout
```
## Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
## Install dependencies

```bash
pip install -r requirements.txt
```
## Create a .env file with your Google API key

GOOGLE_API_KEY=your_gemini_api_key_here

## Requirements

- Python 3.8+
- Streamlit
- Google Generative AI Python SDK
- PyPDF2
- python-docx
- Sentence Transformers
- FAISS
- pandas
- numpy
- python-dotenv

# Usage

## Run the Streamlit app

```bash
streamlit run app.py
```

Open your browser at http://localhost:8501
Upload a resume (PDF or DOCX format)
Begin the technical interview
Complete the interview (use "thank you", "exit", or "quit" to end early)

## How It Works

- Resume Upload: The system parses the uploaded resume to extract key information.
- RAG System Initialization: A retrieval-augmented generation system is initialized with a bank of technical questions.
- Interview Process: The AI assistant conducts a 5-question technical interview, adapting questions based on the candidate's skills and responses.
- Interview Conclusion: After 5 questions or when the candidate indicates they're done, the system provides a wrap-up message.

## Project Structure

- app.py: Main Streamlit application
- requirements.txt: Python dependencies
- .env: Environment variables (API keys)

Future Enhancements

- Integration with ATS systems
- Custom interview templates for different roles
- More advanced resume parsing
- Interview recording and automated summaries
- Integration with calendaring systems for scheduling follow-ups

## Acknowledgments

- Google Generative AI (Gemini)
- Streamlit
- FAISS library
- Sentence Transformers
