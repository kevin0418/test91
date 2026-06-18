# 문제 만들기 앱 - 영어 버전  Gemini
#
import streamlit as st
import re

try:
    import pdfplumber
except ModuleNotFoundError:
    pdfplumber = None

# import openai  # OpenAI 대신 Gemini 사용을 위해 주석 처리
from google import genai # Google Gemini SDK를 임포트
import os
from google.genai.errors import APIError # API 오류 처리를 위해 임포트
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline
import tempfile
import PyPDF2
import docx
import requests
from io import BytesIO

# Streamlit page configuration
st.set_page_config(
    page_title="Test Question Generator",
    page_icon="❓",
    layout="wide"
)

# Title
st.title("📚 Test Question Generator by Kevin")
st.markdown("Automatically generate questions from PDF, Word, or text files.")

# --- Gemini API 클라이언트 초기화 ---
# Gemini API 키를 저장할 변수
gemini_api_key = None
# Gemini 클라이언트 객체
gemini_client = None

# Streamlit secrets 또는 환경변수에서 Gemini API 키를 안전하게 읽기
try:
    if "GEMINI_API_KEY" in st.secrets:
        gemini_api_key = st.secrets["GEMINI_API_KEY"]
    elif "api_keys" in st.secrets and "GEMINI_API_KEY" in st.secrets["api_keys"]:
        gemini_api_key = st.secrets["api_keys"]["GEMINI_API_KEY"]
    else:
        gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
except Exception:
    gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

# Sidebar settings
with st.sidebar:
    st.header("Settings")

    if gemini_api_key:
        try:
            # Gemini 클라이언트 초기화
            gemini_client = genai.Client(api_key=gemini_api_key)
        except Exception as e:
            st.error(f"Error initializing Gemini client: {e}")
            gemini_client = None
    else:
        st.warning(
            "Gemini API 키가 설정되지 않았습니다. "
            ".streamlit/secrets.toml 또는 환경변수(GEMINI_API_KEY)에 키를 추가하세요."
        )

    # Number of questions setting
    num_questions = st.slider(
        "Number of questions  (1-10)",
        min_value=1,
        max_value=10,
        value=5
    )
    
    # Question type selection
    question_type = st.selectbox(
        "Question type",
        ["Multiple Choice", "Short Answer", "Mixed"]
    )
    
    # Model selection (OpenAI GPT 대신 Gemini Pro로 변경)
    # model_choice = st.radio(
    #     "Model to use",
    #     ["Gemini Pro", "HuggingFace Transformers"]
    # )
    
    model_choice = "Gemini Pro"
    # st.radio("Model to use", ["Gemini Pro"], disabled=True)

    # Difficulty level
    difficulty = st.selectbox(
        "Difficulty level",
        ["Easy", "Medium", "Hard", "Adaptive"]
    )

# File upload section
st.header("1. Upload File")
uploaded_file = st.file_uploader(
    "Choose a file",
    type=['pdf', 'txt', 'docx'],
    help="Supported formats: PDF, TXT, DOCX"
)

# Direct text input option
text_input = st.text_area(
    "Or paste your text directly",
    height=150,
    placeholder="Paste the text you want to generate questions from here..."
)

def extract_text_from_pdf(file):
    """Extract text from PDF file"""
    try:
        file.seek(0)

        if pdfplumber is not None:
            with pdfplumber.open(file) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text.strip()

        # Fallback if pdfplumber is not installed
        file.seek(0)
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return None

def extract_text_from_txt(file):
    """Read content from text file"""
    try:
        return file.read().decode('utf-8')
    except Exception as e:
        st.error(f"Error reading text file: {e}")
        return None

def extract_text_from_docx(file):
    """Extract text from Word document"""
    try:
        # file 객체를 docx.Document에 전달
        doc = docx.Document(file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"Error reading Word document: {e}")
        return None

def ensure_nltk_resources():
    """Download required NLTK resources if they are missing."""
    try:
        for resource in ("punkt", "punkt_tab"):
            try:
                nltk.data.find(f"tokenizers/{resource}")
            except LookupError:
                st.warning(f"Downloading NLTK resource: {resource}")
                nltk.download(resource, quiet=True)
    except Exception as e:
        st.warning(f"Could not verify NLTK resources: {e}")


def preprocess_text(text):
    """Preprocess text"""
    if not text:
        return None

    ensure_nltk_resources()

    # Sentence tokenization
    try:
        sentences = sent_tokenize(text)
        st.info(f"Extracted {len(sentences)} sentences from the text.")
        return text
    except Exception as e:
        st.warning(f"Sentence tokenization failed: {e}. Using original text.")
        fallback_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]
        st.info(f"Extracted {len(fallback_sentences)} approximate sentences from the text.")
        return text

# --- Gemini API를 사용하도록 함수 변경 ---
def generate_questions_gemini(client, text, num_questions, question_type, difficulty):
    """Generate questions using Google Gemini"""
    if not client:
        st.error("Gemini API key is not set or client initialization failed.")
        return None
    
    # Map question types to English instructions
    question_type_map = {
        "Multiple Choice": "multiple choice questions with 4 options each",
        "Short Answer": "short answer questions",
        "Mixed": "mixed questions (both multiple choice and short answer)"
    }
    
    prompt = f"""
    Generate {num_questions} {question_type_map[question_type]} from the following text.
    
    Requirements:
    1. Create clear and well-structured questions
    2. For multiple choice questions, include 4 distinct options
    3. Include correct answers with brief explanations
    4. Questions should be appropriate for {difficulty.lower()} difficulty level
    5. Base questions only on the provided text content
    
    Text: {text[:4000]}  # Considering token limits
    
    Format for each question:
    Q1. [Question content]
    A. [Option A]
    B. [Option B]
    C. [Option C]
    D. [Option D]
    Answer: [Correct answer]
    Explanation: [Brief explanation]
    
    For short answer questions:
    Q1. [Question content]
    Answer: [Expected answer]
    Explanation: [Brief explanation]
    """
    
    try:
        # Gemini API 호출 (GenerativeModel 사용)
        response = client.models.generate_content(
            model='gemini-2.5-flash', # gpt-3.5-turbo 대신 gemini-2.5-flash 사용 (빠른 응답을 위해)
            contents=[
                {"role": "user", "parts": [{"text": prompt}]}
            ],
            config={
                "system_instruction": "You are a professional educator. You specialize in creating good quiz questions from given texts.",
                "max_output_tokens": 2048, # max_tokens 대신 max_output_tokens 사용
                "temperature": 0.7
            }
        )
        return response.text
    except APIError as e:
        st.error(f"Error calling Gemini API: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during Gemini API call: {e}")
        return None

def generate_questions_transformers(text, num_questions):
    """Generate questions using HuggingFace Transformers"""
    try:
        # Simple question generation pipeline
        qa_generator = pipeline(
            "text2text-generation",
            model="mrm8488/t5-base-finetuned-question-generation-ap"
        )
        
        # Process text in chunks
        chunk_size = 512
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        
        generated_questions = []
        for chunk in chunks[:3]:  # Process only first 3 chunks
            try:
                result = qa_generator(chunk, max_length=100, num_return_sequences=1)
                if result:
                    generated_questions.append(result[0]['generated_text'])
            except Exception as e:
                continue
        
        return "\n".join(generated_questions) if generated_questions else "Failed to generate questions."
    
    except Exception as e:
        st.error(f"Error loading Transformers model: {e}")
        return "Failed to load model. Please check your internet connection."

# Main processing logic
def main():
    extracted_text = None
    
    # Process file or text input
    if uploaded_file is not None:
        st.success(f"File uploaded successfully: {uploaded_file.name}")
        
        # Extract text based on file type
        if uploaded_file.type == "application/pdf":
            # PDF 파일 처리 시 BytesIO 사용 (pdfplumber 요구 사항 충족)
            uploaded_file.seek(0)
            extracted_text = extract_text_from_pdf(uploaded_file)
        elif uploaded_file.type == "text/plain":
            extracted_text = extract_text_from_txt(uploaded_file)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            extracted_text = extract_text_from_docx(uploaded_file)
    
    elif text_input.strip():
        extracted_text = text_input
        st.success("Text input received.")
    
    # Text preprocessing and preview
    if extracted_text:
        processed_text = preprocess_text(extracted_text)
        
        # Text preview
        with st.expander("Extracted Text Preview"):
            st.text_area("Text content", processed_text[:1000] + "..." if len(processed_text) > 1000 else processed_text, height=200)
        
        st.header("2. Generate Questions")
        
        # Generate questions button
        if st.button("Generate Questions", type="primary"):
            with st.spinner("Generating questions..."):
                
                # 모델 선택에 따라 함수 호출 변경
                if model_choice == "Gemini Pro": # "OpenAI GPT" 대신 "Gemini Pro" 사용
                    if gemini_client:
                        questions = generate_questions_gemini(gemini_client, processed_text, num_questions, question_type, difficulty)
                    else:
                        st.error("Cannot generate questions. Please provide a valid Gemini API Key.")
                        questions = None
                else:
                    questions = generate_questions_transformers(processed_text, num_questions)
                
                # Display results
                if questions:
                    st.header("3. Generated Questions")
                    st.markdown("---")
                    
                    # Format the output better
                    st.write(questions)
                    
                    # Download button
                    st.download_button(
                        label="Download Questions as Text",
                        data=questions,
                        file_name="generated_questions.txt",
                        mime="text/plain"
                    )
                else:
                    st.error("Failed to generate questions.")

    else:
        st.info("Please upload a file or enter text to get started.")

# Check for NLTK data once on startup
ensure_nltk_resources()

if __name__ == "__main__":
    main()