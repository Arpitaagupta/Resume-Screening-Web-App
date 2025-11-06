import streamlit as st
import pickle
import docx
import PyPDF2
import re

# Load pre-trained model and TF-IDF vectorizer
svc_model = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))
le = pickle.load(open('encoder.pkl', 'rb'))


# Function to clean resume text
def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText


# Extract text from PDF
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


# Extract text from DOCX
def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = ''
    for paragraph in doc.paragraphs:
        text += paragraph.text + '\n'
    return text


# Extract text from TXT
def extract_text_from_txt(file):
    try:
        text = file.read().decode('utf-8')
    except UnicodeDecodeError:
        text = file.read().decode('latin-1')
    return text


# Handle file upload and extraction
def handle_file_upload(uploaded_file):
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension == 'pdf':
        text = extract_text_from_pdf(uploaded_file)
    elif file_extension == 'docx':
        text = extract_text_from_docx(uploaded_file)
    elif file_extension == 'txt':
        text = extract_text_from_txt(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Please upload a PDF, DOCX, or TXT file.")
    return text


# Predict resume category
def pred(input_resume):
    cleaned_text = cleanResume(input_resume)
    vectorized_text = tfidf.transform([cleaned_text])
    vectorized_text = vectorized_text.toarray()
    predicted_category = svc_model.predict(vectorized_text)
    predicted_category_name = le.inverse_transform(predicted_category)
    return predicted_category_name[0]


# Streamlit App
def main():
    st.set_page_config(page_title="Resume Category Predictor", page_icon="📄", layout="wide")

    # --- Custom Dark Mode Styling ---
    st.markdown("""
        <style>
        body {
            background-color: #000000;
            color: #e0e0e0;
        }
        .main {
            background-color: transparent;
            padding: 2rem 3rem;
        }
        .title {
            text-align: center;
            font-size: 2.4rem;
            color: #ffffff;
            font-weight: 700;
            margin-bottom: 0.3rem;
            letter-spacing: 1px;
        }
        .subtitle {
            text-align: center;
            color: #cccccc;
            font-size: 1.1rem;
            margin-bottom: 2rem;
        }
        .stButton>button {
            background-color: #0d47a1 !important;
            color: #ffffff !important;
            border-radius: 10px !important;
            font-size: 1rem !important;
            padding: 0.6rem 1.2rem !important;
        }
        .stButton>button:hover {
            background-color: #1565c0 !important;
            transform: scale(1.03);
            transition: 0.3s ease-in-out;
        }
        .prediction-box {
            background: linear-gradient(90deg, #1565c0, #0d47a1);
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
            font-size: 1.2rem;
            font-weight: bold;
            color: white;
            margin-top: 1.5rem;
            box-shadow: 0 0 15px rgba(21,101,192,0.6);
        }
        </style>
    """, unsafe_allow_html=True)

    # --- App Layout ---
    st.markdown('<div class="main">', unsafe_allow_html=True)

    st.markdown('<div class="title">📄 Resume Category Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Upload your resume (PDF, DOCX, or TXT) and discover the predicted job category!</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📁 Upload Resume", type=["pdf", "docx", "txt"])

    if uploaded_file is not None:
        try:
            resume_text = handle_file_upload(uploaded_file)
            st.success("✅ Successfully extracted text from the uploaded resume.")

            if st.checkbox("📝 Show extracted text"):
                st.text_area("Extracted Resume Text", resume_text, height=300)

            if st.button("🔍 Predict Category"):
                category = pred(resume_text)
                st.markdown(f"<div class='prediction-box'>Predicted Category: <b>{category}</b></div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
