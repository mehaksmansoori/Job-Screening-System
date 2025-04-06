import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pickle

# Download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Set page config
st.set_page_config(
    page_title="Job Screening System",
    page_icon="🔍",
    layout="wide"
)

# Function to preprocess text
def preprocess_text(text):
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Tokenize
        tokens = word_tokenize(text)
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        # Join tokens back to string
        return ' '.join(tokens)
    return ""

# Function to extract skills from text
def extract_skills(text, skills_list):
    text = text.lower()
    found_skills = []
    for skill in skills_list:
        skill_pattern = r'\b' + re.escape(skill.lower()) + r'\b'
        if re.search(skill_pattern, text):
            found_skills.append(skill)
    return found_skills

# Function to screen resumes
def screen_resumes(job_description, resumes, skills_list):
    # Preprocess job description
    processed_jd = preprocess_text(job_description)
    
    # Preprocess resumes
    processed_resumes = [preprocess_text(resume) for resume in resumes]
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    
    # Combine job description and resumes for vectorization
    all_documents = [processed_jd] + processed_resumes
    
    # Generate TF-IDF matrix
    tfidf_matrix = vectorizer.fit_transform(all_documents)
    
    # Calculate cosine similarity between job description and each resume
    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
    
    # Extract skills from each resume
    resume_skills = [extract_skills(resume, skills_list) for resume in resumes]
    
    # Calculate skill match percentage
    skill_match_percentages = [
        (len(skills) / len(skills_list)) * 100 if skills_list else 0
        for skills in resume_skills
    ]
    
    # Combine results
    results = []
    for i, (sim_score, skills, skill_percentage) in enumerate(zip(cosine_similarities[0], resume_skills, skill_match_percentages)):
        results.append({
            'Resume': f'Resume {i+1}',
            'Similarity Score': sim_score * 100,  # Convert to percentage
            'Skills': skills,
            'Skills Match (%)': skill_percentage
        })
    
    # Sort results by similarity score (descending)
    results = sorted(results, key=lambda x: x['Similarity Score'], reverse=True)
    
    return results

# Sample skills list - this would ideally be loaded from a more comprehensive source
default_skills_list = [
    "python", "java", "c++", "javascript", "html", "css", "sql", "nosql", 
    "react", "angular", "vue", "node.js", "express", "django", "flask",
    "tensorflow", "pytorch", "keras", "scikit-learn", "pandas", "numpy",
    "data analysis", "machine learning", "deep learning", "nlp", "computer vision",
    "aws", "azure", "gcp", "docker", "kubernetes", "ci/cd", "jenkins",
    "git", "agile", "scrum", "project management", "communication", "teamwork"
]

# Main Streamlit app
st.title("🔍 Job Screening System")

# Sidebar
st.sidebar.header("Configuration")
uploaded_skills = st.sidebar.file_uploader("Upload Custom Skills List (CSV)", type="csv")

if uploaded_skills:
    try:
        skills_df = pd.read_csv(uploaded_skills)
        skills_list = skills_df.iloc[:, 0].tolist()  # Assuming skills are in the first column
        st.sidebar.success(f"Loaded {len(skills_list)} skills")
    except Exception as e:
        st.sidebar.error(f"Error loading skills: {e}")
        skills_list = default_skills_list
else:
    skills_list = default_skills_list

st.sidebar.write(f"Using {len(skills_list)} skills for matching")

# Job Description input
st.header("Job Description")
job_description = st.text_area("Enter the job description here:", height=200)

# Resume upload
st.header("Resume Upload")
uploaded_files = st.file_uploader("Upload resumes (TXT, PDF)", type=["txt", "pdf"], accept_multiple_files=True)

if uploaded_files:
    st.write(f"Uploaded {len(uploaded_files)} resume(s)")
    
    # Read resume content
    resumes = []
    resume_names = []
    
    for file in uploaded_files:
        try:
            if file.type == "application/pdf":
                # For PDF files - in a real app, you'd use a PDF parser like PyPDF2 or pdfplumber
                st.warning(f"PDF parsing is not implemented in this demo. Using filename as placeholder for {file.name}")
                resumes.append(file.name)  # In a real app, you'd extract text from PDF
            else:
                # For text files
                content = file.read().decode("utf-8")
                resumes.append(content)
            resume_names.append(file.name)
        except Exception as e:
            st.error(f"Error reading {file.name}: {e}")
    
    # Sample resumes for demonstration (if no files uploaded)
    if not resumes:
        st.info("Using sample resumes for demonstration")
        resumes = [
            "Experienced Python developer with 5 years of experience in Django and Flask. Skilled in SQL, data analysis with pandas and numpy. Good communication skills and teamwork.",
            "Full stack developer with expertise in React, Node.js, and MongoDB. 3 years of experience in agile development. Knowledge of CI/CD pipelines and Docker.",
            "Machine learning engineer proficient in TensorFlow, PyTorch and scikit-learn. Experience with NLP and computer vision projects. AWS certified."
        ]
        resume_names = ["Sample Resume 1", "Sample Resume 2", "Sample Resume 3"]

    # Screen button
    if st.button("Screen Resumes"):
        if job_description:
            with st.spinner("Screening resumes..."):
                results = screen_resumes(job_description, resumes, skills_list)
                
                # Display results
                st.header("Screening Results")
                
                # Create DataFrame for results
                results_df = pd.DataFrame(results)
                
                # Display summary table
                st.subheader("Summary")
                summary_df = results_df[['Resume', 'Similarity Score', 'Skills Match (%)']]
                summary_df = summary_df.rename(columns={'Resume': 'Resume'})
                
                # Replace default resume names with actual filenames
                for i, name in enumerate(resume_names):
                    summary_df = summary_df.replace(f'Resume {i+1}', name)
                
                st.dataframe(summary_df.style.format({
                    'Similarity Score': '{:.2f}%',
                    'Skills Match (%)': '{:.2f}%'
                }))
                
                # Display detailed results for each resume
                st.subheader("Detailed Results")
                for i, result in enumerate(results):
                    with st.expander(f"{resume_names[i]} - Score: {result['Similarity Score']:.2f}%"):
                        st.write(f"**Skills Found ({len(result['Skills'])})**:")
                        if result['Skills']:
                            st.write(", ".join(result['Skills']))
                        else:
                            st.write("No matching skills found")
        else:
            st.error("Please enter a job description")
else:
    st.info("Please upload resumes to begin screening")

# Add explanations
with st.expander("How it works"):
    st.write("""
    This job screening system works by:
    1. **Text Preprocessing**: Removing special characters, converting to lowercase, and removing common stopwords
    2. **TF-IDF Vectorization**: Converting text to numerical vectors that capture word importance
    3. **Cosine Similarity**: Measuring the similarity between job descriptions and resumes
    4. **Skills Matching**: Identifying specific skills mentioned in resumes from a predefined list
    
    The system provides two key metrics:
    - **Similarity Score**: Overall content match between job description and resume
    - **Skills Match**: Percentage of required skills found in the resume
    """)

# Add footer
st.markdown("---")
st.markdown("Job Screening System | Created with Streamlit")
