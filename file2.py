import os
import json
import numpy as np
from typing import List, Dict, Any
import requests
from sklearn.metrics.pairwise import cosine_similarity

class LLMEmbeddingService:
    """Service for generating and comparing embeddings using LLMs"""
    
    def _init_(self, model_name="all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.use_local = True
        except ImportError:
            # Fall back to API-based embeddings
            self.use_local = False
            self.api_endpoint = "http://localhost:11434/api/embeddings"
            
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding vector for text"""
        if self.use_local:
            return self.model.encode(text)
        else:
            # Use Ollama API
            response = requests.post(
                self.api_endpoint,
                json={"model": "llama3", "prompt": text}
            )
            
            if response.status_code == 200:
                embedding = np.array(response.json().get('embedding', []))
                return embedding
            else:
                raise Exception(f"API request failed with status {response.status_code}")
    
    def compare_texts(self, text1: str, text2: str) -> float:
        """Compare two texts using cosine similarity of embeddings"""
        embedding1 = self.get_embedding(text1)
        embedding2 = self.get_embedding(text2)
        
        # Reshape for sklearn cosine similarity
        e1 = embedding1.reshape(1, -1)
        e2 = embedding2.reshape(1, -1)
        
        return cosine_similarity(e1, e2)[0][0]

class EnhancedMatchingEngine:
    """Advanced matching engine using LLMs and embeddings"""
    
    def _init_(self):
        self.embedding_service = LLMEmbeddingService()
        self.llm_endpoint = "http://localhost:11434/api/generate"
        
    def extract_job_features(self, job: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key features from job description"""
        # Combine relevant fields into a feature vector
        description = job.get('description', '')
        requirements = json.loads(job.get('requirements', '[]'))
        responsibilities = json.loads(job.get('responsibilities', '[]'))
        keywords = json.loads(job.get('keywords', '[]'))
        
        # Create separate embeddings for each aspect
        features = {
            "title_embedding": self.embedding_service.get_embedding(job.get('title', '')),
            "description_embedding": self.embedding_service.get_embedding(description),
            "requirements_embedding": self.embedding_service.get_embedding(' '.join(requirements)),
            "responsibilities_embedding": self.embedding_service.get_embedding(' '.join(responsibilities)),
            "keywords_embedding": self.embedding_service.get_embedding(' '.join(keywords)),
        }
        
        return features
    
    def extract_candidate_features(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key features from candidate profile"""
        # Parse JSON fields
        education = json.loads(candidate.get('education', '[]'))
        experience = json.loads(candidate.get('experience', '[]'))
        top_skills = json.loads(candidate.get('top_skills', '[]'))
        
        # Create separate embeddings for each aspect
        features = {
            "skills_embedding": self.embedding_service.get_embedding(candidate.get('skills', '')),
            "experience_embedding": self.embedding_service.get_embedding(' '.join(experience)),
            "education_embedding": self.embedding_service.get_embedding(' '.join(education)),
            "certifications_embedding": self.embedding_service.get_embedding(candidate.get('certifications', '')),
            "achievements_embedding": self.embedding_service.get_embedding(candidate.get('achievements', '')),
            "tech_stack_embedding": self.embedding_service.get_embedding(candidate.get('tech_stack', '')),
            "top_skills_embedding": self.embedding_service.get_embedding(' '.join(top_skills)),
        }
        
        return features
    
    def calculate_weighted_match_score(
        self, 
        job_features: Dict[str, np.ndarray], 
        candidate_features: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """Calculate weighted match score between job and candidate"""
        # Weights for different match components
        weights = {
            "skills_match": 0.3,
            "experience_match": 0.25,
            "education_match": 0.15,
            "tech_stack_match": 0.2,
            "certifications_match": 0.1,
        }
        
        # Calculate individual component scores
        scores = {}
        
        # Skills match (candidate skills vs job requirements)
        scores["skills_match"] = cosine_similarity(
            candidate_features["skills_embedding"].reshape(1, -1),
            job_features["requirements_embedding"].reshape(1, -1)
                )[0][0]
        
        # Education match (candidate education vs job requirements)
        scores["education_match"] = cosine_similarity(
            candidate_features["education_embedding"].reshape(1, -1),
            job_features["requirements_embedding"].reshape(1, -1)
        )[0][0]
        
        # Tech stack match (candidate tech stack vs job requirements)
        scores["tech_stack_match"] = cosine_similarity(
            candidate_features["tech_stack_embedding"].reshape(1, -1),
            job_features["requirements_embedding"].reshape(1, -1)
        )[0][0]
        
        # Certifications match (candidate certifications vs job requirements)
        scores["certifications_match"] = cosine_similarity(
            candidate_features["certifications_embedding"].reshape(1, -1),
            job_features["requirements_embedding"].reshape(1, -1)
        )[0][0]
        
        # Calculate the final weighted score
        final_score = (
            scores["skills_match"] * weights["skills_match"] +
            scores["experience_match"] * weights["experience_match"] +
            scores["education_match"] * weights["education_match"] +
            scores["tech_stack_match"] * weights["tech_stack_match"] +
            scores["certifications_match"] * weights["certifications_match"]
        )
        
        # Return the scores and the final weighted score
        return {
            "individual_scores": scores,
            "final_score": final_score
        }

    def match_candidate_to_job(self, job: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
        """Match a candidate to a job and return the match score"""
        job_features = self.extract_job_features(job)
        candidate_features = self.extract_candidate_features(candidate)
        
        match_score = self.calculate_weighted_match_score(job_features, candidate_features)
        
        return match_score

# Example usage
if _name_ == "_main_":
    engine = EnhancedMatchingEngine()
    
    job_example = {
        "title": "Software Engineer",
        "description": "Develop and maintain software applications.",
        "requirements": json.dumps(["Python", "Django", "REST APIs"]),
        "responsibilities": json.dumps(["Write clean code", "Collaborate with team"]),
        "keywords": json.dumps(["software", "engineer", "development"])
    }
    
    candidate_example = {
        "skills": "Python, Django, JavaScript",
        "experience": json.dumps(["2 years at Company A", "1 year at Company B"]),
        "education": json.dumps(["BSc in Computer Science"]),
        "certifications": "Certified Python Developer",
        "achievements": "Developed a successful web application",
        "tech_stack": "Django, React",
        "top_skills": json.dumps(["Python", "Django", "Teamwork"])
    }
    
    match_result = engine.match_candidate_to_job(job_example, candidate_example)
    print(match_result)