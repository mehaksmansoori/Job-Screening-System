import os
import json
import sqlite3
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Tuple

# Import Ollama for LLM integration
import requests
from sentence_transformers import SentenceTransformer

# Database connector
class DatabaseConnector:
    def _init_(self, db_path="job_screening.db"):
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        
    def connect(self):
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()
        return self.conn
        
    def close(self):
        if self.conn:
            self.conn.close()
            
    def execute(self, query, params=None):
        if not self.conn:
            self.connect()
        if params:
            self.cursor.execute(query, params)
        else:
            self.cursor.execute(query)
        self.conn.commit()
        return self.cursor
        
    def fetchall(self):
        return self.cursor.fetchall()
        
    def fetchone(self):
        return self.cursor.fetchone()
    
    def insert(self, table, data):
        placeholders = ', '.join(['?'] * len(data))
        columns = ', '.join(data.keys())
        query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
        self.execute(query, list(data.values()))
        return self.cursor.lastrowid

# Base Agent class
class Agent:
    def _init_(self, db_connector=None):
        self.db = db_connector if db_connector else DatabaseConnector()
        
    def run(self, *args, **kwargs):
        raise NotImplementedError("Agents must implement the run method")

# Job Description Analysis Agent
class JobDescriptionAgent(Agent):
    def _init_(self, db_connector=None):
        super()._init_(db_connector)
        self.llm_endpoint = "http://localhost:11434/api/generate"
        
    def run(self, job_description: str) -> Dict[str, Any]:
        """
        Parse and extract structured data from a job description
        """
        # Use Ollama to extract structured information from job description
        prompt = f"""
        Extract the following information from this job description. Return only JSON:
        1. Job title
        2. Required skills (as an array)
        3. Years of experience required
        4. Required education level
        5. Key responsibilities (as an array)
        6. Must-have technical skills (as an array)
        7. Nice-to-have skills (as an array)
        8. Keywords for matching (as an array)
        
        Job Description:
        {job_description}
        
        Return as valid JSON with these exact keys: title, skills, experience_years, education, responsibilities, 
        tech_skills, nice_to_have, keywords
        """
        
        # Call Ollama
        response = requests.post(
            self.llm_endpoint,
            json={"model": "llama3", "prompt": prompt, "stream": False}
        )
        
        if response.status_code == 200:
            try:
                # Extract the JSON content from the response
                content = response.json().get('response', '')
                # Find and extract the JSON part
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_content = content[json_start:json_end]
                    extracted_data = json.loads(json_content)
                    return extracted_data
                else:
                    return {"error": "No valid JSON found in response"}
            except Exception as e:
                return {"error": f"Failed to parse response: {str(e)}"}
        else:
            return {"error": f"API request failed with status {response.status_code}"}
    
    def save_job(self, job_data: Dict[str, Any]) -> int:
        """Save extracted job data to database"""
        # Prepare job data for database
        db_job = {
            "title": job_data["title"],
            "description": job_data.get("full_description", ""),
            "requirements": json.dumps(job_data.get("skills", []) + job_data.get("tech_skills", [])),
            "responsibilities": json.dumps(job_data.get("responsibilities", [])),
            "keywords": json.dumps(job_data.get("keywords", []))
        }
        
        # Insert into database
        job_id = self.db.insert("jobs", db_job)
        return job_id

# Resume/CV Parser Agent
class CVParserAgent(Agent):
    def _init_(self, db_connector=None):
        super()._init_(db_connector)
        self.llm_endpoint = "http://localhost:11434/api/generate"
        
    def run(self, resume_text: str) -> Dict[str, Any]:
        """
        Parse and extract structured data from a resume
        """
        # Use Ollama to extract structured information from resume
        prompt = f"""
        Extract the following information from this resume. Return only JSON:
        1. Full name
        2. Email address
        3. Phone number
        4. Education background (as an array of strings)
        5. Work experience (as an array of strings)
        6. Skills (as a string)
        7. Certifications (as a string)
        8. Notable achievements (as a string)
        9. Technologies used (as a string)
        10. Top 3-5 skills (as an array)
        
        Resume:
        {resume_text}
        
        Return as valid JSON with these exact keys: name, email, phone, education, experience, skills, 
        certifications, achievements, tech_stack, top_skills
        """
        
        # Call Ollama
        response = requests.post(
            self.llm_endpoint,
            json={"model": "llama3", "prompt": prompt, "stream": False}
        )
        
        if response.status_code == 200:
            try:
                # Extract the JSON content from the response
                content = response.json().get('response', '')
                # Find and extract the JSON part
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_content = content[json_start:json_end]
                    extracted_data = json.loads(json_content)
                    return extracted_data
                else:
                    return {"error": "No valid JSON found in response"}
            except Exception as e:
                return {"error": f"Failed to parse response: {str(e)}"}
        else:
            return {"error": f"API request failed with status {response.status_code}"}
    
    def save_candidate(self, candidate_data: Dict[str, Any], resume_text: str) -> int:
        """Save extracted candidate data to database"""
        # Generate an external ID if not provided
        if "external_id" not in candidate_data:
            candidate_data["external_id"] = f"C{datetime.now().strftime('%y%m%d%H%M%S')}"
            
        # Prepare candidate data for database
        db_candidate = {
            "external_id": candidate_data["external_id"],
            "name": candidate_data["name"],
            "email": candidate_data["email"],
            "phone": candidate_data.get("phone", ""),
            "education": json.dumps(candidate_data.get("education", [])),
            "experience": json.dumps(candidate_data.get("experience", [])),
            "skills": candidate_data.get("skills", ""),
            "certifications": candidate_data.get("certifications", ""),
            "achievements": candidate_data.get("achievements", ""),
            "tech_stack": candidate_data.get("tech_stack", ""),
            "top_skills": json.dumps(candidate_data.get("top_skills", [])),
            "resume_text": resume_text
        }
        
        # Insert into database
        candidate_id = self.db.insert("candidates", db_candidate)
        return candidate_id

# Matching Agent using semantic similarity
class MatchingAgent(Agent):
    def _init_(self, db_connector=None):
        super()._init_(db_connector)
        # Load sentence transformer model for semantic matching
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.llm_endpoint = "http://localhost:11434/api/generate"
        
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding vector for text"""
        return self.model.encode(text)
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def match_job_to_candidates(self, job_id: int, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Match job to all candidates and return those above threshold"""
        # Get job data
        job = self.db.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if not job:
            return []
            
        # Create job embedding
        job_text = f"{job['title']} {job['description']} {' '.join(json.loads(job['requirements']))} {' '.join(json.loads(job['keywords']))}"
        job_embedding = self.get_embedding(job_text)
        
        # Get all candidates
        candidates = self.db.execute("SELECT * FROM candidates").fetchall()
        matches = []
        
        for candidate in candidates:
            # Create candidate embedding
            candidate_text = f"{candidate['name']} {' '.join(json.loads(candidate['education']))} {' '.join(json.loads(candidate['experience']))} {candidate['skills']} {candidate['certifications']} {candidate['tech_stack']}"
            candidate_embedding = self.get_embedding(candidate_text)
            
            # Calculate semantic similarity
            similarity = self.cosine_similarity(job_embedding, candidate_embedding)
            
            # If above threshold, add to matches
            if similarity >= threshold:
                # Get detailed match breakdown using LLM
                details = self.get_match_details(job, candidate)
                
                match_data = {
                    "job_id": job_id,
                    "candidate_id": candidate["candidate_id"],
                    "match_score": float(similarity * 100),  # Convert to percentage
                    "details": json.dumps(details)
                }
                
                # Save match to database
                match_id = self.db.insert("matches", match_data)
                
                # Add to results
                match_data["match_id"] = match_id
                match_data["candidate_name"] = candidate["name"]
                match_data["candidate_email"] = candidate["email"]
                matches.append(match_data)
        
        # Sort by match score descending
        return sorted(matches, key=lambda x: x["match_score"], reverse=True)
    
    def get_match_details(self, job: Dict, candidate: Dict) -> Dict[str, Any]:
        """Get detailed breakdown of match using LLM"""
        job_requirements = json.loads(job["requirements"])
        candidate_skills = candidate["skills"]
        candidate_experience = json.loads(candidate["experience"])
        
        # Use Ollama to analyze the match in detail
        prompt = f"""
        Analyze how well this candidate matches the job requirements. Provide percentages and reasoning.
        
        Job title: {job["title"]}
        Job requirements: {', '.join(job_requirements)}
        
        Candidate skills: {candidate_skills}
        Candidate experience: {', '.join(candidate_experience)}
        
        Return as valid JSON with these exact keys:
        1. skill_match_percentage: percentage of required skills the candidate has
        2. experience_match_percentage: how well the experience matches job needs
        3. overall_fit_percentage: overall match percentage
        4. strengths: array of candidate's strengths for this position
        5. gaps: array of areas where candidate may need development
        """
        
        # Call Ollama
        response = requests.post(
            self.llm_endpoint,
            json={"model": "llama3", "prompt": prompt, "stream": False}
        )
        
        if response.status_code == 200:
            try:
                # Extract the JSON content from the response
                content = response.json().get('response', '')
                # Find and extract the JSON part
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_content = content[json_start:json_end]
                    return json.loads(json_content)
                else:
                    return {"error": "No valid JSON found in response"}
            except Exception as e:
                return {"error": f"Failed to parse response: {str(e)}"}
        else:
            return {"error": f"API request failed with status {response.status_code}"}

# Communication Agent
class CommunicationAgent(Agent):
    def _init_(self, db_connector=None):
        super()._init_(db_connector)
        self.llm_endpoint = "http://localhost:11434/api/generate"
        
    def generate_interview_email(self, job_id: int, candidate_id: int) -> Dict[str, str]:
        """Generate a personalized interview request email"""
        # Get job and candidate data
        job = self.db.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        candidate = self.db.execute("SELECT * FROM candidates WHERE candidate_id = ?", (candidate_id,)).fetchone()
        
        if not job or not candidate:
            return {"error": "Job or candidate not found"}
            
        # Get match details if available
        match = self.db.execute(
            "SELECT * FROM matches WHERE job_id = ? AND candidate_id = ?", 
            (job_id, candidate_id)
        ).fetchone()
        
        match_details = json.loads(match["details"]) if match and match["details"] else {}
        
        # Use Ollama to generate personalized email
        prompt = f"""
        Generate a professional and personalized email inviting the candidate for a job interview.
        
        Job title: {job["title"]}
        Job description: {job["description"]}
        
        Candidate name: {candidate["name"]}
        Candidate skills: {candidate["skills"]}
        Candidate top skills: {', '.join(json.loads(candidate["top_skills"]))}
        
        Candidate strengths for this position: {', '.join(match_details.get("strengths", []))}
        
        Include the following in the email:
        1. A personalized greeting
        2. Mention of specific skills that make them a good fit
        3. Brief overview of the interview process
        4. A request to confirm availability for the interview
        5. Professional signature
        
        Return as JSON with these keys: subject, body
        """
        
        # Call Ollama
        response = requests.post(
            self.llm_endpoint,
            json={"model": "llama3", "prompt": prompt, "stream": False}
        )
        
        if response.status_code == 200:
            try:
                # Extract the JSON content from the response
                content = response.json().get('response', '')
                # Find and extract the JSON part
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_content = content[json_start:json_end]
                    email_data = json.loads(json_content)
                    
                    # Save communication to database
                    communication_data = {
                        "candidate_id": candidate_id,
                        "job_id": job_id,
                        "type": "email",
                        "direction": "outbound",
                        "content": email_data["body"],
                        "sent_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        "status": "draft"
                    }
                    
                    communication_id = self.db.insert("communications", communication_data)
                    
                    # Return email with additional data
                    return {
                        "subject": email_data.get("subject", f"Interview Invitation: {job['title']} Position"),
                        "body": email_data.get("body", ""),
                        "to": candidate["email"],
                        "communication_id": communication_id
                    }
                else:
                    return {"error": "No valid JSON found in response"}
            except Exception as e:
                return {"error": f"Failed to parse response: {str(e)}"}
        else:
            return {"error": f"API request failed with status {response.status_code}"}
    
    def schedule_interview(self, job_id: int, candidate_id: int, date_time: str) -> int:
        """Schedule an interview and save to database"""
        interview_data = {
            "job_id": job_id,
            "candidate_id": candidate_id,
            "status": "scheduled",
            "scheduled_date": date_time,
            "notes": ""
        }
        
        interview_id = self.db.insert("interviews", interview_data)
        return interview_id

# Job Screening System - Main coordinator
class JobScreeningSystem:
    def _init_(self, db_path="job_screening.db"):
        self.db = DatabaseConnector(db_path)
        self.job_agent = JobDescriptionAgent(self.db)
        self.cv_agent = CVParserAgent(self.db)
        self.matching_agent = MatchingAgent(self.db)
        self.communication_agent = CommunicationAgent(self.db)
        
    def setup_database(self):
        """Initialize database schema if not exists"""
        # Read schema from file
        with open("schema.sql", "r") as f:
            schema = f.read()
            
        # Execute schema script
        self.db.connect()
        self.db.conn.executescript(schema)
        self.db.close()
        
    def process_job(self, job_description: str) -> int:
        """Process a new job description"""
        # Extract structured data from job description
        job_data = self.job_agent.run(job_description)
        
        # Save to database
        job_id = self.job_agent.save_job(job_data)
        return job_id
        
    def process_resume(self, resume_text: str) -> int:
        """Process a new resume"""
        # Extract structured data from resume
        candidate_data = self.cv_agent.run(resume_text)
        
        # Save to database
        candidate_id = self.cv_agent.save_candidate(candidate_data, resume_text)
        return candidate_id
        
    def find_matches(self, job_id: int, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Find candidates matching a job"""
        return self.matching_agent.match_job_to_candidates(job_id, threshold)
        
    def send_interview_request(self, job_id: int, candidate_id: int) -> Dict[str, str]:
        """Generate and send interview request"""
        email_data = self.communication_agent.generate_interview_email(job_id, candidate_id)
        # In a real system, this would connect to an email service
        # For now, we just return the email content
        return email_data
    
    def schedule_interview(self, job_id: int, candidate_id: int, date_time: str) -> int:
        """Schedule an interview"""
        return self.communication_agent.schedule_interview(job_id, candidate_id, date_time)

# Example usage
if _name_ == "_main_":
    system = JobScreeningSystem()
    system.setup_database()
    
    # Process a job
    job_description = """
    Software Engineer
    
    We are seeking a skilled Software Engineer to design, develop, and maintain software applications. 
    The ideal candidate will write efficient code, troubleshoot issues, and collaborate with teams to 
    deliver high-quality solutions.
    
    Requirements:
    - Bachelor's degree in Computer Science or a related field
    - Proficiency in programming languages like Python, Java, or C++
    - Experience with databases, web development, and software frameworks
    - Strong problem-solving skills and attention to detail
    - Ability to work both independently and in a team environment
    
    Responsibilities:
    - Develop, test, and deploy software applications
    - Write clean, maintainable, and scalable code
    - Collaborate with cross-functional teams to define and implement features
    - Troubleshoot and debug issues for optimal performance
    - Stay updated with emerging technologies and best practices
    """
    
    job_id = system.process_job(job_description)
    print(f"Job processed with ID: {job_id}")
    
    # Process a resume
    resume_text = """
    John Doe
    john.doe@example.com
    555-123-4567
    
    Education:
    - M.S. Computer Science, Stanford University (2018-2020)
    - B.S. Computer Engineering, MIT (2014-2018)
    
    Experience:
    - Software Engineer, Google (2020-Present)
      Developed web applications using React and Node.js
      Optimized database queries for improved performance
      Implemented CI/CD pipelines for automated testing and deployment
    
    - Software Developer Intern, Microsoft (Summer 2019)
      Worked on Azure cloud services
      Developed REST APIs using .NET Core
    
    Skills:
    Python, Java, JavaScript, React, Node.js, SQL, AWS, Docker, Kubernetes
    
    Certifications:
    AWS Certified Developer, MongoDB Certified Developer
    
    Projects:
    - Built a machine learning model for predicting stock prices
    - Developed an open-source library for data visualization
    """
    
    candidate_id = system.process_resume(resume_text)
    print(f"Resume processed with ID: {candidate_id}")
    
    # Find matches
    matches = system.find_matches(job_id, threshold=0.6)
    print(f"Found {len(matches)} matches")
    
    if matches:
        # Send interview request to top match
        top_match = matches[0]
        email = system.send_interview_request(job_id, top_match["candidate_id"])
        print(f"Interview email created: {email['subject']}")
        
        # Schedule interview
        interview_id = system.schedule_interview(
            job_id, 
            top_match["candidate_id"],
            datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        )
        print(f"Interview scheduled with ID: {interview_id}")