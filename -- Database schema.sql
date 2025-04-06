-- Database schema for Job Screening Application

-- Jobs table
CREATE TABLE jobs (
    job_id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    requirements TEXT NOT NULL, -- Stored as JSON array
    responsibilities TEXT NOT NULL, -- Stored as JSON array
    keywords TEXT NOT NULL, -- Stored as JSON array
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Candidates table
CREATE TABLE candidates (
    candidate_id INTEGER PRIMARY KEY AUTOINCREMENT,
    external_id TEXT UNIQUE, -- For reference to external systems
    name TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE,
    phone TEXT,
    education TEXT NOT NULL, -- Stored as JSON array
    experience TEXT NOT NULL, -- Stored as JSON array
    skills TEXT NOT NULL,
    certifications TEXT,
    achievements TEXT,
    tech_stack TEXT,
    top_skills TEXT NOT NULL, -- Stored as JSON array
    resume_text TEXT, -- Full text of resume for searching
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Matches table to store results
CREATE TABLE matches (
    match_id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id INTEGER NOT NULL,
    candidate_id INTEGER NOT NULL,
    match_score REAL NOT NULL, -- Percentage score
    details TEXT, -- JSON with detailed scoring breakdown
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (job_id) REFERENCES jobs (job_id),
    FOREIGN KEY (candidate_id) REFERENCES candidates (candidate_id)
);

-- Interview scheduling table
CREATE TABLE interviews (
    interview_id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id INTEGER NOT NULL,
    candidate_id INTEGER NOT NULL,
    status TEXT NOT NULL, -- 'scheduled', 'completed', 'cancelled'
    scheduled_date TIMESTAMP,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (job_id) REFERENCES jobs (job_id),
    FOREIGN KEY (candidate_id) REFERENCES candidates (candidate_id)
);

-- Communication history table
CREATE TABLE communications (
    communication_id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id INTEGER NOT NULL,
    job_id INTEGER,
    type TEXT NOT NULL, -- 'email', 'phone', etc.
    direction TEXT NOT NULL, -- 'outbound', 'inbound'
    content TEXT NOT NULL,
    sent_at TIMESTAMP,
    status TEXT, -- 'sent', 'delivered', 'opened', 'replied'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (candidate_id) REFERENCES candidates (candidate_id),
    FOREIGN KEY (job_id) REFERENCES jobs (job_id)
);

-- Create indexes for frequently queried fields
CREATE INDEX idx_jobs_title ON jobs(title);
CREATE INDEX idx_candidates_skills ON candidates(skills);
CREATE INDEX idx_matches_score ON matches(match_score);