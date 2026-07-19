Agentic AI Problem Statement: AI-Powered Resume Screening & Candidate Communication
Problem Title
Multi-Agent AI Recruitment Assistant for Resume Screening and Personalized Candidate Communication
Business Context
A recruitment team receives hundreds of resumes for every job opening. Recruiters spend significant time reviewing resumes, matching candidate skills with the Job Description (JD), deciding whether to shortlist candidates, and drafting personalized emails. This manual process is time-consuming, inconsistent, and prone to human bias.
Your task is to build an Agentic AI Recruitment Assistant where multiple AI agents collaborate to automate the end-to-end candidate screening workflow.
 
Problem Statement
Develop an Agentic AI application that accepts a Job Description (JD) and a candidate's resume as input. Multiple AI agents should collaborate to extract candidate information, evaluate the candidate against the JD, make a hiring recommendation, and generate a personalized email.
Each agent should perform a specialized task and pass structured outputs to the next agent.
 
Inputs
•	Job Description (PDF/DOCX/TXT) 
•	Candidate Resume (PDF/DOCX) 
 
Expected Workflow
                +----------------------+
                | Resume               |
                +----------+-----------+
                           |
                           v
                  Resume Parsing Agent
                           |
                           v
                Structured Candidate Data
                           |
                           |
Job Description ---------->|
                           v
                  Candidate Evaluation Agent
                           |
                           v
                 Selection Recommendation
                           |
                           v
                Email Generation Agent
                           |
                           v
                 Personalized Candidate Email
 
Agent 1 – Resume Parsing Agent
Responsibility
Extract structured information from the resume.
Input
•	Resume 
Output Schema
{
  "candidate_name": "",
  "email": "",
  "phone": "",
  "total_experience": "",
  "education": "",
  "skills": [
  ],
  "projects": [
  ],
  "certifications": [
  ]
}
Expected Capabilities
•	Read PDF/DOCX resume 
•	Extract candidate details 
•	Identify technical skills 
•	Extract project names 
•	Calculate total experience 
•	Return structured JSON 
 
Agent 2 – Candidate Evaluation Agent
Responsibility
Compare the candidate profile against the Job Description.
Input
•	Job Description 
•	Skills 
•	Experience 
•	Projects 
Output
{
  "candidate_name":"Harish Kumar",
  "decision":"Selected",

  "matching_score":92,

  "matched_skills":[
      "Python",
      "SQL",
      "Machine Learning",
      "LLM"
  ],

  "missing_skills":[
      "Docker"
  ],

  "strengths":[
      "Strong Banking Domain",
      "Good AI Experience"
  ],

  "weaknesses":[
      "No Docker Experience"
  ],

  "reason":"Candidate satisfies most mandatory skills and has relevant domain experience."
}
If not selected
{
  "decision":"Not Selected",

  "reason":"Mandatory AWS experience missing."
}
 
Agent 3 – Personalized Email Agent
Responsibility
Generate a professional email based on the evaluation result.
Input
•	Candidate Name 
•	Evaluation Result 
•	Reason 
Output (Selected)
Subject:
Congratulations! Your Profile Has Been Shortlisted

Dear Harish,

Thank you for applying for the AI Engineer position.

We reviewed your profile and were impressed with your experience in Python, Machine Learning, and Banking Domain projects.

We are pleased to inform you that you have been shortlisted for the next round of interviews.

Our recruitment team will contact you shortly with further details.

Best Regards,
HR Team
 
Output (Rejected)
Subject:
Update Regarding Your Application

Dear Harish,

Thank you for taking the time to apply for the AI Engineer position.

After reviewing your profile against the job requirements, we found that your experience does not currently match some of the mandatory requirements, particularly AWS and Docker experience.

We encourage you to apply again in the future as new opportunities become available.

We sincerely appreciate your interest in our organization.

Best Regards,
HR Team

