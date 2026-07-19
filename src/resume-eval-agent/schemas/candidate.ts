/**
 * @fileoverview Candidate data schema and types
 * Defines the structured output from the Resume Parsing Agent
 */

/**
 * Structured representation of candidate information extracted from resume
 * @interface CandidateData
 */
export interface CandidateData {
  /** Full name of the candidate */
  candidate_name: string;
  /** Email address (if available) */
  email: string | null;
  /** Phone number (if available) */
  phone: string | null;
  /** Total professional experience (e.g., "5 years", "3.5 years") */
  total_experience: string;
  /** Educational background (e.g., "B.Tech Computer Science, IIT Delhi") */
  education: string;
  /** List of technical skills (e.g., ["Python", "TypeScript", "AWS"]) */
  skills: string[];
  /** List of notable projects (e.g., ["ML Recommendation System", "Banking App"]) */
  projects: string[];
  /** List of certifications and qualifications */
  certifications: string[];
}

/**
 * Type guard to validate CandidateData structure
 * @param data - Data to validate
 * @returns True if data matches CandidateData interface
 */
export function isCandidateData(data: unknown): data is CandidateData {
  if (!data || typeof data !== 'object') return false;
  const obj = data as Record<string, unknown>;
  return (
    typeof obj.candidate_name === 'string' &&
    (obj.email === null || typeof obj.email === 'string') &&
    (obj.phone === null || typeof obj.phone === 'string') &&
    typeof obj.total_experience === 'string' &&
    typeof obj.education === 'string' &&
    Array.isArray(obj.skills) && obj.skills.every(s => typeof s === 'string') &&
    Array.isArray(obj.projects) && obj.projects.every(p => typeof p === 'string') &&
    Array.isArray(obj.certifications) && obj.certifications.every(c => typeof c === 'string')
  );
}
