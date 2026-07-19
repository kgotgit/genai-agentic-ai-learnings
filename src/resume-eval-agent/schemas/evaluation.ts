/**
 * @fileoverview Evaluation result schema and types
 * Defines the structured output from the Candidate Evaluation Agent
 */

/**
 * Structured evaluation result comparing candidate against Job Description
 * @interface EvaluationResult
 */
export interface EvaluationResult {
  /** Candidate name (from parsed resume) */
  candidate_name: string;
  /** Hiring decision: Selected or Not Selected */
  decision: 'Selected' | 'Not Selected';
  /** Matching score (0-100) indicating how well candidate matches JD */
  matching_score: number;
  /** Skills that match between candidate and JD requirements */
  matched_skills: string[];
  /** Skills required by JD but not present in candidate profile */
  missing_skills: string[];
  /** Candidate strengths relevant to the role */
  strengths: string[];
  /** Candidate weaknesses or gaps relevant to the role */
  weaknesses: string[];
  /** Brief explanation of decision and key reasoning */
  reason: string;
}

/**
 * Type guard to validate EvaluationResult structure
 * @param data - Data to validate
 * @returns True if data matches EvaluationResult interface
 */
export function isEvaluationResult(data: unknown): data is EvaluationResult {
  if (!data || typeof data !== 'object') return false;
  const obj = data as Record<string, unknown>;
  return (
    typeof obj.candidate_name === 'string' &&
    (obj.decision === 'Selected' || obj.decision === 'Not Selected') &&
    typeof obj.matching_score === 'number' &&
    obj.matching_score >= 0 &&
    obj.matching_score <= 100 &&
    Array.isArray(obj.matched_skills) &&
    obj.matched_skills.every(s => typeof s === 'string') &&
    Array.isArray(obj.missing_skills) &&
    obj.missing_skills.every(s => typeof s === 'string') &&
    Array.isArray(obj.strengths) &&
    obj.strengths.every(s => typeof s === 'string') &&
    Array.isArray(obj.weaknesses) &&
    obj.weaknesses.every(w => typeof w === 'string') &&
    typeof obj.reason === 'string'
  );
}
