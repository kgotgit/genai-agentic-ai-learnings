/**
 * @fileoverview Candidate Evaluation Agent
 * Compares candidate profile against Job Description and generates evaluation (Groq)
 */

import 'dotenv/config';
import Groq from 'groq-sdk';
import { CandidateData } from '../schemas/candidate';
import { EvaluationResult } from '../schemas/evaluation';
import { extractJSONFromText, parseAndValidateJSON, validateEvaluationResult } from '../utils/validators';
import { extractSkillsFromText, truncateText } from '../utils/text-extractor';

const groq = new Groq({
  apiKey: process.env.GROQ_API_KEY,
});

/**
 * Evaluate candidate against job description
 * Compares skills, experience, and projects to determine hiring decision
 * @param candidateData - Parsed candidate information
 * @param jobDescriptionText - Full job description text
 * @param maxJDChars - Maximum JD characters to send to LLM (default: 3000)
 * @returns Evaluation result with scoring and recommendation
 * @throws Error if evaluation or validation fails
 */
export async function evaluateCandidate(
  candidateData: CandidateData,
  jobDescriptionText: string,
  maxJDChars: number = parseInt(process.env.MAX_JD_CHARS || '3000')
): Promise<EvaluationResult> {
  // Truncate JD to avoid excessive token usage
  const truncatedJD = truncateText(jobDescriptionText, maxJDChars);

  const prompt = `You are an expert recruiter evaluating candidates. Compare the candidate profile against the job requirements and return ONLY a valid JSON object (no markdown, no extra text).

CANDIDATE DATA:
${JSON.stringify(candidateData, null, 2)}

JOB DESCRIPTION:
${truncatedJD}

Analyze and return this exact JSON structure:
{
  "candidate_name": "string (from candidate data)",
  "decision": "Selected" or "Not Selected",
  "matching_score": number (0-100, where 100 is perfect match),
  "matched_skills": ["skill1", "skill2", ...],
  "missing_skills": ["skill1", "skill2", ...],
  "strengths": ["strength1", "strength2", ...],
  "weaknesses": ["weakness1", "weakness2", ...],
  "reason": "string (1-2 sentences explaining the decision)"
}

Evaluation criteria:
- matching_score: Calculate based on skill overlap, experience alignment, and project relevance
- decision: "Selected" if 80+ score AND no critical mandatory skills missing, else "Not Selected"
- matched_skills: Skills candidate has that are required/preferred by JD (max 10)
- missing_skills: Required/preferred skills from JD that candidate lacks (max 10)
- strengths: 3-4 strongest aspects of candidate for this role
- weaknesses: 2-3 main gaps or concerns
- reason: Brief, professional explanation focusing on mandatory requirements

Return ONLY the JSON object, nothing else`;

  try {
    const message = await groq.chat.completions.create({
      model: 'mixtral-8x7b-32768',
      temperature: 0.2,
      max_tokens: 1024,
      messages: [
        {
          role: 'user',
          content: prompt,
        },
      ],
    });

    const responseText = message.choices[0]?.message?.content || '';

    // Extract JSON from response
    const jsonString = extractJSONFromText(responseText);

    // Parse and validate
    const evaluation = parseAndValidateJSON(jsonString, validateEvaluationResult);

    return evaluation;
  } catch (error) {
    throw new Error(
      `Candidate evaluation failed: ${(error as Error).message}`
    );
  }
}

/**
 * Get agent info
 * @returns Human-readable status message
 */
export function getEvaluatorInfo(): string {
  return 'Candidate Evaluation Agent v1.0 - Groq Mixtral-based comparison and scoring';
}
