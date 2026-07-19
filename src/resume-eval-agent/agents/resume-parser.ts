/**
 * @fileoverview Resume Parsing Agent
 * Extracts structured candidate information from resume text using Groq API
 */

import 'dotenv/config';
import Groq from 'groq-sdk';
import { CandidateData } from '../schemas/candidate';
import { extractJSONFromText, parseAndValidateJSON, validateCandidateData } from '../utils/validators';
import { truncateText } from '../utils/text-extractor';

const groq = new Groq({
  apiKey: process.env.GROQ_API_KEY,
});

/**
 * Parse resume text and extract structured candidate data
 * Uses Claude to intelligently extract candidate information
 * @param resumeText - Raw resume text
 * @param maxResumeChars - Maximum characters to send to LLM (default: 4000)
 * @returns Extracted candidate data
 * @throws Error if resume parsing or validation fails
 */
export async function parseResume(
  resumeText: string,
  maxResumeChars: number = parseInt(process.env.MAX_RESUME_CHARS || '4000')
): Promise<CandidateData> {
  // Truncate resume to avoid excessive token usage
  const truncatedResume = truncateText(resumeText, maxResumeChars);

  const prompt = `You are an expert resume parser. Extract the following information from the resume and return ONLY a valid JSON object (no markdown, no extra text, no code blocks).

RESUME TEXT:
${truncatedResume}

Extract and return this exact JSON structure:
{
  "candidate_name": "string (full name)",
  "email": "string or null if not found",
  "phone": "string or null if not found (format as provided)",
  "total_experience": "string (e.g., '5 years' or 'Unknown')",
  "education": "string (e.g., 'B.Tech Computer Science, IIT Delhi')",
  "skills": ["skill1", "skill2", ...],
  "projects": ["project1", "project2", ...],
  "certifications": ["cert1", "cert2", ...]
}

Rules:
- skills: Extract only technical skills, maximum 15 items
- projects: Extract 3-5 most impactful projects
- education: Include degree and institution if available
- If any field is not found, use null or empty array as appropriate
- Return ONLY the JSON object, nothing else`;

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

    // Extract JSON from response (handles markdown code blocks)
    const jsonString = extractJSONFromText(responseText);

    // Parse and validate
    const candidateData = parseAndValidateJSON(jsonString, validateCandidateData);

    return candidateData;
  } catch (error) {
    throw new Error(
      `Resume parsing failed: ${(error as Error).message}`
    );
  }
}

/**
 * Get LLM execution details (for debugging)
 * @returns Human-readable status message
 */
export function getResumeParserInfo(): string {
  return 'Resume Parsing Agent v1.0 - Groq Mixtral-based extraction with validation';
}
