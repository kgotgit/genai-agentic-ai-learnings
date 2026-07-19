/**
 * @fileoverview Email Generation Agent
 * Generates personalized candidate communication emails based on evaluation (Groq)
 */
import 'dotenv/config';import Groq from 'groq-sdk';
import { EvaluationResult } from '../schemas/evaluation';
import { EmailOutput } from '../schemas/email';
import { extractJSONFromText, parseAndValidateJSON, validateEmailOutput } from '../utils/validators';

const groq = new Groq({
  apiKey: process.env.GROQ_API_KEY,
});

/**
 * Generate personalized email for candidate
 * Creates professional acceptance or rejection email based on evaluation
 * @param evaluation - Evaluation result from candidate evaluation agent
 * @returns Generated email with subject and body
 * @throws Error if email generation or validation fails
 */
export async function generateEmail(evaluation: EvaluationResult): Promise<EmailOutput> {
  const isSelected = evaluation.decision === 'Selected';

  const prompt = `You are a professional HR email writer. Generate a personalized email for a candidate based on their evaluation result.

EVALUATION RESULT:
${JSON.stringify(evaluation, null, 2)}

Generate and return ONLY a valid JSON object (no markdown, no extra text):
{
  "subject": "string (email subject line)",
  "body": "string (email body in plain text, use \\n for line breaks)",
  "to_address": "string (recipient email)"
}

${isSelected ? `SELECTED CANDIDATE EMAIL:
Guidelines:
- Subject: Start with "Congratulations"
- Mention 2-3 of their matched skills positively
- Reference strengths from evaluation
- Inform about next interview round
- Professional, warm tone
- 4-5 short paragraphs
- Include "Best Regards, HR Team" closing

Example structure:
1. Thank you for applying paragraph
2. We were impressed by paragraph (mention matched skills and strengths)
3. Pleased to inform you of shortlist paragraph
4. Next steps paragraph
5. Closing

Make it specific to this candidate's profile.` : `NOT SELECTED CANDIDATE EMAIL:
Guidelines:
- Subject: Professional update about application
- Acknowledge their effort and interest
- Explain missing mandatory skills (reference missing_skills from evaluation)
- Professional, respectful tone
- Encourage future applications
- 4-5 short paragraphs
- Include "Best Regards, HR Team" closing

Example structure:
1. Thank you for applying paragraph
2. We reviewed your profile paragraph
- Found that experience doesn't match requirements
3. Missing mandatory skills explanation (use missing_skills)
4. Encouragement for future applications
5. Closing

Be empathetic while being clear about the gap.`}

Use candidate name: ${evaluation.candidate_name}
Use candidate email: ${evaluation.candidate_name.toLowerCase().replace(/\s+/g, '.')}@example.com (if not provided elsewhere)

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
    const emailOutput = parseAndValidateJSON(jsonString, validateEmailOutput);

    return emailOutput;
  } catch (error) {
    throw new Error(
      `Email generation failed: ${(error as Error).message}`
    );
  }
}

/**
 * Get agent info
 * @returns Human-readable status message
 */
export function getEmailGeneratorInfo(): string {
  return 'Email Generation Agent v1.0 - Groq Mixtral-based personalized communication';
}
