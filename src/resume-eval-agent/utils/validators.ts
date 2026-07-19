/**
 * @fileoverview Schema validation utilities
 * Validates structured outputs from LLM agents against defined schemas
 */

import { CandidateData, isCandidateData } from '../schemas/candidate';
import { EvaluationResult, isEvaluationResult } from '../schemas/evaluation';
import { EmailOutput, isEmailOutput } from '../schemas/email';

/**
 * Custom validation error class
 */
export class ValidationError extends Error {
  constructor(
    public readonly fieldPath: string,
    message: string,
    public readonly receivedValue?: unknown
  ) {
    super(`Validation failed at ${fieldPath}: ${message}`);
    this.name = 'ValidationError';
  }
}

/**
 * Validate and parse candidate data from LLM response
 * @param data - Data to validate (typically parsed JSON from LLM)
 * @returns Valid CandidateData if validation passes
 * @throws ValidationError if data doesn't match schema
 */
export function validateCandidateData(data: unknown): CandidateData {
  if (!isCandidateData(data)) {
    throw new ValidationError(
      'CandidateData',
      'Input does not match CandidateData schema',
      data
    );
  }

  // Additional validation rules
  if (!data.candidate_name || data.candidate_name.trim().length === 0) {
    throw new ValidationError(
      'candidate_name',
      'Candidate name is required and cannot be empty'
    );
  }

  if (!Array.isArray(data.skills) || data.skills.length === 0) {
    throw new ValidationError(
      'skills',
      'Skills array is required and must contain at least one item'
    );
  }

  return data;
}

/**
 * Validate and parse evaluation result from LLM response
 * @param data - Data to validate (typically parsed JSON from LLM)
 * @returns Valid EvaluationResult if validation passes
 * @throws ValidationError if data doesn't match schema
 */
export function validateEvaluationResult(data: unknown): EvaluationResult {
  if (!isEvaluationResult(data)) {
    throw new ValidationError(
      'EvaluationResult',
      'Input does not match EvaluationResult schema',
      data
    );
  }

  // Additional validation rules
  if (!data.reason || data.reason.trim().length === 0) {
    throw new ValidationError(
      'reason',
      'Reason is required for the hiring decision'
    );
  }

  // Consistency check: if decided "Selected", should have matched skills
  if (data.decision === 'Selected' && data.matched_skills.length === 0) {
    throw new ValidationError(
      'matched_skills',
      'Selected candidates should have at least one matched skill'
    );
  }

  return data;
}

/**
 * Validate and parse email output from LLM response
 * @param data - Data to validate (typically parsed JSON from LLM)
 * @returns Valid EmailOutput if validation passes
 * @throws ValidationError if data doesn't match schema
 */
export function validateEmailOutput(data: unknown): EmailOutput {
  if (!isEmailOutput(data)) {
    throw new ValidationError(
      'EmailOutput',
      'Input does not match EmailOutput schema',
      data
    );
  }

  // Additional validation rules
  if (!data.subject || data.subject.trim().length === 0) {
    throw new ValidationError(
      'subject',
      'Email subject is required and cannot be empty'
    );
  }

  if (!data.body || data.body.trim().length === 0) {
    throw new ValidationError(
      'body',
      'Email body is required and cannot be empty'
    );
  }

  // Basic email format validation
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  if (!emailRegex.test(data.to_address)) {
    throw new ValidationError(
      'to_address',
      `Invalid email format: ${data.to_address}`
    );
  }

  return data;
}

/**
 * Parse JSON string safely, with validation
 * @param jsonString - JSON string to parse
 * @param validator - Validation function to apply after parsing
 * @returns Validated parsed object
 * @throws Error if JSON is invalid or validation fails
 */
export function parseAndValidateJSON<T>(
  jsonString: string,
  validator: (data: unknown) => T
): T {
  try {
    const parsed = JSON.parse(jsonString);
    return validator(parsed);
  } catch (error) {
    if (error instanceof ValidationError) {
      throw error;
    }
    throw new Error(`Failed to parse JSON: ${(error as Error).message}`);
  }
}

/**
 * Extract JSON from text that may contain markdown code blocks or extra text
 * @param text - Text that may contain JSON
 * @returns Extracted JSON string
 * @throws Error if no valid JSON found
 */
export function extractJSONFromText(text: string): string {
  // Try to find JSON in markdown code block first
  const codeBlockMatch = text.match(/```(?:json)?\s*([\s\S]*?)\s*```/);
  if (codeBlockMatch) {
    return codeBlockMatch[1];
  }

  // Try to find JSON object by matching braces
  const jsonMatch = text.match(/\{[\s\S]*\}/);
  if (jsonMatch) {
    return jsonMatch[0];
  }

  throw new Error('No valid JSON found in text');
}
