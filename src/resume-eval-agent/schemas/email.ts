/**
 * @fileoverview Email output schema and types
 * Defines the structured output from the Email Generation Agent
 */

/**
 * Personalized email generated for candidate communication
 * @interface EmailOutput
 */
export interface EmailOutput {
  /** Email subject line */
  subject: string;
  /** Email body content (plain text) */
  body: string;
  /** Recipient email address */
  to_address: string;
}

/**
 * Type guard to validate EmailOutput structure
 * @param data - Data to validate
 * @returns True if data matches EmailOutput interface
 */
export function isEmailOutput(data: unknown): data is EmailOutput {
  if (!data || typeof data !== 'object') return false;
  const obj = data as Record<string, unknown>;
  return (
    typeof obj.subject === 'string' &&
    typeof obj.body === 'string' &&
    typeof obj.to_address === 'string'
  );
}
