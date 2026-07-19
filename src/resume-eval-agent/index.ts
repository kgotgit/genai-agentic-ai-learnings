/**
 * @fileoverview Resume Evaluation Agent - Main entry point
 * Exports all public APIs for programmatic usage
 */

export { CandidateData, isCandidateData } from './schemas/candidate';
export { EvaluationResult, isEvaluationResult } from './schemas/evaluation';
export { EmailOutput, isEmailOutput } from './schemas/email';

export {
  PipelineInput,
  PipelineOutput,
  PipelineStep,
  runResumeEvaluationPipeline,
  formatPipelineOutput,
} from './orchestrator';

export { parseResume, getResumeParserInfo } from './agents/resume-parser';
export { evaluateCandidate, getEvaluatorInfo } from './agents/evaluator';
export { generateEmail, getEmailGeneratorInfo } from './agents/email-generator';

export {
  loadDocument,
  loadPDF,
  loadDOCX,
  loadTextFile,
  validateFilePath,
} from './utils/document-loader';

export {
  extractEmail,
  extractPhoneNumber,
  extractTotalExperience,
  normalizeText,
  splitSectionsByHeaders,
  truncateText,
  extractSkillsFromText,
} from './utils/text-extractor';

export {
  ValidationError,
  validateCandidateData,
  validateEvaluationResult,
  validateEmailOutput,
  parseAndValidateJSON,
  extractJSONFromText,
} from './utils/validators';
