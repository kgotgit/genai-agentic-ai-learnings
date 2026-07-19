/**
 * @fileoverview Orchestrator for resume evaluation pipeline
 * Coordinates sequential execution of all three agents with error handling
 */

import { CandidateData } from './schemas/candidate';
import { EvaluationResult } from './schemas/evaluation';
import { EmailOutput } from './schemas/email';
import { parseResume } from './agents/resume-parser';
import { evaluateCandidate } from './agents/evaluator';
import { generateEmail } from './agents/email-generator';
import { loadDocument, validateFilePath } from './utils/document-loader';

/**
 * Represents a single step in the pipeline execution
 * @interface PipelineStep
 */
export interface PipelineStep {
  /** Name of the agent or step */
  agentName: string;
  /** Execution status: ok, error, or skipped */
  status: 'ok' | 'error' | 'skipped';
  /** ISO timestamp when step started */
  startedAt: string;
  /** Duration of execution in milliseconds */
  durationMs: number;
  /** Detail message or error description */
  detail: string;
}

/**
 * Complete pipeline input
 * @interface PipelineInput
 */
export interface PipelineInput {
  /** Path to resume file (PDF, DOCX, or TXT) */
  resumeFilePath: string;
  /** Path to job description file (PDF, DOCX, or TXT) */
  jobDescriptionPath: string;
}

/**
 * Complete pipeline output with all agent results
 * @interface PipelineOutput
 */
export interface PipelineOutput {
  /** Extracted candidate data from resume parser */
  candidateData: CandidateData | null;
  /** Evaluation result from candidate evaluator */
  evaluationResult: EvaluationResult | null;
  /** Generated email from email generator */
  emailOutput: EmailOutput | null;
  /** Total pipeline execution time in milliseconds */
  executionTime: number;
  /** Detailed trace of all pipeline steps */
  trace: PipelineStep[];
  /** Overall pipeline success status */
  success: boolean;
  /** Error message if pipeline failed */
  error?: string;
}

/**
 * Execute the complete resume evaluation pipeline
 * Sequentially runs: Resume Parser → Evaluator → Email Generator
 * @param input - Pipeline input (resume and JD paths)
 * @returns Pipeline output with all results and execution trace
 */
export async function runResumeEvaluationPipeline(input: PipelineInput): Promise<PipelineOutput> {
  const startTime = Date.now();
  const trace: PipelineStep[] = [];

  let candidateData: CandidateData | null = null;
  let evaluationResult: EvaluationResult | null = null;
  let emailOutput: EmailOutput | null = null;

  try {
    // Validate input files
    console.log('Validating input files...');
    validateFilePath(input.resumeFilePath);
    validateFilePath(input.jobDescriptionPath);

    // Step 1: Load and parse resume
    console.log('Step 1: Loading resume...');
    let stepStart = Date.now();
    try {
      const resumeText = await loadDocument(input.resumeFilePath);
      console.log('Step 1.1: Parsing resume with Agent 1...');
      candidateData = await parseResume(resumeText);

      trace.push({
        agentName: 'Resume Parser',
        status: 'ok',
        startedAt: new Date(stepStart).toISOString(),
        durationMs: Date.now() - stepStart,
        detail: `Successfully extracted data for ${candidateData.candidate_name}`,
      });
    } catch (error) {
      const duration = Date.now() - stepStart;
      trace.push({
        agentName: 'Resume Parser',
        status: 'error',
        startedAt: new Date(stepStart).toISOString(),
        durationMs: duration,
        detail: (error as Error).message,
      });
      throw error;
    }

    // Step 2: Load JD and evaluate candidate
    console.log('Step 2: Loading job description...');
    stepStart = Date.now();
    try {
      const jdText = await loadDocument(input.jobDescriptionPath);
      console.log('Step 2.1: Evaluating candidate with Agent 2...');
      evaluationResult = await evaluateCandidate(candidateData, jdText);

      trace.push({
        agentName: 'Candidate Evaluator',
        status: 'ok',
        startedAt: new Date(stepStart).toISOString(),
        durationMs: Date.now() - stepStart,
        detail: `Decision: ${evaluationResult.decision} (Score: ${evaluationResult.matching_score})`,
      });
    } catch (error) {
      const duration = Date.now() - stepStart;
      trace.push({
        agentName: 'Candidate Evaluator',
        status: 'error',
        startedAt: new Date(stepStart).toISOString(),
        durationMs: duration,
        detail: (error as Error).message,
      });
      throw error;
    }

    // Step 3: Generate email
    console.log('Step 3: Generating email with Agent 3...');
    stepStart = Date.now();
    try {
      emailOutput = await generateEmail(evaluationResult);

      trace.push({
        agentName: 'Email Generator',
        status: 'ok',
        startedAt: new Date(stepStart).toISOString(),
        durationMs: Date.now() - stepStart,
        detail: `Generated ${evaluationResult.decision === 'Selected' ? 'acceptance' : 'rejection'} email`,
      });
    } catch (error) {
      const duration = Date.now() - stepStart;
      trace.push({
        agentName: 'Email Generator',
        status: 'error',
        startedAt: new Date(stepStart).toISOString(),
        durationMs: duration,
        detail: (error as Error).message,
      });
      throw error;
    }

    // All steps completed successfully
    const executionTime = Date.now() - startTime;
    console.log(`Pipeline completed successfully in ${executionTime}ms`);

    return {
      candidateData,
      evaluationResult,
      emailOutput,
      executionTime,
      trace,
      success: true,
    };
  } catch (error) {
    const executionTime = Date.now() - startTime;
    const errorMessage = (error as Error).message;

    console.error(`Pipeline failed: ${errorMessage}`);

    return {
      candidateData,
      evaluationResult,
      emailOutput,
      executionTime,
      trace,
      success: false,
      error: errorMessage,
    };
  }
}

/**
 * Format pipeline output for display
 * @param output - Pipeline output
 * @returns Formatted string representation
 */
export function formatPipelineOutput(output: PipelineOutput): string {
  const lines: string[] = [];

  lines.push(`\n${'='.repeat(60)}`);
  lines.push(`Resume Evaluation Pipeline - ${output.success ? 'SUCCESS' : 'FAILED'}`);
  lines.push(`${'='.repeat(60)}\n`);

  // Candidate Data
  if (output.candidateData) {
    lines.push('📋 CANDIDATE DATA:');
    lines.push(`  Name: ${output.candidateData.candidate_name}`);
    lines.push(`  Email: ${output.candidateData.email || 'N/A'}`);
    lines.push(`  Phone: ${output.candidateData.phone || 'N/A'}`);
    lines.push(`  Experience: ${output.candidateData.total_experience}`);
    lines.push(`  Education: ${output.candidateData.education}`);
    lines.push(`  Skills: ${output.candidateData.skills.join(', ')}`);
    lines.push(`  Projects: ${output.candidateData.projects.join(', ')}`);
    lines.push('');
  }

  // Evaluation Result
  if (output.evaluationResult) {
    lines.push('🎯 EVALUATION RESULT:');
    lines.push(`  Decision: ${output.evaluationResult.decision}`);
    lines.push(`  Matching Score: ${output.evaluationResult.matching_score}/100`);
    lines.push(`  Matched Skills: ${output.evaluationResult.matched_skills.join(', ')}`);
    lines.push(`  Missing Skills: ${output.evaluationResult.missing_skills.join(', ')}`);
    lines.push(`  Strengths: ${output.evaluationResult.strengths.join(', ')}`);
    lines.push(`  Weaknesses: ${output.evaluationResult.weaknesses.join(', ')}`);
    lines.push(`  Reason: ${output.evaluationResult.reason}`);
    lines.push('');
  }

  // Email Output
  if (output.emailOutput) {
    lines.push('📧 GENERATED EMAIL:');
    lines.push(`  To: ${output.emailOutput.to_address}`);
    lines.push(`  Subject: ${output.emailOutput.subject}`);
    lines.push(`  Body:\n${output.emailOutput.body}`);
    lines.push('');
  }

  // Execution Trace
  lines.push('⏱️  EXECUTION TRACE:');
  for (const step of output.trace) {
    const statusIcon = step.status === 'ok' ? '✓' : '✗';
    lines.push(`  ${statusIcon} ${step.agentName}: ${step.durationMs}ms`);
    lines.push(`    ${step.detail}`);
  }

  lines.push(`\nTotal Execution Time: ${output.executionTime}ms`);
  lines.push(`${'='.repeat(60)}\n`);

  return lines.join('\n');
}
