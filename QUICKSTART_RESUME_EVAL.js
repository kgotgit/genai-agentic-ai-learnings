/**
 * @fileoverview Quick Start Guide for Resume Evaluation Agent
 * 
 * This file contains example commands to get started with the resume evaluation pipeline.
 * Run these commands from the project root directory.
 */

// ============================================================================
// SETUP
// ============================================================================

// 1. Install dependencies
npm install

// 2. Set up environment variables
// Copy the .env.example to .env and update with your API keys
cp .env.example .env

// Required environment variables:
// ANTHROPIC_API_KEY=sk-ant-...

// ============================================================================
// RUNNING THE PIPELINE
// ============================================================================

// Using CLI:
// npx ts-node src/resume-eval-agent/index.ts --resume <path> --jd <path>

// Example with sample files:
npx ts-node src/resume-eval-agent/index.ts \
  --resume ./src/resume-eval-agent/__tests__/sample-resume.txt \
  --jd ./src/resume-eval-agent/__tests__/sample-jd.txt

// Save results to JSON file:
npx ts-node src/resume-eval-agent/index.ts \
  --resume ./resume.pdf \
  --jd ./job-description.txt \
  --output ./evaluation-result.json

// ============================================================================
// STARTING API SERVER
// ============================================================================

// Start the Express server (default port 3200):
npm run eval:web

// Then use curl to submit evaluations:
curl -X POST http://localhost:3200/api/evaluate \
  -F "resume=@./src/resume-eval-agent/__tests__/sample-resume.txt" \
  -F "jobDescription=@./src/resume-eval-agent/__tests__/sample-jd.txt"

// Check health:
curl http://localhost:3200/api/health

// ============================================================================
// PROGRAMMATIC USAGE
// ============================================================================

// In TypeScript/Node.js:
import { runResumeEvaluationPipeline, formatPipelineOutput } from './resume-eval-agent/orchestrator.js';

const result = await runResumeEvaluationPipeline({
  resumeFilePath: './resume.pdf',
  jobDescriptionPath: './jd.txt',
});

console.log(formatPipelineOutput(result));

// ============================================================================
// EXPECTED OUTPUT
// ============================================================================

// Success response includes:
// {
//   "success": true,
//   "candidateData": { ... },
//   "evaluationResult": { 
//     "decision": "Selected" | "Not Selected",
//     "matching_score": 0-100,
//     "matched_skills": [...],
//     "missing_skills": [...],
//     "strengths": [...],
//     "weaknesses": [...],
//     "reason": "..."
//   },
//   "emailOutput": { 
//     "subject": "...",
//     "body": "...",
//     "to_address": "..."
//   },
//   "executionTime": milliseconds,
//   "trace": [
//     { "agentName": "Resume Parser", "status": "ok", "durationMs": ... },
//     { "agentName": "Candidate Evaluator", "status": "ok", "durationMs": ... },
//     { "agentName": "Email Generator", "status": "ok", "durationMs": ... }
//   ]
// }

// ============================================================================
// TROUBLESHOOTING
// ============================================================================

// 1. "API key not found" error
//    → Set ANTHROPIC_API_KEY environment variable
//    → export ANTHROPIC_API_KEY=sk-ant-...

// 2. "File not found" error
//    → Use absolute paths or check file exists
//    → Resume and JD must be .pdf, .docx, or .txt

// 3. "JSON parsing failed" error
//    → LLM output format issue
//    → Check error message for details
//    → Try with shorter resume/JD (MAX_RESUME_CHARS env var)

// 4. "Validation failed" error
//    → LLM returned invalid data structure
//    → Check individual field errors
//    → Adjust prompt templates if needed

// ============================================================================
// ENVIRONMENT VARIABLES
// ============================================================================

// .env file content:

# Required: Claude API access
ANTHROPIC_API_KEY=sk-ant-...

# Optional: Token optimization (for large documents)
MAX_RESUME_CHARS=4000
MAX_JD_CHARS=3000

# Optional: Server configuration
PORT=3200

// ============================================================================
// TESTING WITH SAMPLE DATA
// ============================================================================

// Test files are included in __tests__/:
// - sample-resume.txt (5 years ML experience, banking domain)
// - sample-jd.txt (Senior ML Engineer role requirements)
// - integration.test.ts (example test file)

// Expected test result:
// - Candidate should be SELECTED (80+ matching score)
// - Matched skills: Python, SQL, ML frameworks, AWS
// - Missing skills: Docker, Kubernetes (preferred only)
// - Email: Professional acceptance email

// ============================================================================
// PRODUCTION CHECKLIST
// ============================================================================

// Before deploying to production:
// [ ] Set ANTHROPIC_API_KEY via secure environment management
// [ ] Configure MAX_RESUME_CHARS and MAX_JD_CHARS for cost control
// [ ] Add database layer (PostgreSQL/MongoDB) instead of in-memory storage
// [ ] Implement request rate limiting and authentication
// [ ] Add comprehensive error logging
// [ ] Set up monitoring for API latency and failures
// [ ] Add input validation and file size limits
// [ ] Implement audit logging for compliance
// [ ] Set up backup/recovery procedures
// [ ] Load test API with concurrent requests

// ============================================================================
