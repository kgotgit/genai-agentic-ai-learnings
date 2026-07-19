# Resume Evaluation Agent 🤖

Multi-agent AI recruitment assistant that automates resume screening and candidate communication.

## Overview

This project implements a **3-agent recruitment pipeline** using Claude AI:

1. **Resume Parser Agent** → Extracts structured candidate data (name, skills, experience, etc.)
2. **Candidate Evaluator Agent** → Compares candidate against job description, generates scoring & decision
3. **Email Generator Agent** → Creates personalized acceptance/rejection emails

## Features

✅ Extracts structured data from resumes (PDF, DOCX, TXT)  
✅ Intelligent candidate-to-JD comparison with skill matching  
✅ Automatic hiring decision (Selected / Not Selected) with reasoning  
✅ Personalized email generation for all candidates  
✅ Full execution tracing for debugging  
✅ Token optimization (truncate large documents)  
✅ Comprehensive JSDoc documentation  
✅ Both API (Express) and CLI interfaces  

## Architecture

```
┌─────────────────────┐
│  Resume File        │
│  (PDF/DOCX/TXT)     │
└──────────┬──────────┘
           │
           v
    ┌─────────────────┐
    │  Resume Parser  │ ← Agent 1
    │     Agent       │
    └────────┬────────┘
             │
             v
    ┌─────────────────────┐
    │ Candidate Data JSON │
    └─────────┬───────────┘
              │
    ┌─────────v─────────────┐
    │  Job Description      │ ← Agent 2
    │  Evaluator Agent      │
    └──────────┬────────────┘
               │
               v
    ┌──────────────────────────┐
    │  Evaluation Result JSON  │
    │  - Decision              │
    │  - Score: 0-100          │
    │  - Matched/Missing Skills│
    │  - Strengths/Weaknesses  │
    └──────────┬───────────────┘
               │
    ┌──────────v──────────┐
    │  Email Generator    │ ← Agent 3
    │      Agent          │
    └──────────┬──────────┘
               │
               v
    ┌────────────────────────┐
    │  Email (Acceptance or  │
    │  Rejection) with       │
    │  Subject + Body        │
    └────────────────────────┘
```

## Installation

### Prerequisites
- Node.js 16+ 
- TypeScript
- Claude API key (ANTHROPIC_API_KEY environment variable)

### Setup

```bash
# Install dependencies
npm install

# Add to package.json (already included)
# See Dependencies section below
```

### Dependencies

```json
{
  "@anthropic-ai/sdk": "^0.28.0",
  "express": "^4.18.2",
  "multer": "^1.4.5",
  "uuid": "^9.0.0",
  "commander": "^11.0.0",
  "pdf-parse": "^1.1.1",
  "mammoth": "^1.7.2"
}
```

## Usage

### CLI Interface

```bash
# Basic usage
npx ts-node src/resume-eval-agent/index.ts \
  --resume /path/to/resume.pdf \
  --jd /path/to/job-description.txt

# Save results to JSON file
npx ts-node src/resume-eval-agent/index.ts \
  --resume ./resume.docx \
  --jd ./jd.pdf \
  --output ./evaluation-result.json
```

### API Server

```bash
# Start server (listens on port 3200)
npm run eval:web

# Upload files and run evaluation
curl -X POST http://localhost:3200/api/evaluate \
  -F "resume=@/path/to/resume.pdf" \
  -F "jobDescription=@/path/to/jd.txt"

# Response:
# {
#   "success": true,
#   "data": {
#     "candidateData": {...},
#     "evaluationResult": {...},
#     "emailOutput": {...},
#     "executionTime": 8500,
#     "trace": [...]
#   },
#   "resultId": "uuid"
# }

# Retrieve saved results
curl http://localhost:3200/api/results/{resultId}

# Health check
curl http://localhost:3200/api/health
```

### Programmatic Usage

```typescript
import {
  runResumeEvaluationPipeline,
  formatPipelineOutput,
} from './resume-eval-agent/orchestrator.js';

const result = await runResumeEvaluationPipeline({
  resumeFilePath: './resume.pdf',
  jobDescriptionPath: './jd.txt',
});

if (result.success) {
  console.log(formatPipelineOutput(result));
  console.log(result.candidateData);
  console.log(result.evaluationResult);
  console.log(result.emailOutput);
}
```

## Environment Variables

```bash
# Claude API (required)
ANTHROPIC_API_KEY=sk-ant-...

# Token optimization
MAX_RESUME_CHARS=4000          # Limit resume text sent to LLM
MAX_JD_CHARS=3000              # Limit JD text sent to LLM

# Server
PORT=3200                       # Express server port
```

## Output Schemas

### Candidate Data (from Agent 1)
```json
{
  "candidate_name": "string",
  "email": "string or null",
  "phone": "string or null",
  "total_experience": "string (e.g., '5 years')",
  "education": "string",
  "skills": ["string"],
  "projects": ["string"],
  "certifications": ["string"]
}
```

### Evaluation Result (from Agent 2)
```json
{
  "candidate_name": "string",
  "decision": "Selected | Not Selected",
  "matching_score": 0-100,
  "matched_skills": ["string"],
  "missing_skills": ["string"],
  "strengths": ["string"],
  "weaknesses": ["string"],
  "reason": "string"
}
```

### Email Output (from Agent 3)
```json
{
  "subject": "string",
  "body": "string (with \\n for line breaks)",
  "to_address": "string"
}
```

## Project Structure

```
src/resume-eval-agent/
├── agents/
│   ├── resume-parser.ts        # Agent 1: Extraction
│   ├── evaluator.ts            # Agent 2: Scoring
│   └── email-generator.ts      # Agent 3: Communication
├── schemas/
│   ├── candidate.ts            # CandidateData type
│   ├── evaluation.ts           # EvaluationResult type
│   └── email.ts                # EmailOutput type
├── utils/
│   ├── document-loader.ts      # PDF/DOCX/TXT parsing
│   ├── text-extractor.ts       # Text processing helpers
│   └── validators.ts           # Schema validation
├── orchestrator.ts             # Pipeline coordinator
├── server.ts                   # Express API
├── index.ts                    # CLI + exports
└── __tests__/                  # Test files
```

## npm Scripts

```bash
npm run eval:resume           # Run CLI (shorthand)
npm run eval:web              # Start API server
npm run eval:test             # Run tests
npm run build                 # Compile TypeScript
npm run dev                   # Development mode
```

## Key Design Decisions

| Aspect | Choice | Rationale |
|--------|--------|-----------|
| **Execution** | Sequential (Agent 1→2→3) | Each agent's output feeds the next |
| **LLM** | Claude 3.5 Sonnet | Fast, accurate, cost-effective for structured tasks |
| **Document Format** | PDF, DOCX, TXT | Most common resume and JD formats |
| **Validation** | Strict schema checking | Catches malformed LLM responses early |
| **Storage** | In-memory (API demo) | Use database for production |
| **Error Handling** | Fail-fast with details | Return partial results + error context |

## Error Handling

- **File Not Found**: Clear error message indicating missing file path
- **Unsupported Format**: Rejects formats outside .pdf, .docx, .txt
- **LLM Parsing Failure**: Returns error with LLM response snippet for debugging
- **Validation Failure**: Detailed field-level error indicating what's wrong
- **Pipeline Failure**: Returns partial results from successful agents + error at failed step

## Performance

Typical execution:
- Resume Parsing: 2-3 seconds
- Candidate Evaluation: 3-4 seconds  
- Email Generation: 2-3 seconds
- **Total: ~8-10 seconds per evaluation**

Token usage per evaluation: ~2,500-3,000 tokens (~$0.001)

## Token Optimization

To keep costs low:
- Resume text truncated to first 4000 chars
- JD text truncated to first 3000 chars
- Skills limited to 15 items max
- Prompts use concise language
- JSON-only responses (no prose)

Adjust via environment variables if needed.

## Known Limitations

- DOCX parsing works best with standard formatting
- Large PDF files (100+ pages) truncated to first 4000 chars
- Email generation uses template-style prompts (not fully custom yet)
- No persistence layer (in-memory only for now)

## Future Enhancements

- [ ] Batch processing (evaluate 100 resumes vs 1 JD)
- [ ] Database storage (PostgreSQL/MongoDB)
- [ ] Feedback loop (improve prompts based on recruiter feedback)
- [ ] Bias detection and mitigation
- [ ] Custom scoring weights
- [ ] Interview scheduling integration
- [ ] Resume rewriting suggestions

## Testing

```bash
npm run eval:test
```

Test fixtures included for:
- PDF/DOCX/TXT file handling
- Email/phone/experience extraction
- Schema validation
- Full pipeline integration

## Contributing

Follow existing patterns:
- Add JSDoc comments to all functions
- Validate all LLM outputs
- Test error paths
- Keep token usage optimized
- Document environment variable usage

## License

MIT

---

**Built with Claude AI** • Modular, transparent, scalable
