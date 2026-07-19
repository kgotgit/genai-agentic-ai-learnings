/**
 * @fileoverview Express server for resume evaluation API
 * Provides REST endpoints for running the evaluation pipeline
 * 
 * Note: This simplified version takes file paths in the request body
 * instead of using multipart file upload (multer) to avoid dependency issues
 * 
 * Usage:
 *   npm run eval:web
 *   
 *   curl -X POST http://localhost:3200/api/evaluate \
 *     -H "Content-Type: application/json" \
 *     -d '{
 *       "resumePath": "/path/to/resume.pdf",
 *       "jdPath": "/path/to/job-description.txt"
 *     }'
 */

import express from 'express';
import * as fs from 'fs';
import { v4 as uuidv4 } from 'uuid';
import { runResumeEvaluationPipeline, formatPipelineOutput, PipelineOutput } from './orchestrator';

const app = express();
const port = parseInt(process.env.PORT || '3200');

// Middleware
app.use(express.json());

// Store results in memory (for demo; use database in production)
const evaluationResults = new Map<string, PipelineOutput>();

/**
 * POST /api/evaluate
 * Request body: { resumePath: string, jdPath: string }
 * Run evaluation pipeline on existing files
 */
app.post('/api/evaluate', async (req: any, res: any) => {
  try {
    const { resumePath, jdPath } = req.body;

    if (!resumePath || !jdPath) {
      return res.status(400).json({
        success: false,
        error: 'Both resumePath and jdPath are required in request body',
      });
    }

    if (!fs.existsSync(resumePath)) {
      return res.status(400).json({
        success: false,
        error: `Resume file not found: ${resumePath}`,
      });
    }

    if (!fs.existsSync(jdPath)) {
      return res.status(400).json({
        success: false,
        error: `Job description file not found: ${jdPath}`,
      });
    }

    try {
      // Run pipeline
      const result = await runResumeEvaluationPipeline({
        resumeFilePath: resumePath,
        jobDescriptionPath: jdPath,
      });

      const resultId = uuidv4();
      evaluationResults.set(resultId, result);

      // Return results
      res.json({
        success: result.success,
        data: result,
        resultId,
      });
    } catch (pipelineError) {
      res.status(500).json({
        success: false,
        error: (pipelineError as Error).message,
      });
    }
  } catch (error) {
    res.status(500).json({
      success: false,
      error: (error as Error).message,
    });
  }
});

/**
 * GET /api/results/:resultId
 * Retrieve a previously stored evaluation result
 */
app.get('/api/results/:resultId', (req: any, res: any) => {
  const result = evaluationResults.get(req.params.resultId);
  if (!result) {
    return res.status(404).json({
      success: false,
      error: 'Result not found',
    });
  }
  res.json({
    success: true,
    data: result,
  });
});

/**
 * GET /api/health
 * Health check endpoint
 */
app.get('/api/health', (req: any, res: any) => {
  res.json({
    status: 'ok',
    service: 'Resume Evaluation API',
    timestamp: new Date().toISOString(),
  });
});

// Start server
app.listen(port, () => {
  console.log(`\n${'='.repeat(60)}`);
  console.log(`Resume Evaluation API Server`);
  console.log(`${'='.repeat(60)}`);
  console.log(`Server running on http://localhost:${port}`);
  console.log(`\nEndpoints:`);
  console.log(`  POST   /api/evaluate              - Submit resume and JD paths for evaluation`);
  console.log(`  GET    /api/results/:resultId     - Retrieve evaluation results`);
  console.log(`  GET    /api/health                - Health check`);
  console.log(`\nExample curl:`);
  console.log(`  curl -X POST http://localhost:${port}/api/evaluate \\`);
  console.log(`    -H "Content-Type: application/json" \\`);
  console.log(`    -d '{`);
  console.log(`      "resumePath": "/path/to/resume.pdf",`);
  console.log(`      "jdPath": "/path/to/job-description.txt"`);
  console.log(`    }'`);
  console.log(`${'='.repeat(60)}\n`);
});

export default app;
