/**
 * @fileoverview Sample test file demonstrating resume evaluation pipeline usage
 */

import {
  runResumeEvaluationPipeline,
  formatPipelineOutput,
} from '../orchestrator';
import * as path from 'path';

/**
 * Example test showing how to run the complete pipeline
 * Note: Requires actual resume and JD files to run
 */
async function testResumeEvaluationPipeline() {
  console.log('Resume Evaluation Pipeline - Integration Test\n');

  // Example paths (you would replace with actual files)
  const resumePath = path.join(process.cwd(), 'examples', 'sample-resume.pdf');
  const jdPath = path.join(process.cwd(), 'examples', 'sample-jd.txt');

  try {
    console.log('Running pipeline with:');
    console.log(`  Resume: ${resumePath}`);
    console.log(`  JD: ${jdPath}\n`);

    const result = await runResumeEvaluationPipeline({
      resumeFilePath: resumePath,
      jobDescriptionPath: jdPath,
    });

    if (result.success) {
      console.log('✅ Pipeline executed successfully!\n');
      console.log(formatPipelineOutput(result));

      // Detailed analysis
      if (result.evaluationResult) {
        console.log('\n📊 Evaluation Summary:');
        console.log(`  Decision: ${result.evaluationResult.decision}`);
        console.log(`  Score: ${result.evaluationResult.matching_score}/100`);
        console.log(`  Matched Skills: ${result.evaluationResult.matched_skills.length}`);
        console.log(`  Missing Skills: ${result.evaluationResult.missing_skills.length}`);
      }
    } else {
      console.error('❌ Pipeline failed:');
      console.error(result.error);
      console.error('\nExecution trace:');
      result.trace.forEach(step => {
        console.error(`  ${step.agentName}: ${step.status} (${step.detail})`);
      });
    }
  } catch (error) {
    console.error(`\nTest error: ${(error as Error).message}`);
  }
}

// Run test
testResumeEvaluationPipeline().catch(console.error);
