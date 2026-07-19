/**
 * @fileoverview CLI interface for resume evaluation
 * Allows running the evaluation pipeline from command line
 *
 * Usage:
 *   npx ts-node src/resume-eval-agent/cli.ts <resume_path> <jd_path> [--output <path>]
 *   or
 *   npx ts-node src/resume-eval-agent/cli.ts --resume <path> --jd <path> [--output <path>]
 */

import * as fs from 'fs';
import { runResumeEvaluationPipeline, formatPipelineOutput } from './orchestrator';

/**
 * Simple command line argument parser
 * Supports both positional and flag arguments
 */
function parseArgs() {
  const args: { [key: string]: any; positional: string[] } = { positional: [] };
  
  for (let i = 2; i < process.argv.length; i++) {
    if (process.argv[i].startsWith('--')) {
      const key = process.argv[i].slice(2);
      const value = process.argv[i + 1];
      if (value && !value.startsWith('--')) {
        args[key] = value;
        i++;
      }
    } else if (!process.argv[i].startsWith('-')) {
      args.positional.push(process.argv[i]);
    }
  }
  
  return args;
}

/**
 * Main CLI entry point
 */
async function main() {
  const args = parseArgs();
  
  // Support both positional and flag-based arguments
  let resumePath = args.resume;
  let jdPath = args.jd;
  
  // If positional arguments are provided, use those
  if (args.positional.length >= 2) {
    resumePath = args.positional[0];
    jdPath = args.positional[1];
  }

  if (!resumePath || !jdPath) {
    console.error(`
Resume Evaluation Agent - CLI

Usage:
  # Positional arguments (easiest):
  npx ts-node src/resume-eval-agent/cli.ts <resume_path> <jd_path> [--output <path>]
  
  # Or with flags:
  npx ts-node src/resume-eval-agent/cli.ts \\
    --resume <path> \\
    --jd <path> \\
    [--output <path>]

Arguments:
  <resume_path>   Path to resume file (PDF, DOCX, or TXT) - REQUIRED
  <jd_path>       Path to job description file (PDF, DOCX, or TXT) - REQUIRED
  --output        Optional output file path (JSON) to save results

Examples:
  # Positional:
  npx ts-node src/resume-eval-agent/cli.ts ./resume.pdf ./job-description.txt
  
  # With output file:
  npx ts-node src/resume-eval-agent/cli.ts ./resume.pdf ./job-description.txt --output ./result.json
  
  # With flags:
  npx ts-node src/resume-eval-agent/cli.ts --resume ./resume.pdf --jd ./job-description.txt
`);
    process.exit(1);
  }

  try {
    // Validate input files
    if (!fs.existsSync(resumePath)) {
      console.error(`❌ Resume file not found: ${resumePath}`);
      process.exit(1);
    }

    if (!fs.existsSync(jdPath)) {
      console.error(`❌ Job description file not found: ${jdPath}`);
      process.exit(1);
    }

    console.log('🚀 Starting Resume Evaluation Pipeline...\n');
    console.log(`   Resume:           ${resumePath}`);
    console.log(`   Job Description:  ${jdPath}`);
    console.log('');

    // Run pipeline
    const result = await runResumeEvaluationPipeline({
      resumeFilePath: resumePath,
      jobDescriptionPath: jdPath,
    });

    // Display results
    console.log(formatPipelineOutput(result));

    // Save to output file if specified
    if (args.output) {
      fs.writeFileSync(
        args.output,
        JSON.stringify(result, null, 2),
        'utf-8'
      );
      console.log(`✅ Results saved to: ${args.output}`);
    }

    // Exit with appropriate code
    process.exit(result.success ? 0 : 1);
  } catch (error) {
    console.error(`\n❌ Error: ${(error as Error).message}\n`);
    process.exit(1);
  }
}

main().catch(error => {
  console.error(error);
  process.exit(1);
});
