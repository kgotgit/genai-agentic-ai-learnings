import * as dotenv from "dotenv";
import { pipeline } from "@huggingface/transformers";
import { promises as fs } from "fs";
import * as path from "path";
import { AppChatModel, createChatModel } from "./llm";

dotenv.config();

type DialogSumRow = {
  id: string;
  dialog: string;
  summary: string;
};

type DialogSample = {
  id: string;
  dialog: string;
  references: string[];
};

type SampleResult = {
  id: string;
  references: string[];
  bestReference: string;
  prediction: string;
  bleu4: number;
  rouge1F1: number;
  rouge2F1: number;
  rougeLF1: number;
  bertPrecision: number;
  bertRecall: number;
  bertF1: number;
};

type AggregateMetrics = {
  bleu4: number;
  rouge1F1: number;
  rouge2F1: number;
  rougeLF1: number;
  bertPrecision: number;
  bertRecall: number;
  bertF1: number;
};

type EvalOutput = {
  config: {
    dataset: string;
    split: string;
    offset: number;
    sampleSize: number;
    llmProvider: string;
    llmModel: string;
    bertModel: string;
  };
  aggregate: AggregateMetrics;
  samples: SampleResult[];
  createdAt: string;
};

type HfRowsResponse = {
  rows: Array<{
    row: {
      id?: string;
      dialogue?: string;
      dialog?: string;
      summary?: string;
    };
  }>;
};

const DATASET_NAME = "knkarthick/dialogsum";
const DATASET_CONFIG = process.env.DIALOGSUM_CONFIG ?? "default";
const DEFAULT_SPLIT = process.env.DIALOGSUM_SPLIT ?? "test";
const DEFAULT_OFFSET = Number.parseInt(process.env.DIALOGSUM_OFFSET ?? "0", 10);
const DEFAULT_SAMPLE_SIZE = Number.parseInt(
  process.env.DIALOGSUM_SAMPLE_SIZE ?? "5",
  10
);
const BERT_MODEL = process.env.BERT_MODEL ?? "Xenova/all-MiniLM-L6-v2";
const OUTPUT_PATH = path.resolve(
  process.cwd(),
  process.env.EVAL_OUTPUT_PATH ?? "db/dialogsum-eval-results.json"
);

/**
 * Extracts plain text from LangChain chat model output content.
 */
function extractResponseText(
  content: string | Array<{ type?: string; text?: string }>
): string {
  if (typeof content === "string") {
    return content.trim();
  }

  return content
    .map((item) => (typeof item.text === "string" ? item.text : ""))
    .join("")
    .trim();
}

/**
 * Tokenizes text for lexical overlap metrics.
 */
function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter(Boolean);
}

/**
 * Builds n-gram frequency counts from token sequences.
 */
function makeNGrams(tokens: string[], n: number): Map<string, number> {
  const grams = new Map<string, number>();
  if (tokens.length < n) {
    return grams;
  }

  for (let i = 0; i <= tokens.length - n; i++) {
    const gram = tokens.slice(i, i + n).join(" ");
    grams.set(gram, (grams.get(gram) ?? 0) + 1);
  }

  return grams;
}

/**
 * Computes clipped modified precision for a single reference.
 */
function modifiedPrecision(reference: string[], candidate: string[], n: number): number {
  const candNgrams = makeNGrams(candidate, n);
  const refNgrams = makeNGrams(reference, n);

  let clippedCount = 0;
  let totalCount = 0;

  for (const [gram, count] of candNgrams) {
    const refCount = refNgrams.get(gram) ?? 0;
    clippedCount += Math.min(count, refCount);
    totalCount += count;
  }

  if (totalCount === 0) {
    return 0;
  }

  return clippedCount / totalCount;
}

/**
 * Computes clipped modified precision across multiple references.
 */
function modifiedPrecisionMulti(
  references: string[][],
  candidate: string[],
  n: number
): number {
  const candNgrams = makeNGrams(candidate, n);

  let clippedCount = 0;
  let totalCount = 0;

  const maxRefCounts = new Map<string, number>();
  for (const reference of references) {
    const refNgrams = makeNGrams(reference, n);
    for (const [gram, count] of refNgrams) {
      const prev = maxRefCounts.get(gram) ?? 0;
      if (count > prev) {
        maxRefCounts.set(gram, count);
      }
    }
  }

  for (const [gram, count] of candNgrams) {
    const refCount = maxRefCounts.get(gram) ?? 0;
    clippedCount += Math.min(count, refCount);
    totalCount += count;
  }

  if (totalCount === 0) {
    return 0;
  }

  return clippedCount / totalCount;
}

/**
 * Computes BLEU-4 against a single reference summary.
 */
function bleu4(referenceText: string, candidateText: string): number {
  const reference = tokenize(referenceText);
  const candidate = tokenize(candidateText);

  if (candidate.length === 0) {
    return 0;
  }

  const p: number[] = [];
  for (let n = 1; n <= 4; n++) {
    const mp = modifiedPrecision(reference, candidate, n);
    p.push(mp > 0 ? mp : 1e-9);
  }

  const logMean = p.reduce((sum, value) => sum + 0.25 * Math.log(value), 0);
  const bp =
    candidate.length > reference.length
      ? 1
      : Math.exp(1 - reference.length / Math.max(candidate.length, 1));

  return bp * Math.exp(logMean);
}

/**
 * Computes BLEU-4 with multi-reference matching.
 */
function bleu4Multi(referenceTexts: string[], candidateText: string): number {
  const references = referenceTexts.map((text) => tokenize(text));
  const candidate = tokenize(candidateText);

  if (candidate.length === 0 || references.length === 0) {
    return 0;
  }

  const p: number[] = [];
  for (let n = 1; n <= 4; n++) {
    const mp = modifiedPrecisionMulti(references, candidate, n);
    p.push(mp > 0 ? mp : 1e-9);
  }

  const logMean = p.reduce((sum, value) => sum + 0.25 * Math.log(value), 0);
  const referenceLengths = references.map((tokens) => tokens.length);
  const closestRefLength = referenceLengths.reduce((best, current) => {
    const bestDistance = Math.abs(best - candidate.length);
    const currentDistance = Math.abs(current - candidate.length);

    if (currentDistance < bestDistance) {
      return current;
    }
    if (currentDistance === bestDistance) {
      return current < best ? current : best;
    }
    return best;
  }, referenceLengths[0] ?? 0);

  const bp =
    candidate.length > closestRefLength
      ? 1
      : Math.exp(1 - closestRefLength / Math.max(candidate.length, 1));

  return bp * Math.exp(logMean);
}

/**
 * Computes overlap-based F1 for ROUGE-N components.
 */
function overlapF1(referenceTokens: string[], candidateTokens: string[], n: number): number {
  const refNgrams = makeNGrams(referenceTokens, n);
  const candNgrams = makeNGrams(candidateTokens, n);

  let overlap = 0;
  let refTotal = 0;
  let candTotal = 0;

  for (const count of refNgrams.values()) {
    refTotal += count;
  }
  for (const count of candNgrams.values()) {
    candTotal += count;
  }

  for (const [gram, candCount] of candNgrams) {
    const refCount = refNgrams.get(gram) ?? 0;
    overlap += Math.min(candCount, refCount);
  }

  if (refTotal === 0 || candTotal === 0 || overlap === 0) {
    return 0;
  }

  const precision = overlap / candTotal;
  const recall = overlap / refTotal;

  if (precision + recall === 0) {
    return 0;
  }

  return (2 * precision * recall) / (precision + recall);
}

/**
 * Computes longest common subsequence length for ROUGE-L.
 */
function lcsLength(a: string[], b: string[]): number {
  const dp: number[][] = Array.from({ length: a.length + 1 }, () =>
    Array<number>(b.length + 1).fill(0)
  );

  for (let i = 1; i <= a.length; i++) {
    for (let j = 1; j <= b.length; j++) {
      if (a[i - 1] === b[j - 1]) {
        dp[i][j] = dp[i - 1][j - 1] + 1;
      } else {
        dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
      }
    }
  }

  return dp[a.length][b.length];
}

/**
 * Computes ROUGE-1, ROUGE-2, and ROUGE-L F1 against a single reference.
 */
function rougeScores(referenceText: string, candidateText: string): {
  rouge1F1: number;
  rouge2F1: number;
  rougeLF1: number;
} {
  const referenceTokens = tokenize(referenceText);
  const candidateTokens = tokenize(candidateText);

  const rouge1F1 = overlapF1(referenceTokens, candidateTokens, 1);
  const rouge2F1 = overlapF1(referenceTokens, candidateTokens, 2);

  const lcs = lcsLength(referenceTokens, candidateTokens);
  const precision =
    candidateTokens.length === 0 ? 0 : lcs / Math.max(candidateTokens.length, 1);
  const recall = referenceTokens.length === 0 ? 0 : lcs / Math.max(referenceTokens.length, 1);
  const rougeLF1 =
    precision + recall === 0 ? 0 : (2 * precision * recall) / (precision + recall);

  return { rouge1F1, rouge2F1, rougeLF1 };
}

/**
 * Computes ROUGE-1/2/L F1 using best-of multi-reference scoring.
 */
function rougeScoresMulti(referenceTexts: string[], candidateText: string): {
  rouge1F1: number;
  rouge2F1: number;
  rougeLF1: number;
} {
  if (referenceTexts.length === 0) {
    return { rouge1F1: 0, rouge2F1: 0, rougeLF1: 0 };
  }

  let bestRouge1 = 0;
  let bestRouge2 = 0;
  let bestRougeL = 0;

  for (const referenceText of referenceTexts) {
    const score = rougeScores(referenceText, candidateText);
    bestRouge1 = Math.max(bestRouge1, score.rouge1F1);
    bestRouge2 = Math.max(bestRouge2, score.rouge2F1);
    bestRougeL = Math.max(bestRougeL, score.rougeLF1);
  }

  return {
    rouge1F1: bestRouge1,
    rouge2F1: bestRouge2,
    rougeLF1: bestRougeL,
  };
}

/**
 * Computes cosine similarity between two vectors.
 */
function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length === 0 || b.length === 0 || a.length !== b.length) {
    return 0;
  }

  let dot = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  const denominator = Math.sqrt(normA) * Math.sqrt(normB);
  return denominator === 0 ? 0 : dot / denominator;
}

/**
 * Normalizes feature-extraction outputs into token-level numeric vectors.
 */
async function getTokenEmbeddings(
  extractor: (text: string) => Promise<unknown>,
  text: string
): Promise<number[][]> {
  const output = await extractor(text);

  let normalized: unknown = output;

  if (
    normalized !== null &&
    typeof normalized === "object" &&
    "tolist" in normalized &&
    typeof (normalized as { tolist?: unknown }).tolist === "function"
  ) {
    normalized = (normalized as { tolist: () => unknown }).tolist();
  }

  if (!Array.isArray(normalized)) {
    throw new Error("Unexpected embedding output shape.");
  }

  let vectors: unknown = normalized;
  while (
    Array.isArray(vectors) &&
    vectors.length > 0 &&
    Array.isArray(vectors[0]) &&
    (vectors[0] as unknown[]).length > 0 &&
    Array.isArray((vectors[0] as unknown[])[0])
  ) {
    vectors = vectors[0];
  }

  if (!Array.isArray(vectors) || !Array.isArray(vectors[0])) {
    throw new Error("Unexpected embedding output shape.");
  }

  return (vectors as unknown[])
    .map((item) =>
      Array.isArray(item)
        ? item.map((value) => Number(value)).filter((value) => Number.isFinite(value))
        : []
    )
    .filter((vector) => vector.length > 0);
}

/**
 * Computes a lightweight BERTScore approximation for one reference/candidate pair.
 */
async function bertScore(
  extractor: (text: string) => Promise<unknown>,
  referenceText: string,
  candidateText: string
): Promise<{ precision: number; recall: number; f1: number }> {
  const [referenceVectors, candidateVectors] = await Promise.all([
    getTokenEmbeddings(extractor, referenceText),
    getTokenEmbeddings(extractor, candidateText),
  ]);

  if (referenceVectors.length === 0 || candidateVectors.length === 0) {
    return { precision: 0, recall: 0, f1: 0 };
  }

  const precisionParts = candidateVectors.map((cand) => {
    let maxScore = -1;
    for (const ref of referenceVectors) {
      const score = cosineSimilarity(cand, ref);
      if (score > maxScore) {
        maxScore = score;
      }
    }
    return Math.max(0, maxScore);
  });

  const recallParts = referenceVectors.map((ref) => {
    let maxScore = -1;
    for (const cand of candidateVectors) {
      const score = cosineSimilarity(ref, cand);
      if (score > maxScore) {
        maxScore = score;
      }
    }
    return Math.max(0, maxScore);
  });

  const precision =
    precisionParts.reduce((sum, value) => sum + value, 0) /
    Math.max(precisionParts.length, 1);
  const recall =
    recallParts.reduce((sum, value) => sum + value, 0) /
    Math.max(recallParts.length, 1);
  const f1 =
    precision + recall === 0 ? 0 : (2 * precision * recall) / (precision + recall);

  return { precision, recall, f1 };
}

/**
 * Computes BERTScore with best-of multi-reference selection.
 */
async function bertScoreMulti(
  extractor: (text: string) => Promise<unknown>,
  referenceTexts: string[],
  candidateText: string
): Promise<{ precision: number; recall: number; f1: number; bestReference: string }> {
  if (referenceTexts.length === 0) {
    return { precision: 0, recall: 0, f1: 0, bestReference: "" };
  }

  let best = {
    precision: 0,
    recall: 0,
    f1: 0,
    bestReference: referenceTexts[0],
  };

  for (const referenceText of referenceTexts) {
    const score = await bertScore(extractor, referenceText, candidateText);
    if (score.f1 > best.f1) {
      best = {
        precision: score.precision,
        recall: score.recall,
        f1: score.f1,
        bestReference: referenceText,
      };
    }
  }

  return best;
}

/**
 * Groups DialogSum variants that belong to the same base dialogue.
 */
function dialogGroupKey(id: string): string {
  return id.replace(/_\d+$/, "");
}

/**
 * Fetches enough rows and collapses them into unique dialogues with multiple references.
 */
async function fetchUniqueDialogSamples(
  split: string,
  offset: number,
  sampleSize: number
): Promise<DialogSample[]> {
  const unique = new Map<string, DialogSample>();
  let cursor = offset;
  const batchSize = Math.max(sampleSize * 6, 50);
  const maxFetches = 10;

  for (let attempt = 0; attempt < maxFetches && unique.size < sampleSize; attempt++) {
    const rows = await fetchDialogSumRows(split, cursor, batchSize);
    if (rows.length === 0) {
      break;
    }

    for (const row of rows) {
      const key = dialogGroupKey(row.id);
      const existing = unique.get(key);

      if (!existing) {
        unique.set(key, {
          id: key,
          dialog: row.dialog,
          references: [row.summary],
        });
      } else if (!existing.references.includes(row.summary)) {
        existing.references.push(row.summary);
      }

      if (unique.size >= sampleSize) {
        break;
      }
    }

    cursor += rows.length;
  }

  return Array.from(unique.values()).slice(0, sampleSize);
}

async function fetchDialogSumRows(
  split: string,
  offset: number,
  length: number
): Promise<DialogSumRow[]> {
  const dataset = encodeURIComponent(DATASET_NAME);
  const config = DATASET_CONFIG;
  const url = `https://datasets-server.huggingface.co/rows?dataset=${dataset}&config=${config}&split=${encodeURIComponent(
    split
  )}&offset=${offset}&length=${length}`;

  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch dataset rows: HTTP ${response.status}`);
  }

  const payload = (await response.json()) as HfRowsResponse;
  if (!payload.rows || !Array.isArray(payload.rows)) {
    throw new Error("Unexpected dataset response format.");
  }

  return payload.rows
    .map((item, index) => {
      const row = item.row ?? {};
      return {
        id: String(row.id ?? `row-${offset + index}`),
        dialog: String(row.dialogue ?? row.dialog ?? ""),
        summary: String(row.summary ?? ""),
      };
    })
    .filter((row) => row.dialog.trim().length > 0 && row.summary.trim().length > 0);
}

/**
 * Calls Groq to produce a concise one-sentence summary for a dialogue.
 */
async function summarizeDialog(
  llm: AppChatModel,
  dialog: string
): Promise<string> {
  const prompt = `You are a dialogue summarization assistant.
Write exactly ONE sentence summary.
Keep key decisions, actions, and named entities.
Do not add information not present in the dialogue.
Return only the summary sentence.

Dialogue:
${dialog}`;

  const response = await llm.invoke(prompt);
  return extractResponseText(response.content as string | Array<{ type?: string; text?: string }>);
}

/**
 * Computes the arithmetic mean of numeric values.
 */
function average(values: number[]): number {
  if (values.length === 0) {
    return 0;
  }
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

/**
 * Persists evaluation output JSON to disk.
 */
async function saveOutput(output: EvalOutput): Promise<void> {
  await fs.mkdir(path.dirname(OUTPUT_PATH), { recursive: true });
  await fs.writeFile(OUTPUT_PATH, JSON.stringify(output, null, 2), "utf-8");
}

/**
 * Runs the full DialogSum evaluation pipeline:
 * fetch, summarize, score, aggregate, and save results.
 */
async function runEval(): Promise<void> {
  const { llm, provider, model } = createChatModel({ temperature: 0 });

  const split = DEFAULT_SPLIT;
  const offset = Number.isNaN(DEFAULT_OFFSET) ? 0 : DEFAULT_OFFSET;
  const sampleSize = Number.isNaN(DEFAULT_SAMPLE_SIZE) ? 5 : DEFAULT_SAMPLE_SIZE;

  if (sampleSize <= 0) {
    throw new Error("DIALOGSUM_SAMPLE_SIZE must be greater than 0.");
  }

  console.log(`📥 Fetching ${sampleSize} unique dialogs from ${DATASET_NAME} (${split} split)...`);
  const samples = await fetchUniqueDialogSamples(split, offset, sampleSize);

  if (samples.length === 0) {
    throw new Error("No valid samples were returned from dataset.");
  }

  console.log(`✅ Loaded ${samples.length} unique dialogs.`);

  console.log(`🤖 Using LLM provider: ${provider}`);
  console.log(`🤖 Using LLM model: ${model}`);
  console.log(`🧠 Loading BERT scorer model: ${BERT_MODEL}`);

  const extractor = (await pipeline("feature-extraction", BERT_MODEL)) as (
    text: string
  ) => Promise<unknown>;

  const sampleResults: SampleResult[] = [];

  for (let i = 0; i < samples.length; i++) {
    const sample = samples[i];
    console.log(
      `\n[${i + 1}/${samples.length}] Summarizing dialog id=${sample.id} (refs=${sample.references.length})...`
    );

    const prediction = await summarizeDialog(llm, sample.dialog);
    const bleu = bleu4Multi(sample.references, prediction);
    const rouge = rougeScoresMulti(sample.references, prediction);
    const bert = await bertScoreMulti(extractor, sample.references, prediction);

    sampleResults.push({
      id: sample.id,
      references: sample.references,
      bestReference: bert.bestReference,
      prediction,
      bleu4: bleu,
      rouge1F1: rouge.rouge1F1,
      rouge2F1: rouge.rouge2F1,
      rougeLF1: rouge.rougeLF1,
      bertPrecision: bert.precision,
      bertRecall: bert.recall,
      bertF1: bert.f1,
    });

    console.log(
      `BLEU-4=${bleu.toFixed(4)} | ROUGE-1 F1=${rouge.rouge1F1.toFixed(4)} | ROUGE-2 F1=${rouge.rouge2F1.toFixed(4)} | ROUGE-L F1=${rouge.rougeLF1.toFixed(4)} | BERT F1=${bert.f1.toFixed(4)}`
    );
  }

  const aggregate: AggregateMetrics = {
    bleu4: average(sampleResults.map((item) => item.bleu4)),
    rouge1F1: average(sampleResults.map((item) => item.rouge1F1)),
    rouge2F1: average(sampleResults.map((item) => item.rouge2F1)),
    rougeLF1: average(sampleResults.map((item) => item.rougeLF1)),
    bertPrecision: average(sampleResults.map((item) => item.bertPrecision)),
    bertRecall: average(sampleResults.map((item) => item.bertRecall)),
    bertF1: average(sampleResults.map((item) => item.bertF1)),
  };

  const output: EvalOutput = {
    config: {
      dataset: DATASET_NAME,
      split,
      offset,
      sampleSize: samples.length,
      llmProvider: provider,
      llmModel: model,
      bertModel: BERT_MODEL,
    },
    aggregate,
    samples: sampleResults,
    createdAt: new Date().toISOString(),
  };

  await saveOutput(output);

  console.log("\n================ EVALUATION SUMMARY ================");
  console.log(`Dataset     : ${DATASET_NAME} (config=${DATASET_CONFIG}, split=${split})`);
  console.log(`Samples     : ${samples.length} unique dialogs (offset=${offset})`);
  console.log(`LLM         : ${provider}/${model}`);
  console.log(`BERT model  : ${BERT_MODEL}`);
  console.log(`BLEU-4      : ${aggregate.bleu4.toFixed(4)}`);
  console.log(`ROUGE-1 F1  : ${aggregate.rouge1F1.toFixed(4)}`);
  console.log(`ROUGE-2 F1  : ${aggregate.rouge2F1.toFixed(4)}`);
  console.log(`ROUGE-L F1  : ${aggregate.rougeLF1.toFixed(4)}`);
  console.log(`BERT P      : ${aggregate.bertPrecision.toFixed(4)}`);
  console.log(`BERT R      : ${aggregate.bertRecall.toFixed(4)}`);
  console.log(`BERT F1     : ${aggregate.bertF1.toFixed(4)}`);
  console.log(`Saved JSON  : ${OUTPUT_PATH}`);
  console.log("====================================================");
}

void runEval().catch((error) => {
  console.error("❌ Evaluation failed:", error instanceof Error ? error.message : String(error));
  process.exit(1);
});
