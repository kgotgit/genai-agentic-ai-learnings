import * as dotenv from "dotenv";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/huggingface_transformers";
import { BM25Retriever } from "@langchain/community/retrievers/bm25";
import { FaissStore } from "@langchain/community/vectorstores/faiss";
import { Document } from "@langchain/core/documents";
import { EnsembleRetriever } from "@langchain/classic/retrievers/ensemble";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { promises as fs } from "fs";
import * as path from "path";
import * as readline from "readline";
import { createChatModel } from "./llm";

dotenv.config();

type RetrievalMode = "similarity" | "mmr" | "hybrid";

const EMBEDDING_MODEL = resolveEmbeddingModel(
  process.env.EMBEDDING_MODEL ?? "Xenova/bge-m3"
);
const DEFAULT_RETRIEVAL_MODE: RetrievalMode = "hybrid";
const SIMILARITY_SCORE_THRESHOLD = 0.25;
const RERANK_SCORE_THRESHOLD = 0.2;
const MMR_FETCH_K = 20;
const MMR_LAMBDA = 0.5;
const MMR_TOP_K = 5;
const SIMILARITY_K = 5;
const HYBRID_BM25_K = 5;
const HYBRID_VECTOR_K = 5;
const HYBRID_TOP_K = 8;
const HYBRID_WEIGHTS: [number, number] = [0.3, 0.7];
const CACHE_ENABLED_MODES: RetrievalMode[] = ["similarity", "mmr", "hybrid"];
const QUERY_RESPONSE_CACHE_TTL_MS = 24 * 60 * 60 * 1000;
const DOCS_DIRECTORY_PATH = path.resolve(process.cwd(), "docs");
const FAISS_INDEX_PATH = path.resolve(process.cwd(), "db", "faiss");
const QUERY_RESPONSE_CACHE_PATH = path.resolve(
  process.cwd(),
  "db",
  "query-response-cache.json"
);

type QueryResponseCacheEntry = {
  response: string;
  createdAt: string;
};

type QueryResponseCache = Record<string, QueryResponseCacheEntry>;

type ChunkScore = {
  chunk: Document;
  score: number;
};

function resolveEmbeddingModel(modelName: string): string {
  if (modelName === "BAAI/bge-m3") {
    console.log(
      "ℹ️ Using Xenova/bge-m3 ONNX weights for local JavaScript inference."
    );
    return "Xenova/bge-m3";
  }

  return modelName;
}

function isRetrievalMode(value: string): value is RetrievalMode {
  return value === "similarity" || value === "mmr" || value === "hybrid";
}

function getRetrievalMode(): RetrievalMode {
  const cliModeArg = process.argv
    .slice(2)
    .find((arg) => arg.startsWith("--retrieval-mode=") || arg.startsWith("--mode="));

  const cliValue = cliModeArg?.split("=")[1]?.toLowerCase();
  if (cliValue && isRetrievalMode(cliValue)) {
    return cliValue;
  }

  const envValue = process.env.RETRIEVAL_MODE?.toLowerCase();
  if (envValue && isRetrievalMode(envValue)) {
    return envValue;
  }

  return DEFAULT_RETRIEVAL_MODE;
}

function shouldForceReindex(): boolean {
  return (
    process.argv.includes("--reindex") ||
    process.env.FORCE_REINDEX?.toLowerCase() === "true"
  );
}

export async function setupRAGPipeline() {
  try {
    const retrievalMode = getRetrievalMode();
    const { llm, provider, model } = createChatModel({ temperature: 0 });

    console.log(`🔧 Active retrieval mode: ${retrievalMode}`);
    console.log(`🧠 Embedding model: ${EMBEDDING_MODEL}`);
    console.log(`🤖 LLM provider: ${provider}`);
    console.log(`🤖 LLM model: ${model}`);

    const userInput = (await promptUser("🔍 Enter your query: ")).trim();
    if (!userInput) {
      throw new Error("Query cannot be empty.");
    }

    console.log(`\n📡 Searching for: "${userInput}"\n`);

    const cacheEnabled = CACHE_ENABLED_MODES.includes(retrievalMode);
    const cacheKey = buildCacheKey(retrievalMode, userInput);
    const queryResponseCache: QueryResponseCache = cacheEnabled
      ? await loadQueryResponseCache(QUERY_RESPONSE_CACHE_PATH)
      : {};

    if (cacheEnabled) {
      const cachedResponse = getCachedResponse(
        queryResponseCache,
        cacheKey,
        QUERY_RESPONSE_CACHE_TTL_MS
      );

      if (cachedResponse) {
        console.log("⚡ Cache hit: returning stored response\n");
        console.log("=".repeat(60));
        console.log("💬 Model Response (from cache):");
        console.log("=".repeat(60));
        console.log(cachedResponse);
        console.log("=".repeat(60));

        return {
          pages: [],
          chunks: [],
          vectorStore: null,
          hfEmbeddings: null,
          relevantChunks: [],
          response: cachedResponse,
        };
      }

      console.log("🗃️ Cache miss: running retrieval and generation\n");
    }

    console.log("📚 Loading documents from docs directory...");
    const pages = await loadDocumentsFromDirectory(DOCS_DIRECTORY_PATH);

    if (pages.length === 0) {
      throw new Error(`No readable documents found in ${DOCS_DIRECTORY_PATH}`);
    }

    console.log(`✅ Loaded ${pages.length} documents from docs directory\n`);

    console.log("✂️  Splitting documents into chunks...");
    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 500,
      chunkOverlap: 100,
    });

    const chunks = await textSplitter.splitDocuments(pages);
    if (chunks.length === 0) {
      throw new Error("Document loading completed, but no chunks were created.");
    }

    const vectorReadyChunks = chunks.map(
      (chunk) =>
        new Document({
          pageContent: chunk.pageContent,
          metadata: sanitizeMetadata((chunk.metadata ?? {}) as Record<string, unknown>),
        })
    );

    console.log(`✅ Created ${chunks.length} chunks from documents\n`);
    console.log("📋 First chunk preview:");
    console.log(`${chunks[0].pageContent.substring(0, 200)}...\n`);

    console.log("🧠 Initializing HuggingFace embeddings model...");
    const hfEmbeddings = new HuggingFaceTransformersEmbeddings({
      model: EMBEDDING_MODEL,
    });
    console.log("✅ Embeddings model initialized\n");

    console.log("🗄️  Creating/loading FAISS vector database...");
    const vectorStore = await loadOrCreateFaissVectorStore(
      vectorReadyChunks,
      hfEmbeddings,
      shouldForceReindex()
    );

    console.log("✅ FAISS vector database ready\n");
    console.log("📊 Vector Store Summary:");
    console.log(`   - Index Path: ${FAISS_INDEX_PATH}`);
    console.log(`   - Documents: ${chunks.length}`);
    console.log(`   - Embedding Model: ${EMBEDDING_MODEL}\n`);

    let candidateChunks: Document[] = [];

    if (retrievalMode === "hybrid") {
      const bm25Retriever = BM25Retriever.fromDocuments(vectorReadyChunks, {
        k: HYBRID_BM25_K,
      });
      const denseRetriever = vectorStore.asRetriever({
        searchType: "similarity",
        k: HYBRID_VECTOR_K,
      });
      const hybridRetriever = new EnsembleRetriever({
        retrievers: [bm25Retriever, denseRetriever],
        weights: HYBRID_WEIGHTS,
      });

      console.log(
        `🔎 Retrieval mode: hybrid (bm25_k=${HYBRID_BM25_K}, vector_k=${HYBRID_VECTOR_K}, weights=[${HYBRID_WEIGHTS[0]}, ${HYBRID_WEIGHTS[1]}])`
      );

      candidateChunks = await hybridRetriever.invoke(userInput);
    } else {
      candidateChunks = await vectorStore.similaritySearch(
        userInput,
        retrievalMode === "mmr" ? MMR_FETCH_K : SIMILARITY_K
      );

      console.log(
        `🔎 Retrieval mode: ${retrievalMode} (k=${retrievalMode === "mmr" ? MMR_FETCH_K : SIMILARITY_K}${retrievalMode === "mmr" ? `, localMMR topK=${MMR_TOP_K}, lambda=${MMR_LAMBDA}` : ""})`
      );
    }

    console.log(`✅ Retrieved ${candidateChunks.length} candidate chunks\n`);

    const selectedCandidates =
      retrievalMode === "mmr"
        ? await selectMMRChunks(
            userInput,
            candidateChunks,
            hfEmbeddings,
            MMR_TOP_K,
            MMR_LAMBDA
          )
        : candidateChunks.slice(
            0,
            retrievalMode === "hybrid" ? HYBRID_TOP_K : SIMILARITY_K
          );

    console.log(
      `✅ Selected ${selectedCandidates.length} chunks using ${retrievalMode === "mmr" ? "local MMR" : retrievalMode === "hybrid" ? "hybrid retrieval" : "similarity"}\n`
    );

    const thresholdedCandidates = await applySimilarityThreshold(
      userInput,
      selectedCandidates,
      hfEmbeddings,
      SIMILARITY_SCORE_THRESHOLD
    );

    console.log(
      `✅ Kept ${thresholdedCandidates.length} chunks after similarity_score_threshold=${SIMILARITY_SCORE_THRESHOLD}\n`
    );

    console.log("🧠 Reranking candidate chunks with embedding similarity...");
    const rankedChunks = await rerankChunks(
      userInput,
      thresholdedCandidates,
      hfEmbeddings
    );

    const relevantChunks = rankedChunks
      .filter((item) => item.score >= RERANK_SCORE_THRESHOLD)
      .slice(0, 3)
      .map((item) => item.chunk);

    console.log("✅ Reranking complete. Top chunks:");
    rankedChunks.slice(0, 3).forEach((item, index) => {
      console.log(
        `   ${index + 1}. score=${item.score.toFixed(4)} page=${item.chunk.metadata?.pageNumber ?? "unknown"}`
      );
    });

    console.log(
      `✅ Kept ${relevantChunks.length} chunks after score_threshold=${RERANK_SCORE_THRESHOLD}\n`
    );

    relevantChunks.forEach((chunk, index) => {
      console.log(`--- Chunk ${index + 1} ---`);
      const sourceLabel =
        typeof chunk.metadata?.fileName === "string"
          ? chunk.metadata.fileName
          : typeof chunk.metadata?.source === "string"
            ? chunk.metadata.source
            : "unknown";
      const pageLabel =
        typeof chunk.metadata?.pageNumber === "number" && chunk.metadata.pageNumber >= 0
          ? `, Page ${chunk.metadata.pageNumber}`
          : "";

      console.log(`📄 Source: ${sourceLabel}${pageLabel}`);
      console.log(`📝 Content:\n${chunk.pageContent}\n`);
    });

    const finalContext = relevantChunks.map((chunk) => chunk.pageContent).join("\n");
    const prompt = `
You are an expert in analyzing user queries. Refer to the below user query and context to build a response.
If the context is not aligned with the user query, reply with 'No context found'.
Do not use your own knowledge to build the response.
Return only the final answer in a concise form.
Do not include thinking process, analysis steps, or internal reasoning.

User query: ${userInput}

Context: ${finalContext}
    `;

    console.log(`🤖 Calling ${provider} model: ${model}...\n`);

    const llmResponse = await llm.invoke(prompt);
    const response = extractResponseText(llmResponse.content);

    console.log("=".repeat(60));
    console.log("💬 Model Response:");
    console.log("=".repeat(60));
    console.log(response);
    console.log("=".repeat(60));

    if (cacheEnabled) {
      queryResponseCache[cacheKey] = {
        response,
        createdAt: new Date().toISOString(),
      };
      await saveQueryResponseCache(QUERY_RESPONSE_CACHE_PATH, queryResponseCache);
      console.log("💾 Saved response to query-response cache\n");
    }

    return {
      pages,
      chunks,
      vectorStore,
      hfEmbeddings,
      relevantChunks,
      response,
    };
  } catch (error) {
    console.error(
      "❌ Error:",
      error instanceof Error ? error.message : String(error)
    );
    process.exit(1);
  }
}

function extractResponseText(
  content: string | Array<{ type?: string; text?: string }>
): string {
  if (typeof content === "string") {
    return content;
  }

  return content
    .map((item) => (typeof item.text === "string" ? item.text : ""))
    .join("")
    .trim();
}

async function loadOrCreateFaissVectorStore(
  documents: Document[],
  embeddings: HuggingFaceTransformersEmbeddings,
  forceReindex: boolean
): Promise<FaissStore> {
  const hasIndex = await hasPersistedFaissIndex(FAISS_INDEX_PATH);

  if (!forceReindex && hasIndex) {
    console.log("📦 Loading existing FAISS index from disk...");
    return await FaissStore.load(FAISS_INDEX_PATH, embeddings);
  }

  console.log(
    forceReindex
      ? "♻️  Rebuilding FAISS index because reindex was requested..."
      : "🆕 Building new FAISS index..."
  );

  const vectorStore = await FaissStore.fromDocuments(documents, embeddings);
  await fs.mkdir(FAISS_INDEX_PATH, { recursive: true });
  await vectorStore.save(FAISS_INDEX_PATH);

  console.log("💾 Saved FAISS index to disk");
  return vectorStore;
}

async function hasPersistedFaissIndex(indexPath: string): Promise<boolean> {
  try {
    const files = await fs.readdir(indexPath);
    return files.length > 0;
  } catch {
    return false;
  }
}

async function rerankChunks(
  query: string,
  chunks: Document[],
  embeddings: HuggingFaceTransformersEmbeddings
): Promise<ChunkScore[]> {
  const queryEmbedding = await embeddings.embedQuery(query);
  const chunkEmbeddings = await embeddings.embedDocuments(
    chunks.map((chunk) => chunk.pageContent)
  );

  const scored = chunks.map((chunk, index) => ({
    chunk,
    score: cosineSimilarity(queryEmbedding, chunkEmbeddings[index] ?? []),
  }));

  return scored.sort((a, b) => b.score - a.score);
}

async function applySimilarityThreshold(
  query: string,
  chunks: Document[],
  embeddings: HuggingFaceTransformersEmbeddings,
  threshold: number
): Promise<Document[]> {
  const queryEmbedding = await embeddings.embedQuery(query);
  const chunkEmbeddings = await embeddings.embedDocuments(
    chunks.map((chunk) => chunk.pageContent)
  );

  return chunks.filter((chunk, index) => {
    const similarity = cosineSimilarity(
      queryEmbedding,
      chunkEmbeddings[index] ?? []
    );
    chunk.metadata = {
      ...chunk.metadata,
      similarityScore: similarity,
    };
    return similarity >= threshold;
  });
}

async function selectMMRChunks(
  query: string,
  chunks: Document[],
  embeddings: HuggingFaceTransformersEmbeddings,
  topK: number,
  lambda: number
): Promise<Document[]> {
  if (chunks.length <= topK) {
    return chunks;
  }

  const queryEmbedding = await embeddings.embedQuery(query);
  const chunkEmbeddings = await embeddings.embedDocuments(
    chunks.map((chunk) => chunk.pageContent)
  );

  const selectedIndexes: number[] = [];
  const remainingIndexes = new Set<number>(chunks.map((_, index) => index));

  while (selectedIndexes.length < topK && remainingIndexes.size > 0) {
    let bestIndex = -1;
    let bestScore = Number.NEGATIVE_INFINITY;

    for (const candidateIndex of remainingIndexes) {
      const relevance = cosineSimilarity(
        queryEmbedding,
        chunkEmbeddings[candidateIndex] ?? []
      );

      const diversity =
        selectedIndexes.length === 0
          ? 0
          : Math.max(
              ...selectedIndexes.map((selectedIndex) =>
                cosineSimilarity(
                  chunkEmbeddings[candidateIndex] ?? [],
                  chunkEmbeddings[selectedIndex] ?? []
                )
              )
            );

      const mmrScore = lambda * relevance - (1 - lambda) * diversity;

      if (mmrScore > bestScore) {
        bestScore = mmrScore;
        bestIndex = candidateIndex;
      }
    }

    if (bestIndex === -1) {
      break;
    }

    selectedIndexes.push(bestIndex);
    remainingIndexes.delete(bestIndex);
  }

  return selectedIndexes.map((index) => chunks[index]);
}

function cosineSimilarity(a: number[], b: number[]): number {
  if (!a.length || !b.length || a.length !== b.length) {
    return 0;
  }

  let dotProduct = 0;
  let magnitudeA = 0;
  let magnitudeB = 0;

  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    magnitudeA += a[i] * a[i];
    magnitudeB += b[i] * b[i];
  }

  const denominator = Math.sqrt(magnitudeA) * Math.sqrt(magnitudeB);
  return denominator === 0 ? 0 : dotProduct / denominator;
}

function promptUser(question: string): Promise<string> {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  return new Promise((resolve) => {
    rl.question(question, (answer) => {
      rl.close();
      resolve(answer);
    });
  });
}

async function loadDocumentsFromDirectory(directoryPath: string): Promise<Document[]> {
  const entries = await fs.readdir(directoryPath, { withFileTypes: true });
  const files = entries
    .filter((entry) => entry.isFile())
    .map((entry) => entry.name)
    .sort((a, b) => a.localeCompare(b));

  const loadedDocuments = await Promise.all(
    files.map(async (fileName) => {
      const filePath = path.join(directoryPath, fileName);
      const extension = path.extname(fileName).toLowerCase();

      try {
        if (extension === ".pdf") {
          const loader = new PDFLoader(filePath);
          return await loader.load();
        }

        const content = await fs.readFile(filePath, "utf-8");
        return [
          new Document({
            pageContent: content,
            metadata: {
              source: filePath,
              fileName,
              fileType: extension || "unknown",
            },
          }),
        ];
      } catch (error) {
        console.warn(
          `⚠️  Skipping unreadable file: ${fileName} (${error instanceof Error ? error.message : String(error)})`
        );
        return [];
      }
    })
  );

  return loadedDocuments.flat();
}

function sanitizeMetadata(
  metadata: Record<string, unknown> = {}
): Record<string, string | number | boolean | null> {
  const pageNumber =
    typeof metadata.loc === "object" && metadata.loc !== null
      ? ((metadata.loc as { pageNumber?: unknown }).pageNumber as unknown)
      : undefined;

  return {
    source: typeof metadata.source === "string" ? metadata.source : "unknown",
    fileName: typeof metadata.fileName === "string" ? metadata.fileName : null,
    fileType: typeof metadata.fileType === "string" ? metadata.fileType : null,
    pageNumber: typeof pageNumber === "number" ? pageNumber : -1,
    pdf: metadata.pdf ? JSON.stringify(metadata.pdf) : null,
  };
}

function normalizeQuery(query: string): string {
  return query.trim().toLowerCase().replace(/\s+/g, " ");
}

function resolveCacheModelKey(): string {
  const provider = process.env.LLM_PROVIDER?.toLowerCase() === "azure" ? "azure" : "groq";

  if (provider === "azure") {
    const deployment = process.env.AZURE_OPENAI_DEPLOYMENT ?? "unknown-azure-deployment";
    return `${provider}:${deployment}`;
  }

  const model = process.env.GROQ_MODEL ?? "llama-3.3-70b-versatile";
  return `${provider}:${model}`;
}

function buildCacheKey(retrievalMode: RetrievalMode, query: string): string {
  return `${retrievalMode}::${resolveCacheModelKey()}::${EMBEDDING_MODEL}::${normalizeQuery(query)}`;
}

async function loadQueryResponseCache(
  cachePath: string
): Promise<QueryResponseCache> {
  try {
    const content = await fs.readFile(cachePath, "utf-8");
    const parsed = JSON.parse(content) as QueryResponseCache;
    return parsed ?? {};
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") {
      return {};
    }

    console.warn(
      `⚠️  Failed to read query cache. Starting with empty cache. (${error instanceof Error ? error.message : String(error)})`
    );
    return {};
  }
}

async function saveQueryResponseCache(
  cachePath: string,
  cache: QueryResponseCache
): Promise<void> {
  const dirPath = path.dirname(cachePath);
  await fs.mkdir(dirPath, { recursive: true });
  await fs.writeFile(cachePath, JSON.stringify(cache, null, 2), "utf-8");
}

function getCachedResponse(
  cache: QueryResponseCache,
  cacheKey: string,
  ttlMs: number
): string | null {
  const entry = cache[cacheKey];
  if (!entry) {
    return null;
  }

  const createdAt = new Date(entry.createdAt).getTime();
  if (Number.isNaN(createdAt)) {
    return null;
  }

  if (Date.now() - createdAt > ttlMs) {
    return null;
  }

  return entry.response;
}

if (require.main === module) {
  void setupRAGPipeline();
}
