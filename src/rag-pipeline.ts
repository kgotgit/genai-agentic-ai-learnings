import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/huggingface_transformers";
import { Chroma } from "@langchain/community/vectorstores/chroma";
import { Ollama } from "@langchain/ollama";
import { Document } from "@langchain/core/documents";
import * as readline from "readline";

const SIMILARITY_SCORE_THRESHOLD = 0.25;
const RERANK_SCORE_THRESHOLD = 0.20;
const RETRIEVAL_MODE: "similarity" | "mmr" = "mmr";
const MMR_FETCH_K = 20;
const MMR_LAMBDA = 0.5;
const MMR_TOP_K = 5;

async function setupRAGPipeline() {
  try {
    console.log("📚 Loading PDF document...");

    // Load PDF
    const pdfPath =
      "/Users/karthikgotrala/Documents/AgileFeverAICourse/genai-agentic-ai-learnings/docs/1706.03762v7.pdf";
    const loader = new PDFLoader(pdfPath);
    const pages = await loader.load();

    console.log(`✅ Loaded ${pages.length} pages from PDF\n`);

    // Text splitting
    console.log("✂️  Splitting documents into chunks...");
    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 500,
      chunkOverlap: 100,
    });

    const chunks = await textSplitter.splitDocuments(pages);
    const chromaReadyChunks = chunks.map(
      (chunk) =>
        new Document({
          pageContent: chunk.pageContent,
          metadata: sanitizeMetadata(chunk.metadata),
        })
    );

    console.log(`✅ Created ${chunks.length} chunks from documents\n`);
    console.log("📋 First chunk preview:");
    console.log(chunks[0].pageContent.substring(0, 200) + "...\n");

    // Get embedding model
    console.log("🧠 Initializing HuggingFace embeddings model...");
    const hfEmbeddings = new HuggingFaceTransformersEmbeddings({
      model: "Xenova/all-MiniLM-L6-v2",
    });

    console.log("✅ Embeddings model initialized\n");

    // Create vector database with Chroma
    console.log("🗄️  Creating Chroma vector database...");
    const vectorStore = await Chroma.fromDocuments(chromaReadyChunks, hfEmbeddings, {
      collectionName: "pdf_documents",
      url: "http://localhost:8000",
      collectionMetadata: {
        source: "transformer_paper",
      },
    });

    console.log("✅ Vector database created and persisted\n");
    console.log("📊 Vector Store Summary:");
    console.log(`   - Collection: pdf_documents`);
    console.log(`   - Documents: ${chunks.length}`);
    console.log(`   - Embedding Model: Xenova/all-MiniLM-L6-v2\n`);

    // Take user input and perform similarity search
    const userInput = await promptUser("🔍 Enter your query: ");

    console.log(`\n📡 Searching for: "${userInput}"\n`);

    // Performing retrieval to fetch candidate chunks
    const retriever = vectorStore.asRetriever({
      searchType: "similarity",
      k: RETRIEVAL_MODE === "mmr" ? MMR_FETCH_K : 5,
    });

    console.log(
      `🔎 Retrieval mode: ${RETRIEVAL_MODE} (k=${RETRIEVAL_MODE === "mmr" ? MMR_FETCH_K : 5}${RETRIEVAL_MODE === "mmr" ? `, localMMR topK=${MMR_TOP_K}, lambda=${MMR_LAMBDA}` : ""})`
    );

    const candidateChunks = await retriever.invoke(userInput);

    console.log(`✅ Retrieved ${candidateChunks.length} candidate chunks\n`);

    const selectedCandidates =
      RETRIEVAL_MODE === "mmr"
        ? await selectMMRChunks(
            userInput,
            candidateChunks,
            hfEmbeddings,
            MMR_TOP_K,
            MMR_LAMBDA
          )
        : candidateChunks.slice(0, 5);

    console.log(
      `✅ Selected ${selectedCandidates.length} chunks using ${RETRIEVAL_MODE === "mmr" ? "local MMR" : "similarity"}\n`
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

    // Rerank chunks using embedding similarity scores
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

    console.log(`✅ Found ${relevantChunks.length} relevant chunks\n`);

    relevantChunks.forEach((chunk, index) => {
      console.log(`--- Chunk ${index + 1} ---`);
      console.log(`📄 Source: Page ${chunk.metadata?.loc?.pageNumber ?? "unknown"}`);
      console.log(`📝 Content:\n${chunk.pageContent}\n`);
    });

    // Prepare final context from retrieved chunks
    const finalContext = relevantChunks.map((chunk) => chunk.pageContent).join("\n");

    // Build prompt
    const prompt = `
        You are an expert in analyzing user queries. Refer to the below user query and context to build a response.
        If the context is not aligned with the user query, reply with 'No context found'.
        Do not use your own knowledge to build the response.
      Return only the final answer in a concise form.
      Do not include thinking process, analysis steps, or internal reasoning.

        User query: ${userInput}

        Context: ${finalContext}
        `;

    // Call Ollama qwen model
    console.log("🤖 Calling Ollama qwen3.5 model...\n");
    const llm = new Ollama({
      baseUrl: "http://localhost:11434",
      model: "qwen3.5",
      think: false,
    });

    const response = await llm.invoke(prompt);

    console.log("=".repeat(60));
    console.log("💬 Model Response:");
    console.log("=".repeat(60));
    console.log(response);
    console.log("=".repeat(60));

    return { pages, chunks, vectorStore, hfEmbeddings, relevantChunks, response };
  } catch (error) {
    console.error(
      "❌ Error:",
      error instanceof Error ? error.message : String(error)
    );
    process.exit(1);
  }
}

type ChunkScore = {
  chunk: Document;
  score: number;
};

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

function sanitizeMetadata(metadata: Record<string, unknown>): Record<string, string | number | boolean | null> {
  const pageNumber =
    typeof metadata?.loc === "object" && metadata.loc !== null
      ? ((metadata.loc as { pageNumber?: unknown }).pageNumber as unknown)
      : undefined;

  return {
    source: typeof metadata?.source === "string" ? metadata.source : "unknown",
    pageNumber: typeof pageNumber === "number" ? pageNumber : -1,
    pdf: metadata?.pdf ? JSON.stringify(metadata.pdf) : null,
  };
}

setupRAGPipeline();
