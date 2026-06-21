import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/huggingface_transformers";
import { Chroma } from "@langchain/community/vectorstores/chroma";
import { Ollama } from "@langchain/ollama";
import { Document } from "@langchain/core/documents";
import * as readline from "readline";

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

    // Performing similarity search (equivalent to db.as_retriever)
    const retriever = vectorStore.asRetriever({
      searchType: "similarity",
      k: 3,
    });

    const relevantChunks = await retriever.invoke(userInput);

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

        User query: ${userInput}

        Context: ${finalContext}
        `;

    // Call Ollama qwen model
    console.log("🤖 Calling Ollama qwen3.5 model...\n");
    const llm = new Ollama({
      baseUrl: "http://localhost:11434",
      model: "qwen3.5",
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
