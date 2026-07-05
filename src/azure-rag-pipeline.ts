import * as dotenv from "dotenv";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import {
  AzureAISearchQueryType,
  AzureAISearchVectorStore,
} from "@langchain/community/vectorstores/azure_aisearch";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { AzureChatOpenAI, AzureOpenAIEmbeddings } from "@langchain/openai";
import { Document } from "@langchain/core/documents";
import { promises as fs } from "fs";
import * as path from "path";
import * as readline from "readline";

dotenv.config();

const DOCS_DIRECTORY_PATH = path.resolve(process.cwd(), "docs");

type AzureConfig = {
  endpoint: string;
  apiKey: string;
  apiVersion: string;
  embeddingDeployment: string;
  chatDeployment: string;
  searchEndpoint: string;
  searchApiKey: string;
  searchIndexName: string;
};

function requireEnv(name: string, value: string | undefined): string {
  if (!value || value.trim().length === 0) {
    throw new Error(`Missing ${name} environment variable.`);
  }
  return value.trim();
}

function normalizeAzureEndpoint(rawValue: string): string {
  const withoutQuery = rawValue.trim().replace(/\?.*$/, "").replace(/\/+$/, "");
  const lower = withoutQuery.toLowerCase();

  if (lower.includes("/openai/deployments/")) {
    return withoutQuery.slice(0, lower.indexOf("/openai/deployments/"));
  }

  if (lower.endsWith("/openai/v1")) {
    return withoutQuery.slice(0, -"/openai/v1".length);
  }

  if (lower.endsWith("/openai")) {
    return withoutQuery.slice(0, -"/openai".length);
  }

  return withoutQuery;
}

function resolveAzureConfig(): AzureConfig {
  const endpoint = normalizeAzureEndpoint(
    requireEnv(
      "AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_BASE_URL",
      process.env.AZURE_OPENAI_ENDPOINT ?? process.env.AZURE_OPENAI_BASE_URL
    )
  );

  return {
    endpoint,
    apiKey: requireEnv(
      "AZURE_OPENAI_API_KEY or AZURE_OPENAI_API_EMBEDDING_KEY",
      process.env.AZURE_OPENAI_API_KEY ??
        process.env.AZURE_OPENAI_API_EMBEDDING_KEY ??
        process.env.AZURE_API_KEY
    ),
    apiVersion: process.env.AZURE_OPENAI_API_VERSION ?? "2023-05-15",
    embeddingDeployment:
      process.env.AZURE_OPENAI_EMBEDDING_DEPLOYMENT ?? "text-embedding-3-small",
    chatDeployment:
      process.env.AZURE_OPENAI_CHAT_DEPLOYMENT ??
      process.env.AZURE_OPENAI_DEPLOYMENT ??
      "gpt-5.4-mini",
    searchEndpoint: requireEnv(
      "AZURE_AISEARCH_ENDPOINT or AZURE_SEARCH_ENDPOINT",
      process.env.AZURE_AISEARCH_ENDPOINT ?? process.env.AZURE_SEARCH_ENDPOINT
    ),
    searchApiKey: requireEnv(
      "AZURE_AISEARCH_KEY or AZURE_SEARCH_API_KEY",
      process.env.AZURE_AISEARCH_KEY ?? process.env.AZURE_SEARCH_API_KEY
    ),
    searchIndexName:
      process.env.AZURE_AISEARCH_INDEX_NAME ?? process.env.AZURE_SEARCH_INDEX_NAME ?? "practice-demo",
  };
}

function buildAzureEmbeddings(config: AzureConfig): AzureOpenAIEmbeddings {
  return new AzureOpenAIEmbeddings({
    azureOpenAIApiKey: config.apiKey,
    azureOpenAIEndpoint: config.endpoint,
    azureOpenAIApiVersion: config.apiVersion,
    azureOpenAIApiDeploymentName: config.embeddingDeployment,
    deploymentName: config.embeddingDeployment,
  });
}

function buildAzureChatModel(config: AzureConfig): AzureChatOpenAI {
  return new AzureChatOpenAI(config.chatDeployment, {
    azureOpenAIApiKey: config.apiKey,
    azureOpenAIEndpoint: config.endpoint,
    azureOpenAIApiVersion: config.apiVersion,
    temperature: 0,
  });
}

function buildAzureSearchVectorStore(
  embeddings: AzureOpenAIEmbeddings,
  config: AzureConfig
): AzureAISearchVectorStore {
  return new AzureAISearchVectorStore(embeddings, {
    endpoint: config.searchEndpoint,
    key: config.searchApiKey,
    indexName: config.searchIndexName,
    search: {
      type: AzureAISearchQueryType.SimilarityHybrid,
    },
  });
}

async function loadDocumentsFromDirectory(directoryPath: string): Promise<Document[]> {
  const entries = await fs.readdir(directoryPath, { withFileTypes: true });
  const loadedDocuments = await Promise.all(
    entries.map(async (entry) => {
      const filePath = path.join(directoryPath, entry.name);

      if (entry.isDirectory()) {
        return await loadDocumentsFromDirectory(filePath);
      }

      const extension = path.extname(entry.name).toLowerCase();

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
              fileName: entry.name,
              fileType: extension || "unknown",
            },
          }),
        ];
      } catch (error) {
        console.warn(
          `⚠️  Skipping unreadable file: ${entry.name} (${error instanceof Error ? error.message : String(error)})`
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

async function promptUser(question: string): Promise<string> {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  return await new Promise<string>((resolve) => {
    rl.question(question, (answer) => {
      rl.close();
      resolve(answer);
    });
  });
}

export async function setupAzureRAGPipeline() {
  try {
    const config = resolveAzureConfig();

    console.log("🧠 Initializing Azure OpenAI embeddings...");
    console.log(`   - Azure endpoint: ${config.endpoint}`);
    console.log(`   - Embedding deployment: ${config.embeddingDeployment}`);
    console.log(`   - API version: ${config.apiVersion}`);

    const embeddings = buildAzureEmbeddings(config);
    const llm = buildAzureChatModel(config);

    console.log("📚 Loading documents from docs directory...");
    const pages = await loadDocumentsFromDirectory(DOCS_DIRECTORY_PATH);

    if (pages.length === 0) {
      throw new Error(`No readable documents found in ${DOCS_DIRECTORY_PATH}`);
    }

    console.log(`✅ Loaded ${pages.length} documents\n`);

    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 500,
      chunkOverlap: 100,
    });

    const chunks = await textSplitter.splitDocuments(pages);
    if (chunks.length === 0) {
      throw new Error("No chunks were created from the loaded documents.");
    }

    const vectorReadyChunks = chunks.map(
      (chunk) =>
        new Document({
          pageContent: chunk.pageContent,
          metadata: sanitizeMetadata((chunk.metadata ?? {}) as Record<string, unknown>),
        })
    );

    console.log(`✅ Created ${vectorReadyChunks.length} chunks`);
    console.log("🗄️  Connecting to Azure AI Search vector store...");
    console.log(`   - Search endpoint: ${config.searchEndpoint}`);
    console.log(`   - Search index: ${config.searchIndexName}`);

    const vectorStore = buildAzureSearchVectorStore(embeddings, config);

    console.log("⬆️ Uploading chunks to Azure AI Search...");
    await vectorStore.addDocuments(vectorReadyChunks);
    console.log("✅ Upload complete\n");

    const userInput = (await promptUser("🔍 Enter your query: ")).trim();
    if (!userInput) {
      throw new Error("Query cannot be empty.");
    }

    console.log(`\n📡 Retrieving from Azure AI Search for: "${userInput}"\n`);
    const relevantChunks = await vectorStore.similaritySearch(userInput, 5);

    if (relevantChunks.length === 0) {
      console.log("No context found");
      return;
    }

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

    console.log("🤖 Calling Azure chat model...\n");
    const llmResponse = await llm.invoke(prompt);
    const response = extractResponseText(llmResponse.content);

    console.log("=".repeat(60));
    console.log("💬 Model Response:");
    console.log("=".repeat(60));
    console.log(response);
    console.log("=".repeat(60));
  } catch (error) {
    console.error(
      "❌ Error:",
      error instanceof Error ? error.message : String(error)
    );
    process.exit(1);
  }
}

void setupAzureRAGPipeline();
