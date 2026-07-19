import { AIMessage, BaseMessage, HumanMessage, SystemMessage } from "@langchain/core/messages";
import { StateGraph, MessagesAnnotation, START, END } from "@langchain/langgraph";
import { TavilySearchAPIWrapper } from "@langchain/tavily";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/huggingface_transformers";
import { ChromaClient, Collection } from "chromadb";
import { promises as fs } from "fs";
import * as path from "path";
import * as dotenv from "dotenv";
import { createChatModel } from "../llm";

dotenv.config();

export type RouteType = "web_search" | "chroma_rag" | "general_llm";

export type WorkflowStep = {
  node: string;
  status: "ok" | "error";
  startedAt: string;
  durationMs: number;
  detail: string;
};

export type SourceItem = {
  title: string;
  url?: string;
  snippet: string;
};

export type ChatResult = {
  response: string;
  route: RouteType;
  workflow: WorkflowStep[];
  sources: SourceItem[];
  traceId: string;
};

type ThreadStore = Record<string, BaseMessage[]>;

type RouteDecision = {
  route: RouteType;
  confidence: number;
  reason: string;
};

type ChromaMatch = {
  source: string;
  content: string;
  score: number;
};

const ROUTE_TAG = "[ROUTE_DECISION]";
const NODE_TAG = "[NODE_OUTPUT]";
const TRACE_TAG = "[TRACE]";

const CHROMA_URL = process.env.CHROMA_URL ?? "http://localhost:8000";
const CHROMA_COLLECTION = process.env.CHROMA_COLLECTION ?? "agentic_rag_docs";
const DOCS_PATH = path.resolve(process.cwd(), "docs");
const EMBEDDING_MODEL = process.env.EMBEDDING_MODEL ?? "Xenova/bge-m3";
const MIN_CHROMA_SCORE = 0.25;
const TAVILY_API_KEY = process.env.TAVILY_API_KEY?.trim() ?? "";

const threadStore: ThreadStore = {};
const workflowTraceStore: Record<string, WorkflowStep[]> = {};
const sourceStore: Record<string, SourceItem[]> = {};

const tavily = TAVILY_API_KEY
  ? new TavilySearchAPIWrapper({ tavilyApiKey: TAVILY_API_KEY })
  : null;

type LlmBundle = ReturnType<typeof createChatModel>;

let llmBundle: LlmBundle | null = null;
let llmInitError: string | null = null;

/**
 * Lazily initializes and returns the configured chat model bundle.
 * Returns null when initialization fails and caches the failure message.
 */
function getLlmBundle(): LlmBundle | null {
  if (llmBundle) {
    return llmBundle;
  }

  if (llmInitError) {
    return null;
  }

  try {
    llmBundle = createChatModel({ temperature: 0.2 });
    return llmBundle;
  } catch (error) {
    llmInitError = error instanceof Error ? error.message : "LLM initialization failed.";
    return null;
  }
}

const embeddings = new HuggingFaceTransformersEmbeddings({
  model: EMBEDDING_MODEL,
});

const chromaClient = new ChromaClient({ path: CHROMA_URL });
let cachedCollection: Collection | null = null;
let indexedAtLeastOnce = false;
let cachedDocTopics: string[] | null = null;

/**
 * Ensures a thread exists in memory and returns its message history.
 */
function ensureThread(threadId: string): BaseMessage[] {
  if (!threadStore[threadId]) {
    threadStore[threadId] = [];
  }
  return threadStore[threadId];
}

/**
 * Parses a prefixed JSON payload encoded in a system message.
 */
function parseJsonPayload<T>(content: string, prefix: string): T | null {
  if (!content.startsWith(prefix)) {
    return null;
  }

  const raw = content.slice(prefix.length).trim();

  try {
    return JSON.parse(raw) as T;
  } catch {
    return null;
  }
}

/**
 * Creates a system message with a prefix tag and JSON payload.
 */
function buildSystemMessage(prefix: string, payload: unknown): SystemMessage {
  return new SystemMessage(`${prefix} ${JSON.stringify(payload)}`);
}

/**
 * Extracts the most recent route decision from graph system messages.
 */
function extractRouteDecision(messages: BaseMessage[]): RouteDecision {
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    const msg = messages[i];
    if (msg.getType() !== "system") {
      continue;
    }

    const parsed = parseJsonPayload<RouteDecision>(String(msg.content), ROUTE_TAG);
    if (parsed) {
      return parsed;
    }
  }

  return {
    route: "general_llm",
    confidence: 0.5,
    reason: "No route payload found; defaulting to general_llm.",
  };
}

/**
 * Extracts node output content for a specific node from system messages.
 */
function extractNodeOutput(messages: BaseMessage[], node: string): string {
  const marker = `${NODE_TAG}:${node}`;
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    const msg = messages[i];
    if (msg.getType() !== "system") {
      continue;
    }

    const payload = parseJsonPayload<{ node: string; content: string }>(
      String(msg.content),
      marker
    );
    if (payload) {
      return payload.content;
    }
  }

  return "";
}

/**
 * Collects all workflow trace entries from system messages.
 */
function extractTrace(messages: BaseMessage[]): WorkflowStep[] {
  const traces: WorkflowStep[] = [];

  for (const msg of messages) {
    if (msg.getType() !== "system") {
      continue;
    }

    const trace = parseJsonPayload<WorkflowStep>(String(msg.content), TRACE_TAG);
    if (trace) {
      traces.push(trace);
    }
  }

  return traces;
}

/**
 * Returns the latest human query from the message history.
 */
function extractLatestUserQuery(messages: BaseMessage[]): string {
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    const msg = messages[i];
    if (msg.getType() === "human") {
      return String(msg.content);
    }
  }

  return "";
}

/**
 * Returns a millisecond timestamp for node timing.
 */
function startTimer(): number {
  return Date.now();
}

/**
 * Builds a normalized workflow trace object for a node execution.
 */
function buildTrace(
  node: string,
  status: "ok" | "error",
  startedAt: number,
  detail: string
): WorkflowStep {
  return {
    node,
    status,
    startedAt: new Date(startedAt).toISOString(),
    durationMs: Date.now() - startedAt,
    detail,
  };
}

/**
 * Loads text/markdown docs and splits them into chunks for vector indexing.
 */
async function loadDocsForIndexing(): Promise<Array<{ id: string; content: string; source: string }>> {
  const items = await fs.readdir(DOCS_PATH, { withFileTypes: true });
  const textFiles = items.filter((item) => item.isFile() && /\.(txt|md)$/i.test(item.name));

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 700,
    chunkOverlap: 120,
  });

  const chunks: Array<{ id: string; content: string; source: string }> = [];

  for (const file of textFiles) {
    const fullPath = path.join(DOCS_PATH, file.name);
    const raw = await fs.readFile(fullPath, "utf8");
    const docs = await splitter.createDocuments([raw], [{ source: file.name }]);

    docs.forEach((doc, index) => {
      chunks.push({
        id: `${file.name}-${index}`,
        content: doc.pageContent,
        source: file.name,
      });
    });
  }

  return chunks;
}

/**
 * Tokenizes a filename or topic string into normalized route-matching keywords.
 */
function splitTopicTokens(value: string): string[] {
  return value
    .toLowerCase()
    .replace(/\.[^.]+$/, "")
    .split(/[^a-z0-9]+/g)
    .map((token) => token.trim())
    .filter((token) => token.length >= 3 && token !== "company" && token !== "overview");
}

/**
 * Derives route topic keywords from docs filenames and caches them.
 */
async function getDocTopics(): Promise<string[]> {
  if (cachedDocTopics) {
    return cachedDocTopics;
  }

  try {
    const items = await fs.readdir(DOCS_PATH, { withFileTypes: true });
    const files = items.filter((item) => item.isFile()).map((item) => item.name);
    const topicSet = new Set<string>();

    files.forEach((file) => {
      splitTopicTokens(file).forEach((token) => topicSet.add(token));
    });

    cachedDocTopics = Array.from(topicSet);
    return cachedDocTopics;
  } catch {
    cachedDocTopics = [];
    return cachedDocTopics;
  }
}

/**
 * Returns or creates the Chroma collection used by this example.
 */
async function getCollection(): Promise<Collection> {
  if (cachedCollection) {
    return cachedCollection;
  }

  cachedCollection = await chromaClient.getOrCreateCollection({
    name: CHROMA_COLLECTION,
    metadata: {
      description: "Agentic RAG example collection",
    },
  });

  return cachedCollection;
}

/**
 * Builds the Chroma index lazily on first retrieval request.
 */
async function ensureChromaIndexed(): Promise<void> {
  if (indexedAtLeastOnce) {
    return;
  }

  const collection = await getCollection();
  const docs = await loadDocsForIndexing();

  if (docs.length === 0) {
    indexedAtLeastOnce = true;
    return;
  }

  const vectors = await embeddings.embedDocuments(docs.map((doc) => doc.content));

  await collection.upsert({
    ids: docs.map((doc) => doc.id),
    documents: docs.map((doc) => doc.content),
    metadatas: docs.map((doc) => ({ source: doc.source })),
    embeddings: vectors,
  });

  indexedAtLeastOnce = true;
}

/**
 * Classifies a query into one of the graph routes and records trace metadata.
 */
async function classifyNode(state: typeof MessagesAnnotation.State) {
  const startedAt = startTimer();
  const query = extractLatestUserQuery(state.messages);
  const modelBundle = getLlmBundle();
  const queryLower = query.toLowerCase();
  const docTopics = await getDocTopics();
  const matchedTopic = docTopics.find((topic) => queryLower.includes(topic));
  const knownLocalEntity = /(microsoft|apple)/.test(queryLower);
  const localIntent =
    /local doc|our doc|from doc|in doc|document|knowledge base|kb|internal/.test(queryLower) ||
    Boolean(matchedTopic) ||
    knownLocalEntity;

  if (localIntent) {
    return {
      messages: [
        buildSystemMessage(ROUTE_TAG, {
          route: "chroma_rag",
          confidence: matchedTopic || knownLocalEntity ? 0.92 : 0.85,
          reason: matchedTopic
            ? `Query matches local docs topic: ${matchedTopic}`
            : knownLocalEntity
              ? "Query references known local company entity; prefer local docs first."
            : "Query asks for local/internal document context.",
        }),
        buildSystemMessage(
          TRACE_TAG,
          buildTrace(
            "classify_query",
            "ok",
            startedAt,
            matchedTopic ? `topic=${matchedTopic}` : knownLocalEntity ? "known_local_entity" : "local_intent"
          )
        ),
      ],
    };
  }

  if (!modelBundle) {
    const route: RouteType = /latest|news|today|recent|current/.test(queryLower)
      ? tavily
        ? "web_search"
        : "general_llm"
      : /document|docs|microsoft|apple|company|overview/.test(queryLower)
        ? "chroma_rag"
        : "general_llm";

    const reason = llmInitError ?? "LLM unavailable";
    return {
      messages: [
        buildSystemMessage(ROUTE_TAG, {
          route,
          confidence: 0.45,
          reason: `Heuristic fallback router used because LLM is unavailable: ${reason}`,
        }),
        buildSystemMessage(TRACE_TAG, buildTrace("classify_query", "error", startedAt, reason)),
      ],
    };
  }

  const { llm } = modelBundle;

  const prompt = [
    "You are a query router for an agentic system.",
    "Choose one route: web_search, chroma_rag, general_llm.",
    "Return only compact JSON with keys route, confidence, reason.",
    "Rules:",
    "- web_search for current events, latest updates, news, realtime facts.",
    "- chroma_rag for questions likely answerable from local company/product documents.",
    `- Prefer chroma_rag when query mentions known local topics: ${docTopics.slice(0, 20).join(", ") || "none"}.`,
    "- general_llm for generic explanations, brainstorming, coding, and chat.",
    !tavily ? "- IMPORTANT: web_search is unavailable because Tavily API key is missing; do not choose web_search." : "",
    `Query: ${query}`,
  ].join("\n");

  try {
    const response = await llm.invoke(prompt);
    const raw = String(response.content).trim();
    const normalized = raw.startsWith("{") ? raw : raw.replace(/^```json|```$/g, "").trim();

    let parsed: RouteDecision = {
      route: "general_llm",
      confidence: 0.6,
      reason: "Default decision.",
    };

    try {
      const candidate = JSON.parse(normalized) as Partial<RouteDecision>;
      if (
        candidate.route === "web_search" ||
        candidate.route === "chroma_rag" ||
        candidate.route === "general_llm"
      ) {
        parsed = {
          route: candidate.route,
          confidence:
            typeof candidate.confidence === "number" ? Math.max(0, Math.min(1, candidate.confidence)) : 0.6,
          reason: typeof candidate.reason === "string" ? candidate.reason : "No reason provided.",
        };
      }
    } catch {
      parsed = {
        route: "general_llm",
        confidence: 0.55,
        reason: "Router JSON parse failed, defaulted to general_llm.",
      };
    }

    return {
      messages: [
        buildSystemMessage(ROUTE_TAG, parsed),
        buildSystemMessage(TRACE_TAG, buildTrace("classify_query", "ok", startedAt, `${parsed.route} (${parsed.confidence})`)),
      ],
    };
  } catch (error) {
    const detail = error instanceof Error ? error.message : "Unknown classification error";

    return {
      messages: [
        buildSystemMessage(ROUTE_TAG, {
          route: "general_llm",
          confidence: 0.4,
          reason: "Classification node failed; fallback route selected.",
        }),
        buildSystemMessage(TRACE_TAG, buildTrace("classify_query", "error", startedAt, detail)),
      ],
    };
  }
}

/**
 * Chooses the next graph edge based on the latest route decision.
 */
function routeAfterClassification(state: typeof MessagesAnnotation.State): RouteType {
  const route = extractRouteDecision(state.messages).route;
  if (route === "web_search" && !tavily) {
    return "general_llm";
  }
  return route;
}

/**
 * Executes Tavily web search and stores normalized output in system messages.
 */
async function webSearchNode(state: typeof MessagesAnnotation.State) {
  const startedAt = startTimer();
  const query = extractLatestUserQuery(state.messages);

  if (!tavily) {
    const detail = "Tavily API key is not configured; skipped web search.";
    return {
      messages: [
        buildSystemMessage(`${NODE_TAG}:web_search`, {
          node: "web_search",
          content: JSON.stringify({
            answer: "",
            results: [],
            error: detail,
          }),
        }),
        buildSystemMessage(TRACE_TAG, buildTrace("web_search", "error", startedAt, detail)),
      ],
    };
  }

  try {
    const response = await tavily.rawResults({
      query,
      max_results: 4,
      include_answer: true,
      search_depth: "basic",
      topic: "general",
    });

    const topResults = (response.results ?? []).slice(0, 4).map((item) => ({
      title: item.title ?? "Untitled",
      url: item.url,
      snippet: item.content ?? "",
    }));

    return {
      messages: [
        buildSystemMessage(`${NODE_TAG}:web_search`, {
          node: "web_search",
          content: JSON.stringify({
            answer: response.answer ?? "",
            results: topResults,
          }),
        }),
        buildSystemMessage(
          TRACE_TAG,
          buildTrace("web_search", "ok", startedAt, `results=${topResults.length}`)
        ),
      ],
    };
  } catch (error) {
    const detail = error instanceof Error ? error.message : "Unknown Tavily error";
    return {
      messages: [
        buildSystemMessage(`${NODE_TAG}:web_search`, {
          node: "web_search",
          content: JSON.stringify({
            answer: "",
            results: [],
            error: detail,
          }),
        }),
        buildSystemMessage(TRACE_TAG, buildTrace("web_search", "error", startedAt, detail)),
      ],
    };
  }
}

/**
 * Runs Chroma retrieval for the query and stores top local matches.
 */
async function chromaRagNode(state: typeof MessagesAnnotation.State) {
  const startedAt = startTimer();
  const query = extractLatestUserQuery(state.messages);

  try {
    await ensureChromaIndexed();
    const collection = await getCollection();
    const queryEmbedding = await embeddings.embedQuery(query);

    const queryResult = await collection.query({
      queryEmbeddings: [queryEmbedding],
      nResults: 4,
      include: ["documents", "metadatas", "distances"],
    });

    const docs = queryResult.documents?.[0] ?? [];
    const metas = queryResult.metadatas?.[0] ?? [];
    const distances = queryResult.distances?.[0] ?? [];

    const matches: ChromaMatch[] = docs
      .map((doc, index) => {
        if (typeof doc !== "string") {
          return null;
        }

        const distance = typeof distances[index] === "number" ? distances[index] : 1;
        const score = Math.max(0, 1 - distance);
        const metadata = metas[index] as Record<string, unknown> | undefined;

        return {
          source: typeof metadata?.source === "string" ? metadata.source : "unknown",
          content: doc,
          score,
        };
      })
      .filter((item): item is ChromaMatch => item !== null)
      .filter((item) => item.score >= MIN_CHROMA_SCORE)
      .sort((a, b) => b.score - a.score)
      .slice(0, 3);

    return {
      messages: [
        buildSystemMessage(`${NODE_TAG}:chroma_rag`, {
          node: "chroma_rag",
          content: JSON.stringify(matches),
        }),
        buildSystemMessage(
          TRACE_TAG,
          buildTrace("chroma_rag", "ok", startedAt, `matches=${matches.length}`)
        ),
      ],
    };
  } catch (error) {
    const detail = error instanceof Error ? error.message : "Unknown Chroma error";
    return {
      messages: [
        buildSystemMessage(`${NODE_TAG}:chroma_rag`, {
          node: "chroma_rag",
          content: JSON.stringify({ error: detail }),
        }),
        buildSystemMessage(TRACE_TAG, buildTrace("chroma_rag", "error", startedAt, detail)),
      ],
    };
  }
}

/**
 * Emits a lightweight marker when the direct LLM path is selected.
 */
async function generalContextNode(_state: typeof MessagesAnnotation.State) {
  const startedAt = startTimer();
  return {
    messages: [
      buildSystemMessage(`${NODE_TAG}:general_llm`, {
        node: "general_llm",
        content: "Direct LLM response path selected.",
      }),
      buildSystemMessage(TRACE_TAG, buildTrace("general_llm", "ok", startedAt, "No retrieval used")),
    ],
  };
}

/**
 * Synthesizes the final user response from routed node outputs.
 */
async function synthesizeNode(state: typeof MessagesAnnotation.State) {
  const startedAt = startTimer();
  const query = extractLatestUserQuery(state.messages);
  const route = extractRouteDecision(state.messages);
  const modelBundle = getLlmBundle();

  const webRaw = extractNodeOutput(state.messages, "web_search");
  const ragRaw = extractNodeOutput(state.messages, "chroma_rag");
  const generalRaw = extractNodeOutput(state.messages, "general_llm");

  const prompt = [
    "You are the final response synthesizer in an agent workflow.",
    "Use only available node outputs and do not fabricate citations.",
    `Selected route: ${route.route}`,
    `Route reason: ${route.reason}`,
    `User query: ${query}`,
    "web_search_output:",
    webRaw || "<empty>",
    "chroma_rag_output:",
    ragRaw || "<empty>",
    "general_output:",
    generalRaw || "<empty>",
    "Write a concise final answer.",
    "If data is missing from routed node, explain limits briefly and still help.",
  ].join("\n\n");

  if (!modelBundle) {
    const detail = llmInitError ?? "LLM is unavailable";
    const fallback =
      route.route === "chroma_rag" && ragRaw
        ? `LLM is unavailable (${detail}). Retrieved context exists but cannot be synthesized. Please configure GROQ_API_KEY or Azure OpenAI settings and retry.`
        : route.route === "web_search" && webRaw
          ? `LLM is unavailable (${detail}). Web results were fetched but cannot be synthesized. Please configure GROQ_API_KEY or Azure OpenAI settings and retry.`
          : `LLM is unavailable (${detail}). Please set GROQ_API_KEY (or Azure OpenAI env vars) and retry.`;

    return {
      messages: [
        new AIMessage(fallback),
        buildSystemMessage(
          TRACE_TAG,
          buildTrace("synthesize_response", "error", startedAt, detail)
        ),
      ],
    };
  }

  const { llm, provider, model } = modelBundle;

  try {
    const modelResponse = await llm.invoke(prompt);
    const responseText = String(modelResponse.content);

    return {
      messages: [
        buildSystemMessage(
          TRACE_TAG,
          buildTrace("synthesize_response", "ok", startedAt, `provider=${provider},model=${model}`)
        ),
        new AIMessage(responseText),
      ],
    };
  } catch (error) {
    const detail = error instanceof Error ? error.message : "Unknown synth error";
    return {
      messages: [
        buildSystemMessage(TRACE_TAG, buildTrace("synthesize_response", "error", startedAt, detail)),
        new AIMessage("I could not synthesize a final response right now. Please try again."),
      ],
    };
  }
}

const graph = new StateGraph(MessagesAnnotation)
  .addNode("classify_query", classifyNode)
  .addNode("web_search", webSearchNode)
  .addNode("chroma_rag", chromaRagNode)
  .addNode("general_llm", generalContextNode)
  .addNode("synthesize_response", synthesizeNode)
  .addEdge(START, "classify_query")
  .addConditionalEdges("classify_query", routeAfterClassification, {
    web_search: "web_search",
    chroma_rag: "chroma_rag",
    general_llm: "general_llm",
  })
  .addEdge("web_search", "synthesize_response")
  .addEdge("chroma_rag", "synthesize_response")
  .addEdge("general_llm", "synthesize_response")
  .addEdge("synthesize_response", END)
  .compile();

/**
 * Parses and normalizes web search sources for UI rendering.
 */
function parseWebSources(raw: string): SourceItem[] {
  if (!raw) {
    return [];
  }

  try {
    const payload = JSON.parse(raw) as {
      results?: Array<{ title?: string; url?: string; snippet?: string }>;
    };

    return (payload.results ?? []).map((item) => ({
      title: item.title ?? "Web result",
      url: item.url,
      snippet: item.snippet ?? "",
    }));
  } catch {
    return [];
  }
}

/**
 * Parses and normalizes Chroma source snippets for UI rendering.
 */
function parseRagSources(raw: string): SourceItem[] {
  if (!raw) {
    return [];
  }

  try {
    const payload = JSON.parse(raw) as ChromaMatch[];
    if (!Array.isArray(payload)) {
      return [];
    }

    return payload.map((item) => ({
      title: item.source,
      snippet: `${item.content.slice(0, 220)}${item.content.length > 220 ? "..." : ""}`,
    }));
  } catch {
    return [];
  }
}

/**
 * Reads in-memory message history for a thread.
 */
function getThreadHistory(threadId: string): BaseMessage[] {
  return ensureThread(threadId);
}

/**
 * Returns human/AI transcript messages for a thread.
 */
export function getThreadTranscript(threadId: string): Array<{ type: string; content: string }> {
  return getThreadHistory(threadId)
    .filter((message) => message.getType() === "human" || message.getType() === "ai")
    .map((message) => ({
      type: message.getType(),
      content: String(message.content),
    }));
}

/**
 * Clears thread messages, workflow traces, and source cache.
 */
export function clearThread(threadId: string): void {
  threadStore[threadId] = [];
  workflowTraceStore[threadId] = [];
  sourceStore[threadId] = [];
}

/**
 * Returns the latest workflow trace for a thread.
 */
export function getThreadWorkflow(threadId: string): WorkflowStep[] {
  return workflowTraceStore[threadId] ?? [];
}

/**
 * Returns the latest source list for a thread.
 */
export function getThreadSources(threadId: string): SourceItem[] {
  return sourceStore[threadId] ?? [];
}

/**
 * Runs one AgenticRAG turn and updates in-memory thread state.
 */
export async function runAgenticRagChat(threadId: string, message: string): Promise<ChatResult> {
  const traceId = `trace_${Date.now()}_${Math.floor(Math.random() * 10000)}`;
  const history = ensureThread(threadId);

  const nextMessages = [...history, new HumanMessage(message)];
  const result = await graph.invoke({ messages: nextMessages });

  const finalMessage = result.messages[result.messages.length - 1];
  const response = String(finalMessage.content);
  const route = extractRouteDecision(result.messages).route;
  const workflow = extractTrace(result.messages).slice(-8);

  const webSources = parseWebSources(extractNodeOutput(result.messages, "web_search"));
  const ragSources = parseRagSources(extractNodeOutput(result.messages, "chroma_rag"));
  const sources = route === "web_search" ? webSources : route === "chroma_rag" ? ragSources : [];

  threadStore[threadId] = [...nextMessages, new AIMessage(response)];
  workflowTraceStore[threadId] = workflow;
  sourceStore[threadId] = sources;

  return {
    response,
    route,
    workflow,
    sources,
    traceId,
  };
}
