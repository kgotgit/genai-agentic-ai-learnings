import express, { Request, Response } from "express";
import { ChatGroq } from "@langchain/groq";
import { tool } from "@langchain/core/tools";
import { ToolMessage, HumanMessage, AIMessage, BaseMessage } from "@langchain/core/messages";
import { StateGraph, MessagesAnnotation, END, START } from "@langchain/langgraph";
import { TavilySearchAPIWrapper } from "@langchain/tavily";
import { z } from "zod";
import * as dotenv from "dotenv";

dotenv.config();

const app = express();
app.use(express.json());
app.use(express.static("public"));

// ─── Thread Management ───────────────────────────────────────────────────────
interface ThreadStore {
  [threadId: string]: BaseMessage[];
}

const threadStore: ThreadStore = {};

function getThreadMessages(threadId: string): BaseMessage[] {
  if (!threadStore[threadId]) {
    threadStore[threadId] = [];
  }
  return threadStore[threadId];
}

function addMessageToThread(threadId: string, message: BaseMessage): void {
  if (!threadStore[threadId]) {
    threadStore[threadId] = [];
  }
  threadStore[threadId].push(message);
}

// ─── Web Search Tool ────────────────────────────────────────────────────────
const tavilySearchApi = new TavilySearchAPIWrapper({
  tavilyApiKey: process.env.TAVILY_API_KEY,
});

const webSearchTool = tool(
  async (input: { query: string }) => {
    try {
      const response = await tavilySearchApi.rawResults({
        query: input.query,
        max_results: 3,
        search_depth: "basic",
        topic: "general",
        include_answer: true,
      });

      return JSON.stringify(
        {
          query: input.query,
          answer: response.answer ?? "No answer available",
          results: (response.results ?? []).map((result) => ({
            title: result.title,
            url: result.url,
            content: result.content,
          })),
        },
        null,
        2
      );
    } catch (error) {
      return JSON.stringify({
        query: input.query,
        error: error instanceof Error ? error.message : "Unknown error",
        results: [],
      });
    }
  },
  {
    name: "web_search",
    description:
      "Search the web for current information and answer user questions. Use this when the user asks about recent events, current information, or anything that requires up-to-date data.",
    schema: z.object({
      query: z.string().describe("The search query to use for web search."),
    }),
  }
);

const tools = [webSearchTool];
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const toolMap: Record<string, any> = Object.fromEntries(tools.map((t) => [t.name, t]));

// ─── Groq LLM with tools bound ───────────────────────────────────────────────
const llm = new ChatGroq({
  model: "llama-3.3-70b-versatile",
  temperature: 0.7,
  apiKey: process.env.GROQ_API_KEY,
}).bindTools(tools);

// ─── Graph nodes ─────────────────────────────────────────────────────────────
async function callModel(state: typeof MessagesAnnotation.State) {
  const response = await llm.invoke(state.messages);
  return { messages: [response] };
}

async function callTools(state: typeof MessagesAnnotation.State) {
  const lastMessage = state.messages[state.messages.length - 1] as AIMessage;
  const toolCalls = lastMessage.tool_calls ?? [];

  const results: ToolMessage[] = await Promise.all(
    toolCalls.map(async (tc) => {
      const selectedTool = toolMap[tc.name];
      if (!selectedTool) {
        return new ToolMessage({
          tool_call_id: tc.id ?? "",
          content: `Tool "${tc.name}" not found.`,
        });
      }
      const output = await selectedTool.invoke(tc.args);
      return new ToolMessage({
        tool_call_id: tc.id ?? "",
        content: String(output),
      });
    })
  );

  return { messages: results };
}

function shouldContinue(state: typeof MessagesAnnotation.State): "tools" | typeof END {
  const lastMessage = state.messages[state.messages.length - 1] as AIMessage;
  if (lastMessage.tool_calls && lastMessage.tool_calls.length > 0) {
    return "tools";
  }
  return END;
}

// ─── Build the StateGraph ────────────────────────────────────────────────────
const graph = new StateGraph(MessagesAnnotation)
  .addNode("agent", callModel)
  .addNode("tools", callTools)
  .addEdge(START, "agent")
  .addConditionalEdges("agent", shouldContinue, { tools: "tools", [END]: END })
  .addEdge("tools", "agent")
  .compile();

// ─── Chat Function ──────────────────────────────────────────────────────────
async function chat(userInput: string, threadId: string): Promise<string> {
  const userMessage = new HumanMessage(userInput);
  addMessageToThread(threadId, userMessage);

  const threadMessages = getThreadMessages(threadId);

  const result = await graph.invoke({
    messages: threadMessages,
  });

  const lastMessage: BaseMessage = result.messages[result.messages.length - 1];
  const responseText = String(lastMessage.content);

  const aiMessage = new AIMessage(responseText);
  addMessageToThread(threadId, aiMessage);

  return responseText;
}

// ─── Email Validation ───────────────────────────────────────────────────────
function isValidEmail(email: string): boolean {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailRegex.test(email);
}

// ─── API Endpoints ──────────────────────────────────────────────────────────

// Validate email and initialize thread
app.post("/api/init", (req: Request, res: Response) => {
  const { email } = req.body;

  if (!email || !isValidEmail(email.trim())) {
    return res.status(400).json({ error: "Invalid email address" });
  }

  const threadId = email.trim();
  res.json({ threadId, message: "Thread initialized successfully" });
});

// Send a message to the agent
app.post("/api/chat", async (req: Request, res: Response) => {
  try {
    const { threadId, message } = req.body as { threadId?: string; message?: string };

    if (!threadId || !message) {
      return res.status(400).json({ error: "Missing threadId or message" });
    }

    const response = await chat(message, threadId);
    res.json({ response });
  } catch (error) {
    console.error("Chat error:", error);
    res.status(500).json({ error: error instanceof Error ? error.message : "Unknown error" });
  }
});

// Get conversation history for a thread
app.get("/api/history/:threadId", (req: Request, res: Response) => {
  const threadId = Array.isArray(req.params.threadId) ? req.params.threadId[0] : req.params.threadId;
  const messages = getThreadMessages(threadId);

  const formattedMessages = messages.map((msg) => ({
    type: msg.getType(),
    content: msg.content,
  }));

  res.json({ history: formattedMessages });
});

// Clear history for a thread
app.delete("/api/history/:threadId", (req: Request, res: Response) => {
  const threadId = Array.isArray(req.params.threadId) ? req.params.threadId[0] : req.params.threadId;
  threadStore[threadId] = [];
  res.json({ message: "History cleared" });
});

// Health check
app.get("/api/health", (_req: Request, res: Response) => {
  res.json({ status: "ok" });
});

// ─── Start Server ───────────────────────────────────────────────────────────
const PORT = parseInt(process.env.PORT ?? "3000", 10);

app.listen(PORT, () => {
  console.log(`\n${"═".repeat(70)}`);
  console.log(`🤖 Agentic AI Chatbot Server Running`);
  console.log(`${"═".repeat(70)}`);
  console.log(`🌐 Open browser: http://localhost:${PORT}`);
  console.log(`📡 API Server: http://localhost:${PORT}/api`);
  console.log(`${"═".repeat(70)}\n`);
});
