import { ChatGroq } from "@langchain/groq";
import { tool } from "@langchain/core/tools";
import { ToolMessage, HumanMessage, AIMessage, BaseMessage } from "@langchain/core/messages";
import { StateGraph, MessagesAnnotation, END, START } from "@langchain/langgraph";
import { TavilySearchAPIWrapper } from "@langchain/tavily";
import { z } from "zod";
import * as dotenv from "dotenv";
import * as readline from "readline";

dotenv.config();

// ─── Thread Management (using email as thread ID) ───────────────────────────
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
  // Add user message to thread history
  const userMessage = new HumanMessage(userInput);
  addMessageToThread(threadId, userMessage);

  // Get all messages from this thread
  const threadMessages = getThreadMessages(threadId);

  // Invoke the graph with thread history
  const result = await graph.invoke({
    messages: threadMessages,
  });

  // Extract the final response
  const lastMessage: BaseMessage = result.messages[result.messages.length - 1];
  const responseText = String(lastMessage.content);

  // Add AI response to thread history
  const aiMessage = new AIMessage(responseText);
  addMessageToThread(threadId, aiMessage);

  return responseText;
}

// ─── Utility to validate email ──────────────────────────────────────────────
function isValidEmail(email: string): boolean {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailRegex.test(email);
}

// ─── Interactive CLI ────────────────────────────────────────────────────────
async function startInteractiveChat(threadId: string): Promise<void> {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  const question = (prompt: string): Promise<string> => {
    return new Promise((resolve) => {
      rl.question(prompt, resolve);
    });
  };

  console.log("\n" + "─".repeat(70));
  console.log(`✅ Chat initialized with thread ID: ${threadId}`);
  console.log("📭 Type 'exit' to quit, 'clear' to clear history");
  console.log("─".repeat(70) + "\n");

  // eslint-disable-next-line no-constant-condition
  while (true) {
    const userInput = await question("You: ");

    if (!userInput.trim()) {
      continue;
    }

    if (userInput.toLowerCase() === "exit") {
      console.log("\n👋 Goodbye!");
      rl.close();
      break;
    }

    if (userInput.toLowerCase() === "clear") {
      threadStore[threadId] = [];
      console.log("🗑️  History cleared.\n");
      continue;
    }

    try {
      console.log("\n🤖 Agent processing...\n");
      const response = await chat(userInput, threadId);
      console.log(`Agent: ${response}\n`);
    } catch (error) {
      console.error(
        `\n❌ Error: ${error instanceof Error ? error.message : "Unknown error"}\n`
      );
    }
  }
}

// ─── Main ────────────────────────────────────────────────────────────────────
async function main(): Promise<void> {
  console.log("\n");
  console.log("╔" + "═".repeat(68) + "╗");
  console.log("║ 🤖 Agentic AI Chatbot with Web Search                            ║");
  console.log("║ ─────────────────────────────────────────────────────────────── ║");
  console.log("║ Features:                                                        ║");
  console.log("║   • Thread-based history tracking (using email as thread ID)   ║");
  console.log("║   • Web search capability via Tavily                          ║");
  console.log("║   • LangChain StateGraph with Groq LLM                        ║");
  console.log("║   • Persistent conversation history per email                 ║");
  console.log("╚" + "═".repeat(68) + "╝\n");

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  const question = (prompt: string): Promise<string> => {
    return new Promise((resolve) => {
      rl.question(prompt, resolve);
    });
  };

  // Prompt for email
  console.log("📧 Please enter your email address to initialize your thread:\n");
  let email = "";
  // eslint-disable-next-line no-constant-condition
  while (true) {
    email = await question("Email: ");
    if (isValidEmail(email.trim())) {
      email = email.trim();
      break;
    }
    console.log("❌ Invalid email. Please try again.\n");
  }

  rl.close();

  // Start interactive chat
  await startInteractiveChat(email);
}

main().catch(console.error);
