import { ChatGroq } from "@langchain/groq";
import { tool } from "@langchain/core/tools";
import { ToolMessage, HumanMessage, AIMessage, BaseMessage } from "@langchain/core/messages";
import { StateGraph, MessagesAnnotation, END, START } from "@langchain/langgraph";
import { z } from "zod";
import * as dotenv from "dotenv";

dotenv.config();

// ─── Tool 1: Convert USD to INR ──────────────────────────────────────────────
const convertUsdToInrTool = tool(
  async (input: { usd: number }) => {
    const conversionRate = 83.5; // 1 USD = 83.5 INR (approximate)
    const inr = Number((input.usd * conversionRate).toFixed(2));
    return JSON.stringify(
      {
        usd: input.usd,
        exchangeRate: conversionRate,
        inr,
      },
      null,
      2
    );
  },
  {
    name: "convert_usd_to_inr",
    description:
      "Convert a USD amount to INR using the current exchange rate. Call this first for each USD transaction before calculating the combined total.",
    schema: z.object({
      usd: z.number().describe("USD amount to convert into INR."),
    }),
  }
);

// ─── Tool 2: Add two INR amounts ─────────────────────────────────────────────
const addInrAmountsTool = tool(
  async (input: { amount1: number; amount2: number }) => {
    const total = Number((input.amount1 + input.amount2).toFixed(2));
    return JSON.stringify(
      {
        amount1: input.amount1,
        amount2: input.amount2,
        totalInr: total,
      },
      null,
      2
    );
  },
  {
    name: "add_inr_amounts",
    description:
      "Add two INR amounts together. Call this AFTER the USD transactions have been converted to INR to compute the combined total.",
    schema: z.object({
      amount1: z.number().describe("First INR amount."),
      amount2: z.number().describe("Second INR amount."),
    }),
  }
);

const tools = [convertUsdToInrTool, addInrAmountsTool];
// Build a lookup map for fast tool dispatch
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const toolMap: Record<string, any> = Object.fromEntries(tools.map((t) => [t.name, t]));

// ─── Groq LLM with tools bound ───────────────────────────────────────────────
const llm = new ChatGroq({
  model: "llama-3.3-70b-versatile",
  temperature: 0,
  apiKey: process.env.GROQ_API_KEY,
}).bindTools(tools);

// ─── Graph nodes ─────────────────────────────────────────────────────────────

/**
 * Node 1 — call the LLM and let it decide which tool to call next.
 * The LLM sees tool outputs in the message history and can make decisions based on them.
 */
async function callModel(state: typeof MessagesAnnotation.State) {
  const response = await llm.invoke(state.messages);
  return { messages: [response] };
}

/**
 * Node 2 — execute tool calls sequentially.
 * Each tool call output is added to the message history, so the LLM can see it for the next call.
 */
async function callTools(state: typeof MessagesAnnotation.State) {
  const lastMessage = state.messages[state.messages.length - 1] as AIMessage;
  const toolCalls = lastMessage.tool_calls ?? [];

  const results: ToolMessage[] = [];

  // Process each tool call sequentially
  for (const tc of toolCalls) {
    const selectedTool = toolMap[tc.name];
    if (!selectedTool) {
      results.push(
        new ToolMessage({
          tool_call_id: tc.id ?? "",
          content: `Tool "${tc.name}" not found.`,
        })
      );
      continue;
    }

    console.log(`  → Executing: ${tc.name} with args:`, JSON.stringify(tc.args, null, 2));
    const output = await selectedTool.invoke(tc.args);
    console.log(`  ← Tool result:`, output);

    results.push(
      new ToolMessage({
        tool_call_id: tc.id ?? "",
        content: String(output),
      })
    );
  }

  return { messages: results };
}

/** Conditional edge — route to tools if there are pending tool calls, else end */
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
  .addEdge("tools", "agent") // After tools execute, loop back to agent
  .compile();

// ─── Runner ──────────────────────────────────────────────────────────────────
async function runAgent(userPrompt: string) {
  console.log("\n" + "=".repeat(80));
  console.log(`📝 User Request: ${userPrompt}`);
  console.log("=".repeat(80));

  const result = await graph.invoke({
    messages: [new HumanMessage(userPrompt)],
  });

  const lastMessage: BaseMessage = result.messages[result.messages.length - 1];
  console.log("\n" + "─".repeat(80));
  console.log(`✅ Final Answer:\n${lastMessage.content}`);
  console.log("─".repeat(80));
}

// ─── Main ────────────────────────────────────────────────────────────────────
async function main() {
  console.log("\n");
  console.log("╔" + "═".repeat(78) + "╗");
  console.log("║ Sequential Tool-Calling LangChain Agent Demo                              ║");
  console.log("║ ──────────────────────────────────────────────────────────────────────── ║");
  console.log("║ This agent demonstrates SEQUENTIAL tool calling:                         ║");
  console.log("║   • Tool 1: convert_usd_to_inr                                           ║");
  console.log("║   • Tool 2: add_inr_amounts (uses output from Tool 1)                    ║");
  console.log("╚" + "═".repeat(78) + "╝\n");

  // Example 1: Convert two USD amounts and add them
  await runAgent(
    "I have two USD transactions: $25 and $40. Please convert both to INR and tell me the total sum in INR."
  );

  // Example 2: Another scenario
  await runAgent(
    "Convert $100 and $150 from USD to INR, then calculate the total amount in INR."
  );
}

main().catch(console.error);
