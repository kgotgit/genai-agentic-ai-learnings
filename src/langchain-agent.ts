import { ChatGroq } from "@langchain/groq";
import { tool } from "@langchain/core/tools";
import { ToolMessage, HumanMessage, AIMessage, BaseMessage } from "@langchain/core/messages";
import { StateGraph, MessagesAnnotation, END, START } from "@langchain/langgraph";
import { z } from "zod";
import * as dotenv from "dotenv";

dotenv.config();

// ─── Tool 1: Generate a random number between 1 and 10000 ───────────────────
const generateRandomNumberTool = tool(
  async (_input: { dummy?: string }) => {
    const randomNumber = Math.floor(Math.random() * 10000) + 1;
    return `Generated random number: ${randomNumber}`;
  },
  {
    name: "generate_random_number",
    description:
      "Generates a random number between 1 and 10000. Use this when the user asks to generate, produce, or get a random number.",
    schema: z.object({
      dummy: z.string().optional().describe("Not used — placeholder only."),
    }),
  }
);

// ─── Tool 2: Add two numbers ─────────────────────────────────────────────────
const addTwoNumbersTool = tool(
  async (input: { a: number; b: number }) => {
    const result = input.a + input.b;
    return `The sum of ${input.a} and ${input.b} is: ${result}`;
  },
  {
    name: "add_two_numbers",
    description:
      "Adds two numbers and returns the result. Use this when the user wants to add, sum, or calculate the total of two numbers.",
    schema: z.object({
      a: z.number().describe("The first number to add."),
      b: z.number().describe("The second number to add."),
    }),
  }
);

const tools = [generateRandomNumberTool, addTwoNumbersTool];
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

/** Node 1 — call the LLM and let it decide which tool (if any) to invoke */
async function callModel(state: typeof MessagesAnnotation.State) {
  const response = await llm.invoke(state.messages);
  return { messages: [response] };
}

/** Node 2 — execute every tool call the LLM requested */
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
  .addEdge("tools", "agent") // after tools, loop back to agent
  .compile();

// ─── Runner ──────────────────────────────────────────────────────────────────
async function runAgent(userPrompt: string) {
  console.log("\n" + "=".repeat(60));
  console.log(`User: ${userPrompt}`);
  console.log("=".repeat(60));

  const result = await graph.invoke({
    messages: [new HumanMessage(userPrompt)],
  });

  const lastMessage: BaseMessage = result.messages[result.messages.length - 1];
  console.log(`Agent: ${lastMessage.content}`);
}

// ─── Main ────────────────────────────────────────────────────────────────────
async function main() {
  console.log("LangChain Agent with Groq (StateGraph) — Two Tools Demo");
  console.log("Tools available:");
  console.log("  1. generate_random_number — generates a random number 1–10000");
  console.log("  2. add_two_numbers        — adds two numbers together\n");

  // Triggers generate_random_number
  await runAgent("Can you generate a random number for me?");

  // Triggers add_two_numbers
  await runAgent("What is 342 + 758?");

  // Triggers generate_random_number
  await runAgent("Give me a random number between 1 and 10000.");

  // Triggers add_two_numbers
  await runAgent("Please add 1500 and 2500 together.");
}

main().catch(console.error);
