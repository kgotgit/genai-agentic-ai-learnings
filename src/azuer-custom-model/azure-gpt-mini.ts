import * as dotenv from "dotenv";
import { ChatOpenAI } from "@langchain/openai";

dotenv.config();

const endpoint =
  process.env.AZURE_OPENAI_BASE_URL ??
  process.env.AZURE_OPENAI_ENDPOINT ??
  "https://open-ai-learn-07-05.openai.azure.com/openai/v1/chat/completions";
const deploymentName = process.env.AZURE_OPENAI_DEPLOYMENT ?? "gpt-5.4-mini";
const apiKey = process.env.AZURE_API_KEY ?? process.env.AZURE_OPENAI_API_KEY;

function normalizeAzureBaseUrl(baseUrlRaw: string): string {
  const trimmed = baseUrlRaw.trim().replace(/\/+$/, "");
  const withoutChatCompletions = trimmed.replace(/\/chat\/completions$/i, "");

  if (/\/openai\/v1$/i.test(withoutChatCompletions)) {
    return withoutChatCompletions;
  }

  if (/\/openai$/i.test(withoutChatCompletions)) {
    return `${withoutChatCompletions}/v1`;
  }

  return `${withoutChatCompletions}/openai/v1`;
}

async function main() {
  if (!apiKey) {
    throw new Error(
      "Missing Azure API key. Set AZURE_API_KEY or AZURE_OPENAI_API_KEY in .env"
    );
  }

  const llm = new ChatOpenAI({
    apiKey,
    model: deploymentName,
    temperature: 0,
    configuration: {
      baseURL: normalizeAzureBaseUrl(endpoint),
    },
  });

  const completion = await llm.invoke([
    { role: "system", content: "You talk like a pirate." },
    { role: "user", content: "Can you help me?" },
  ]);

  console.log(completion.content);
}

void main();