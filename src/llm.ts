import { ChatGroq } from "@langchain/groq";
import { ChatOpenAI } from "@langchain/openai";

export type SupportedLlmProvider = "groq" | "azure";
export type AppChatModel = ChatGroq | ChatOpenAI;

type CreateChatModelOptions = {
  temperature?: number;
};

type ResolvedChatModel = {
  llm: AppChatModel;
  provider: SupportedLlmProvider;
  model: string;
};

function parseProvider(value: string | undefined): SupportedLlmProvider {
  const normalized = value?.toLowerCase().trim();

  if (normalized === "azure") {
    return "azure";
  }

  return "groq";
}

function requireEnv(name: string, value: string | undefined): string {
  if (!value || value.trim().length === 0) {
    throw new Error(`Missing ${name} environment variable.`);
  }

  return value.trim();
}

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

function createGroqModel(temperature: number): ResolvedChatModel {
  const apiKey = requireEnv("GROQ_API_KEY", process.env.GROQ_API_KEY);
  const model = process.env.GROQ_MODEL ?? "llama-3.3-70b-versatile";

  return {
    llm: new ChatGroq({
      apiKey,
      model,
      temperature,
    }),
    provider: "groq",
    model,
  };
}

function createAzureModel(temperature: number): ResolvedChatModel {
  const apiKey = requireEnv(
    "AZURE_API_KEY or AZURE_OPENAI_API_KEY",
    process.env.AZURE_API_KEY ?? process.env.AZURE_OPENAI_API_KEY
  );
  const deployment = requireEnv(
    "AZURE_OPENAI_DEPLOYMENT",
    process.env.AZURE_OPENAI_DEPLOYMENT
  );
  const baseUrlRaw = requireEnv(
    "AZURE_OPENAI_BASE_URL or AZURE_OPENAI_ENDPOINT",
    process.env.AZURE_OPENAI_BASE_URL ?? process.env.AZURE_OPENAI_ENDPOINT
  );

  const baseURL = normalizeAzureBaseUrl(baseUrlRaw);

  return {
    llm: new ChatOpenAI({
      apiKey,
      model: deployment,
      temperature,
      configuration: {
        baseURL,
      },
    }),
    provider: "azure",
    model: deployment,
  };
}

export function createChatModel(
  options: CreateChatModelOptions = {}
): ResolvedChatModel {
  const temperature = options.temperature ?? 0;
  const provider = parseProvider(process.env.LLM_PROVIDER);

  if (provider === "azure") {
    return createAzureModel(temperature);
  }

  return createGroqModel(temperature);
}
