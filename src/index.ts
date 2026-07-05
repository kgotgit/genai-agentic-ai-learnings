import * as dotenv from "dotenv";
import { createChatModel } from "./llm";
import { setupRAGPipeline } from "./rag-pipeline";

dotenv.config();

async function testGroqConnection() {
  try {
    const { llm, provider, model } = createChatModel({ temperature: 0 });

    console.log(`🚀 Initializing ${provider} with ${model} model...`);

    console.log("✅ LLM connection established!");
    console.log("\n📝 Testing sample prompt...\n");

    const prompt = "What is TypeScript and why is it useful?";
    console.log(`📌 Prompt: ${prompt}`);
    console.log("---");
    const response = await llm.invoke(prompt);
    const text =
      typeof response.content === "string"
        ? response.content
        : response.content
            .map((item) =>
              "text" in item && typeof item.text === "string" ? item.text : ""
            )
            .join("");

    console.log(`Response: ${text}\n`);
    console.log("✨ Test completed successfully!");
  } catch (error) {
    console.error(
      "❌ Error:",
      error instanceof Error ? error.message : String(error)
    );
    console.error(
      "\n⚠️  Check LLM_PROVIDER and its required environment variables."
    );
    process.exit(1);
  }
}

async function main() {
  if (process.argv.includes("--test-groq")) {
    await testGroqConnection();
    return;
  }

  await setupRAGPipeline();
}

void main();
