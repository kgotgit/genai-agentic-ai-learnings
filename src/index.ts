import * as dotenv from "dotenv";
import { ChatGroq } from "@langchain/groq";
import { setupRAGPipeline } from "./rag-pipeline";

dotenv.config();

async function testGroqConnection() {
  try {
    const apiKey = process.env.GROQ_API_KEY;
    if (!apiKey) {
      throw new Error("Missing GROQ_API_KEY environment variable.");
    }

    const model = process.env.GROQ_MODEL ?? "llama-3.3-70b-versatile";

    console.log(`🚀 Initializing Groq with ${model} model...`);

    const llm = new ChatGroq({
      apiKey,
      model,
      temperature: 0,
    });

    console.log("✅ Groq connection established!");
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
    console.error("\n⚠️  Make sure GROQ_API_KEY is set in the environment.");
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
