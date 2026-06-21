import { Ollama } from "@langchain/ollama";

async function testOllamaConnection() {
  try {
    console.log("🚀 Initializing Ollama with qwen3.5 model...");

    const llm = new Ollama({
      baseUrl: "http://localhost:11434",
      model: "qwen3.5",
    });

    console.log("✅ Ollama connection established!");
    console.log("\n📝 Testing sample prompts...\n");

    // Test 1: Simple question
    const prompt1 = "What is TypeScript and why is it useful?";
    console.log(`📌 Prompt 1: ${prompt1}`);
    console.log("---");
    const response1 = await llm.invoke(prompt1);
    console.log(`Response: ${response1}\n`);

    // Test 2: Code generation
    const prompt2 =
      "Write a simple async function in TypeScript that fetches data from an API";
    console.log(`📌 Prompt 2: ${prompt2}`);
    console.log("---");
    const response2 = await llm.invoke(prompt2);
    console.log(`Response: ${response2}\n`);

    // Test 3: Explanation
    const prompt3 = "Explain the concept of LLMs in one paragraph";
    console.log(`📌 Prompt 3: ${prompt3}`);
    console.log("---");
    const response3 = await llm.invoke(prompt3);
    console.log(`Response: ${response3}\n`);

    console.log("✨ All tests completed successfully!");
  } catch (error) {
    console.error(
      "❌ Error:",
      error instanceof Error ? error.message : String(error)
    );
    console.error("\n⚠️  Make sure Ollama is running at http://localhost:11434");
    console.error("   Run: ollama serve");
    console.error("   Then pull model: ollama pull qwen3.5");
    process.exit(1);
  }
}

//testOllamaConnection();
//setupRAGPipeline(); // moved to src/rag-pipeline.ts
