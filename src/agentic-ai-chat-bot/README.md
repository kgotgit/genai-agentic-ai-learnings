# Agentic AI Chatbot

## Overview

This is an **Agentic AI Chatbot** that combines intelligent conversation with web search capabilities. The chatbot uses:

- **LangChain StateGraph** for agentic decision-making
- **Groq LLM (llama-3.3-70b-versatile)** for natural language understanding
- **Tavily Search API** for real-time web search
- **Thread-based History** using email addresses to maintain persistent conversation context

## Features

✨ **Thread-Based Conversation History**
- Users enter their email address at startup
- Email is used as a unique thread ID
- All messages are stored per thread, allowing multi-session conversations

🌐 **Web Search Tool**
- Agent can decide to search the web for current information
- Uses Tavily API to fetch up-to-date results
- Automatically provides answers and relevant URLs

🤖 **Intelligent Agent Decision-Making**
- Groq LLM decides when to search the web vs. responding from knowledge
- StateGraph ensures proper tool-calling workflow
- Tools execute sequentially with feedback to the agent

🔄 **Persistent Conversation Context**
- All historical messages are maintained per thread
- Agent can reference previous messages in the conversation
- Allows for coherent multi-turn conversations

## How It Works

### Execution Flow

```
User enters email → Thread ID created/loaded
↓
User types message → Added to thread history
↓
Agent LLM reads all thread history
↓
Agent decides: "Search web?" → Yes/No
↓
If YES → Web Search Tool → Returns results
↓
If NO → Direct response from LLM
↓
Response added to thread history
↓
User gets answer with context
```

### Thread Management

```
Thread Store (In-Memory):
{
  "user@example.com": [
    HumanMessage: "What is the capital of France?",
    AIMessage: "Paris is the capital of France...",
    HumanMessage: "Tell me about its history",
    AIMessage: "Let me search for recent information..."
  ],
  "another@example.com": [
    HumanMessage: "Latest news on AI",
    ...
  ]
}
```

## Installation & Setup

### Prerequisites

- Node.js 16+
- npm or yarn
- Environment variables configured:
  - `GROQ_API_KEY` — Your Groq API key
  - `TAVILY_API_KEY` — Your Tavily Search API key

### Configuration

1. **Set up environment variables** in `.env`:
   ```bash
   GROQ_API_KEY=your_groq_key_here
   TAVILY_API_KEY=your_tavily_key_here
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

## Running the Chatbot

### Start the Interactive Chatbot

```bash
npm run agent:chatbot
```

### Usage Steps

1. **Enter Email**: When prompted, enter a valid email address
   ```
   📧 Please enter your email address to initialize your thread:
   Email: john@example.com
   ```

2. **Ask Questions**: Type your questions or statements
   ```
   You: What are the latest updates in TypeScript?
   ```

3. **Get Responses**: The agent will respond, using web search if needed
   ```
   Agent: Based on recent information, TypeScript 5.9...
   ```

4. **Navigate History**: The agent remembers all previous messages in this session

5. **Special Commands**:
   - `clear` — Clear conversation history for this thread
   - `exit` — Exit the chatbot

### Example Conversation

```
╔════════════════════════════════════════════════════════════════════╗
║ 🤖 Agentic AI Chatbot with Web Search                            ║
║ Features:                                                        ║
║   • Thread-based history tracking (using email as thread ID)   ║
║   • Web search capability via Tavily                          ║
║   • LangChain StateGraph with Groq LLM                        ║
║   • Persistent conversation history per email                 ║
╚════════════════════════════════════════════════════════════════════╝

📧 Please enter your email address to initialize your thread:

Email: developer@company.com

──────────────────────────────────────────────────────────────────────
✅ Chat initialized with thread ID: developer@company.com
📭 Type 'exit' to quit, 'clear' to clear history
──────────────────────────────────────────────────────────────────────

You: What is the latest news about AI?

🤖 Agent processing...

Agent: Let me search for the latest news on AI developments.

Based on recent reports, AI is advancing rapidly in several areas:
- Large Language Models becoming more efficient
- Multimodal AI capabilities expanding
- Enterprise adoption accelerating...

You: Can you tell me more about LLMs?

🤖 Agent processing...

Agent: Large Language Models (LLMs) are neural networks trained on vast amounts of text data. 
Recent developments include...

You: exit

👋 Goodbye!
```

## Technical Architecture

### StateGraph Structure

```typescript
START
  ↓
"agent" node (LLM Decision)
  ↓
shouldContinue() check
  ├─ YES (tool_calls) → "tools" node → execute tools
  │                         ↓
  │                    add tool results
  │                         ↓
  │                    loop back to "agent"
  │
  └─ NO (no tool_calls) → END

Graph completes → Final answer extracted
```

### Components

#### 1. **Thread Store**
- In-memory storage of conversations per email
- Each thread maintains full message history
- Messages persist during the session

#### 2. **Web Search Tool**
- Takes a `query` parameter
- Returns JSON with:
  - `query` — The search query
  - `answer` — LLM-generated answer
  - `results` — Array of search results with title, URL, content

#### 3. **Groq LLM**
- Model: `llama-3.3-70b-versatile`
- Temperature: 0.7 (for natural conversation)
- Tools bound: web_search tool

#### 4. **StateGraph**
- Two nodes: `agent` and `tools`
- Conditional routing based on tool_calls
- Feedback loop for sequential tool use

## Code References

See these files for implementation details:

- **Main chatbot**: [src/agentic-ai-chat-bot/agentic-ai-chatbot.ts](../src/agentic-ai-chat-bot/agentic-ai-chatbot.ts)
- **Agent pattern reference**: [src/langchain-agent.ts](../src/langchain-agent.ts)
- **Sequential tooling reference**: [src/langchain-sequential-agent.ts](../src/langchain-sequential-agent.ts)

## Limitations

- History is stored **in-memory only** (lost when process exits)
- Only one tool (web_search) is available
- Search results limited to 3 per query

## Future Enhancements

1. **Persistent Storage**: Save threads to database (MongoDB, PostgreSQL)
2. **More Tools**: Add email, calendar, file operations
3. **User Management**: Proper authentication and user profiles
4. **Conversation Analytics**: Track topics and query patterns
5. **Multi-turn Memory**: Semantic summarization for long conversations

## Troubleshooting

### Error: "Cannot find module '@langchain/tavily'"
```bash
npm install @langchain/tavily --legacy-peer-deps
```

### Error: "Invalid email"
- Ensure email format is correct: `user@domain.com`
- No spaces allowed

### Slow Response Times
- Check internet connection for web search
- Groq API might be rate-limited; wait a moment

### No Web Search Results
- Verify `TAVILY_API_KEY` is set correctly
- Check Tavily API account has available credits

## API Documentation

### Thread Functions

```typescript
getThreadMessages(threadId: string): BaseMessage[]
// Get all messages for a thread

addMessageToThread(threadId: string, message: BaseMessage): void
// Add a new message to a thread

chat(userInput: string, threadId: string): Promise<string>
// Process a user message and return agent response
```

### Chat Flow

```typescript
async function chat(userInput: string, threadId: string): Promise<string> {
  // 1. Add user message to thread
  // 2. Get all thread messages
  // 3. Invoke graph with thread history
  // 4. Extract response
  // 5. Add AI message to thread
  // 6. Return response text
}
```

## Contact & Support

For issues or questions:
- Check the task.md file in the agentic-ai-chat-bot folder
- Review SEQUENTIAL_TOOLING.md for tool-calling patterns
- Refer to langchain-agent.ts for the base agent implementation

---

**Last Updated**: July 12, 2026
**Version**: 1.0.0
