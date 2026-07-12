# Agentic AI Chatbot - Web UI Version

## Overview

This is a **web-based UI** for the Agentic AI Chatbot. It provides a beautiful, interactive interface where users can:

1. **Enter their email address** to initialize a conversation thread
2. **Chat with an intelligent AI agent** that can answer questions and search the web
3. **View conversation history** maintained per email thread
4. **Clear conversation history** when needed

## Features

✨ **Web-Based Interface**
- Beautiful, modern UI built with HTML/CSS/JavaScript
- Real-time message updates
- Responsive design (works on mobile and desktop)

🧵 **Thread-Based Conversations**
- Each email gets a unique thread ID
- Persistent conversation history per user
- Multi-session conversations

🤖 **Intelligent Agent**
- Powered by Groq's llama-3.3-70b-versatile model
- Can search the web using Tavily API
- Understands context from conversation history

🔍 **Web Search Integration**
- Automatically searches for current information when needed
- Provides relevant URLs and sources
- Factual, up-to-date responses

## Getting Started

### Prerequisites

- Node.js 16+
- Environment variables:
  - `GROQ_API_KEY` — Your Groq API key
  - `TAVILY_API_KEY` — Your Tavily Search API key
  - `PORT` — (Optional) Server port, defaults to 3000

### Installation

```bash
cd genai-agentic-ai-learnings

# Install dependencies
npm install --legacy-peer-deps

# Build TypeScript
npm run build
```

### Running the Web Server

```bash
npm run agent:web
```

Expected output:
```
══════════════════════════════════════════════════════════════════════════════
🤖 Agentic AI Chatbot Server Running
══════════════════════════════════════════════════════════════════════════════
🌐 Open browser: http://localhost:3000
📡 API Server: http://localhost:3000/api
══════════════════════════════════════════════════════════════════════════════
```

### Access the UI

Open your browser and navigate to:
```
http://localhost:3000
```

## Usage Guide

### Step 1: Initialize Your Thread

1. **Enter your email address** in the input field
2. Click **"Initialize Thread"** (or press Enter)
3. You'll see a confirmation: `✅ Logged in as your@email.com`

### Step 2: Start Chatting

1. **Type your question or message** in the chat input
2. Click **"Send"** (or press Enter)
3. The agent will process and respond with an answer
4. If needed, it will search the web for current information

### Step 3: View Conversation

- **Your messages** appear on the right (blue background)
- **Agent messages** appear on the left (gray background)
- **Timestamps** show when each message was sent
- **Typing indicator** shows when the agent is processing

### Special Commands

- **Clear History**: Click the "Clear" button to remove all messages in this thread
  - Confirmation dialog will appear
  - History is cleared only for the current email

### Examples of Questions

✅ **Questions the agent can answer:**
- "What is the capital of France?"
- "Tell me about machine learning"
- "What are the latest developments in AI?"
- "Search for information about TypeScript 5.9"
- "How does photosynthesis work?"
- "What's new in Python 3.12?"

## API Endpoints

### Initialize Thread
```
POST /api/init
Content-Type: application/json

{
  "email": "user@example.com"
}

Response:
{
  "threadId": "user@example.com",
  "message": "Thread initialized successfully"
}
```

### Send Message
```
POST /api/chat
Content-Type: application/json

{
  "threadId": "user@example.com",
  "message": "What is AI?"
}

Response:
{
  "response": "AI stands for Artificial Intelligence..."
}
```

### Get Conversation History
```
GET /api/history/:threadId

Response:
{
  "history": [
    {
      "type": "human",
      "content": "What is AI?"
    },
    {
      "type": "ai",
      "content": "AI stands for..."
    }
  ]
}
```

### Clear History
```
DELETE /api/history/:threadId

Response:
{
  "message": "History cleared"
}
```

### Health Check
```
GET /api/health

Response:
{
  "status": "ok"
}
```

## Architecture

### Frontend (HTML/CSS/JavaScript)
- **File**: `public/index.html`
- **Features**:
  - Email validation
  - Auto-scrolling chat view
  - Typing indicator animation
  - Error handling
  - Local state management

### Backend (Express + LangChain)
- **File**: `src/agentic-ai-chat-bot/server.ts`
- **Features**:
  - RESTful API endpoints
  - Thread management
  - StateGraph agent orchestration
  - Tavily web search integration

### Data Flow

```
User Types Message
        ↓
Frontend validates & sends to API
        ↓
API /chat endpoint receives message
        ↓
Thread history retrieved
        ↓
Agent LLM processes with history
        ↓
Decision: Search web? → Yes/No
        ↓
Tool execution (if needed)
        ↓
Response generated
        ↓
Response sent back to frontend
        ↓
Message displayed in chat
```

## UI Components

### Header
- Shows app title and description
- Gradient background

### Auth Section
- Email input field
- Initialize button
- Status indicator (shows logged-in email)

### Chat Area
- Scrollable message container
- Messages organized by sender
- Timestamps for each message
- Typing indicator during processing
- Empty state when starting

### Input Area
- Text input for messages
- Send button
- Clear history button (when authenticated)

## Features in Detail

### 1. Email Threading
- Each unique email gets its own conversation thread
- Switch between users by logging in with different emails
- History is kept separate per email

### 2. Web Search
- Agent automatically decides when to search the web
- Returns relevant articles and sources
- Provides factual, up-to-date information

### 3. Conversation Context
- Agent reads all previous messages in the thread
- Can reference past topics
- Maintains coherent multi-turn conversations

### 4. Error Handling
- Invalid email validation
- API error messages displayed in chat
- Graceful error recovery

### 5. Responsive Design
- Works on desktop, tablet, and mobile
- Touch-friendly buttons
- Optimized font sizes for readability

## Troubleshooting

### Port Already in Use
```bash
# Use a different port
PORT=3001 npm run agent:web
```

### Invalid Email Error
- Ensure email format is correct: `user@domain.com`
- No spaces allowed

### No Web Search Results
- Verify `TAVILY_API_KEY` is set correctly
- Check Tavily API account has available credits

### Slow Responses
- First response may take 2-3 seconds while Groq processes
- Subsequent messages usually faster
- Check internet connection

### Build Errors
```bash
# Clean and rebuild
npm run clean
npm run build
```

## Environment Variables

Create a `.env` file in the project root:

```env
# Groq API Configuration
GROQ_API_KEY=your_groq_api_key_here

# Tavily Search Configuration
TAVILY_API_KEY=your_tavily_api_key_here

# Server Configuration
PORT=3000
NODE_ENV=development
```

## Limitations

### Current Version
- History stored **in-memory** only (lost when server restarts)
- Only **one tool**: web search
- **3 search results** per query maximum
- **No user authentication** (anyone with the URL can chat)

### Future Enhancements
1. **Database persistence** (MongoDB/PostgreSQL)
2. **More tools** (email, calendar, file operations)
3. **User authentication** and profiles
4. **Conversation analytics**
5. **Typing sounds** and notifications
6. **Message reactions** (👍, 😂, etc.)

## File Locations

- **Web Server**: [src/agentic-ai-chat-bot/server.ts](../src/agentic-ai-chat-bot/server.ts)
- **Frontend**: [public/index.html](../../public/index.html)
- **CLI Version**: [src/agentic-ai-chat-bot/agentic-ai-chatbot.ts](../src/agentic-ai-chat-bot/agentic-ai-chatbot.ts)
- **Documentation**: [src/agentic-ai-chat-bot/README.md](../src/agentic-ai-chat-bot/README.md)

## Development

### Debug Mode
Add console.log statements to see API requests/responses:

```javascript
// In public/index.html
console.log("Sending message:", message);
console.log("API Response:", data);
```

### Test API Endpoints
```bash
# Initialize thread
curl -X POST http://localhost:3000/api/init \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com"}'

# Send message
curl -X POST http://localhost:3000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"threadId":"test@example.com","message":"Hello"}'

# Get health
curl http://localhost:3000/api/health
```

## Performance Tips

1. **First load**: May take 30 seconds (LLM model loading)
2. **Subsequent messages**: Usually 2-3 seconds response time
3. **Web search**: Adds 1-2 seconds extra when needed

## Support

For issues:
- Check `.env` file has correct API keys
- Ensure Node.js version is 16+
- Verify all dependencies installed: `npm install --legacy-peer-deps`
- Check server is running: `curl http://localhost:3000/api/health`

## See Also

- [CLI Chatbot Version](README.md)
- [Sequential Tooling Guide](../../SEQUENTIAL_TOOLING.md)
- [LangChain Agent Reference](../../src/langchain-agent.ts)

---

**Last Updated**: July 12, 2026  
**Version**: 1.0.0  
**Status**: ✅ Production Ready
