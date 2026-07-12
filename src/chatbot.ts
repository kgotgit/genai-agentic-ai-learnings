import "dotenv/config";
import express, { Request, Response } from "express";
import { HumanMessage, AIMessage, SystemMessage } from "@langchain/core/messages";
import { InMemoryChatMessageHistory } from "@langchain/core/chat_history";
import { createChatModel } from "./llm";

const app = express();
app.use(express.json());

// Initialize LLM
const { llm, model, provider } = createChatModel({ temperature: 0.7 });

// ── LangChain ConversationBufferWindowMemory (LCEL-style) ──────────────────
// InMemoryChatMessageHistory stores all messages; before each LLM call we
// slice the last `windowSize` human+AI pairs, mimicking the classic
// ConversationBufferWindowMemory behaviour without token counting.

const WINDOW_SIZE = parseInt(process.env.MEMORY_WINDOW_SIZE ?? "5", 10);
const SYSTEM_PROMPT = "You are a helpful assistant.";

// LangChain's official in-memory message store
const chatHistory = new InMemoryChatMessageHistory();

async function chat(userInput: string): Promise<string> {
  // 1. Persist the incoming human message
  await chatHistory.addMessage(new HumanMessage(userInput));

  // 2. Retrieve full history and apply the window (last windowSize pairs = 2N messages)
  const allMessages = await chatHistory.getMessages();
  const windowed = allMessages.slice(-(WINDOW_SIZE * 2));

  // 3. Build context: system prompt + windowed history
  const context = [new SystemMessage(SYSTEM_PROMPT), ...windowed];

  // 4. Invoke LLM
  const output = await llm.invoke(context);
  const reply = typeof output.content === "string"
    ? output.content
    : JSON.stringify(output.content);

  // 5. Persist AI response back to history
  await chatHistory.addMessage(new AIMessage(reply));

  return reply;
}

async function getMemorySize(): Promise<number> {
  const msgs = await chatHistory.getMessages();
  return Math.floor(msgs.length / 2); // each exchange = 1 human + 1 AI
}

async function clearMemory(): Promise<void> {
  await chatHistory.clear();
}

// Serve the chat UI
app.get("/", (_req: Request, res: Response) => {
  res.send(`<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>AI Chatbot</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: #f0f2f6;
      height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
    }

    header {
      width: 100%;
      background: #ff4b4b;
      color: white;
      padding: 14px 24px;
      font-size: 1.3rem;
      font-weight: 700;
      display: flex;
      align-items: center;
      gap: 10px;
    }

    header span.badge {
      background: rgba(255,255,255,0.25);
      font-size: 0.7rem;
      padding: 2px 8px;
      border-radius: 12px;
      font-weight: 400;
    }

    #memory-bar {
      width: 100%;
      max-width: 760px;
      padding: 6px 16px;
      font-size: 0.75rem;
      color: #888;
      display: flex;
      gap: 6px;
      align-items: center;
    }

    .memory-pip {
      width: 10px; height: 10px;
      border-radius: 50%;
      background: #e0e0e0;
      transition: background 0.3s;
    }
    .memory-pip.filled { background: #ff4b4b; }

    #chat-container {
      width: 100%;
      max-width: 760px;
      flex: 1;
      overflow-y: auto;
      padding: 24px 16px 8px;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }

    .message {
      display: flex;
      flex-direction: column;
      max-width: 80%;
    }

    .message.user { align-self: flex-end; align-items: flex-end; }
    .message.assistant { align-self: flex-start; align-items: flex-start; }

    .bubble {
      padding: 12px 16px;
      border-radius: 18px;
      line-height: 1.55;
      font-size: 0.95rem;
      white-space: pre-wrap;
    }

    .message.user .bubble {
      background: #ff4b4b;
      color: white;
      border-bottom-right-radius: 4px;
    }

    .message.assistant .bubble {
      background: white;
      color: #1a1a2e;
      border-bottom-left-radius: 4px;
      box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    }

    .label {
      font-size: 0.72rem;
      color: #888;
      margin-bottom: 4px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }

    .typing-indicator .bubble {
      display: flex;
      gap: 5px;
      align-items: center;
      padding: 14px 18px;
    }

    .dot {
      width: 8px; height: 8px;
      background: #ccc;
      border-radius: 50%;
      animation: bounce 1.2s infinite;
    }
    .dot:nth-child(2) { animation-delay: 0.2s; }
    .dot:nth-child(3) { animation-delay: 0.4s; }

    @keyframes bounce {
      0%, 60%, 100% { transform: translateY(0); }
      30% { transform: translateY(-6px); }
    }

    #input-area {
      width: 100%;
      max-width: 760px;
      padding: 12px 16px 20px;
      display: flex;
      gap: 10px;
    }

    #user-input {
      flex: 1;
      padding: 12px 16px;
      border-radius: 24px;
      border: 2px solid #e0e0e0;
      font-size: 0.95rem;
      outline: none;
      transition: border-color 0.2s;
      resize: none;
      min-height: 48px;
      max-height: 140px;
      overflow-y: auto;
      line-height: 1.4;
    }

    #user-input:focus { border-color: #ff4b4b; }

    #send-btn {
      background: #ff4b4b;
      color: white;
      border: none;
      border-radius: 50%;
      width: 48px;
      height: 48px;
      font-size: 1.2rem;
      cursor: pointer;
      flex-shrink: 0;
      transition: background 0.2s, transform 0.1s;
      display: flex;
      align-items: center;
      justify-content: center;
    }

    #send-btn:hover { background: #e03e3e; }
    #send-btn:active { transform: scale(0.95); }
    #send-btn:disabled { background: #ccc; cursor: not-allowed; }

    #clear-btn {
      background: transparent;
      color: #888;
      border: 2px solid #e0e0e0;
      border-radius: 24px;
      padding: 0 16px;
      height: 48px;
      font-size: 0.85rem;
      cursor: pointer;
      transition: all 0.2s;
      flex-shrink: 0;
    }

    #clear-btn:hover { border-color: #ff4b4b; color: #ff4b4b; }
  </style>
</head>
<body>
  <header>
    🤖 AI Chatbot
    <span class="badge">${provider} · ${model}</span>
    <span class="badge">window: ${WINDOW_SIZE} exchanges</span>
  </header>

  <div id="memory-bar"><span style="color:#aaa;font-size:0.72rem;text-transform:uppercase;letter-spacing:.05em;margin-right:4px">Memory</span></div>

  <div id="chat-container"></div>

  <div id="input-area">
    <textarea
      id="user-input"
      placeholder="Ask me anything… (Enter to send, Shift+Enter for newline)"
      rows="1"
    ></textarea>
    <button id="clear-btn" title="Clear conversation">Clear</button>
    <button id="send-btn" title="Send">&#9658;</button>
  </div>

  <script>
    const chatContainer = document.getElementById('chat-container');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const clearBtn = document.getElementById('clear-btn');
    const memoryBar = document.getElementById('memory-bar');
    const WINDOW_SIZE = ${WINDOW_SIZE};

    function updateMemoryBar(filled) {
      // preserve the label (first child)
      const label = memoryBar.firstChild;
      memoryBar.innerHTML = '';
      memoryBar.appendChild(label);
      for (let i = 0; i < WINDOW_SIZE; i++) {
        const pip = document.createElement('div');
        pip.className = 'memory-pip' + (i < filled ? ' filled' : '');
        memoryBar.appendChild(pip);
      }
      const txt = document.createElement('span');
      txt.style.cssText = 'margin-left:6px;color:#aaa';
      txt.textContent = filled + '/' + WINDOW_SIZE + ' exchanges';
      memoryBar.appendChild(txt);
    }
    updateMemoryBar(0);

    function appendMessage(role, text) {
      const wrapper = document.createElement('div');
      wrapper.className = 'message ' + role;

      const label = document.createElement('div');
      label.className = 'label';
      label.textContent = role === 'user' ? 'You' : 'Assistant';

      const bubble = document.createElement('div');
      bubble.className = 'bubble';
      bubble.textContent = text;

      wrapper.appendChild(label);
      wrapper.appendChild(bubble);
      chatContainer.appendChild(wrapper);
      chatContainer.scrollTop = chatContainer.scrollHeight;
      return wrapper;
    }

    function showTyping() {
      const wrapper = document.createElement('div');
      wrapper.className = 'message assistant typing-indicator';
      wrapper.innerHTML = '<div class="label">Assistant</div><div class="bubble"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>';
      chatContainer.appendChild(wrapper);
      chatContainer.scrollTop = chatContainer.scrollHeight;
      return wrapper;
    }

    async function sendMessage() {
      const text = userInput.value.trim();
      if (!text) return;

      userInput.value = '';
      userInput.style.height = 'auto';
      sendBtn.disabled = true;

      appendMessage('user', text);
      const typing = showTyping();

      try {
        const res = await fetch('/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: text }),
        });
        const data = await res.json();
        chatContainer.removeChild(typing);

        if (data.error) {
          appendMessage('assistant', '⚠️ Error: ' + data.error);
        } else {
          appendMessage('assistant', data.reply);
          updateMemoryBar(data.memorySize);
        }
      } catch (err) {
        chatContainer.removeChild(typing);
        appendMessage('assistant', '⚠️ Network error. Please try again.');
      }

      sendBtn.disabled = false;
      userInput.focus();
    }

    sendBtn.addEventListener('click', sendMessage);

    userInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    });

    // Auto-resize textarea
    userInput.addEventListener('input', () => {
      userInput.style.height = 'auto';
      userInput.style.height = Math.min(userInput.scrollHeight, 140) + 'px';
    });

    clearBtn.addEventListener('click', async () => {
      await fetch('/clear', { method: 'POST' });
      chatContainer.innerHTML = '';
      updateMemoryBar(0);
    });
  </script>
</body>
</html>`);
});

// Chat API endpoint
app.post("/chat", async (req: Request, res: Response) => {
  const { message } = req.body as { message: string };

  if (!message || message.trim().length === 0) {
    res.status(400).json({ error: "Message cannot be empty." });
    return;
  }

  try {
    const reply = await chat(message);
    const memorySize = await getMemorySize();
    res.json({ reply, memorySize, windowSize: WINDOW_SIZE });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : "Unknown error occurred";
    res.status(500).json({ error: message });
  }
});

// Clear conversation history
app.post("/clear", async (_req: Request, res: Response) => {
  await clearMemory();
  res.json({ status: "cleared" });
});

// Memory stats
app.get("/memory", async (_req: Request, res: Response) => {
  const exchanges = await getMemorySize();
  res.json({ exchanges, windowSize: WINDOW_SIZE });
});

const PORT = process.env.PORT ?? 3000;
app.listen(PORT, () => {
  console.log(`\n🤖 Chatbot running at http://localhost:${PORT}\n`);
});
