import express, { Request, Response } from "express";
import * as dotenv from "dotenv";
import * as path from "path";
import {
  clearThread,
  getThreadMemorySnapshot,
  getThreadSources,
  getThreadTranscript,
  getThreadWorkflow,
  runAgenticRagChat,
} from "./workflow";

dotenv.config();

const app = express();
app.use(express.json());

const uiRoot = path.resolve(process.cwd(), "public", "agentic-rag");
app.use("/agentic-rag", express.static(uiRoot));

function isValidEmail(email: string): boolean {
  return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
}

app.get("/", (_req: Request, res: Response) => {
  res.redirect("/agentic-rag");
});

app.get("/api/health", (_req: Request, res: Response) => {
  res.json({ status: "ok", example: "agentic-rag" });
});

app.post("/api/init", (req: Request, res: Response) => {
  const emailRaw = typeof req.body?.email === "string" ? req.body.email.trim() : "";

  if (!isValidEmail(emailRaw)) {
    return res.status(400).json({ error: "Invalid email address" });
  }

  return res.json({ threadId: emailRaw });
});

app.post("/api/chat", async (req: Request, res: Response) => {
  try {
    const threadId = typeof req.body?.threadId === "string" ? req.body.threadId.trim() : "";
    const message = typeof req.body?.message === "string" ? req.body.message.trim() : "";

    if (!threadId || !message) {
      return res.status(400).json({ error: "Missing threadId or message" });
    }

    const result = await runAgenticRagChat(threadId, message);

    return res.json({
      response: result.response,
      route: result.route,
      workflow: result.workflow,
      sources: result.sources,
      memory: result.memory,
      traceId: result.traceId,
    });
  } catch (error) {
    const detail = error instanceof Error ? error.message : "Unknown error";
    return res.status(500).json({ error: detail });
  }
});

app.get("/api/history/:threadId", (req: Request, res: Response) => {
  const threadId = Array.isArray(req.params.threadId)
    ? req.params.threadId[0]
    : req.params.threadId;
  return res.json({ history: getThreadTranscript(threadId) });
});

app.get("/api/workflow/:threadId", (req: Request, res: Response) => {
  const threadId = Array.isArray(req.params.threadId)
    ? req.params.threadId[0]
    : req.params.threadId;
  return res.json({
    workflow: getThreadWorkflow(threadId),
    sources: getThreadSources(threadId),
    memory: getThreadMemorySnapshot(threadId),
  });
});

app.delete("/api/history/:threadId", (req: Request, res: Response) => {
  const threadId = Array.isArray(req.params.threadId)
    ? req.params.threadId[0]
    : req.params.threadId;
  clearThread(threadId);
  return res.json({ message: "Thread state cleared" });
});

const PORT = Number(process.env.PORT ?? 3100);
app.listen(PORT, () => {
  console.log("============================================");
  console.log("Agentic RAG example server is running");
  console.log(`Open UI: http://localhost:${PORT}/agentic-rag`);
  console.log(`API: http://localhost:${PORT}/api`);
  console.log("============================================");
});
