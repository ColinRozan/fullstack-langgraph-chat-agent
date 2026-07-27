import { useStream } from "@langchain/langgraph-sdk/react";
import type { Message } from "@langchain/langgraph-sdk";
import { useState, useEffect, useRef, useCallback } from "react";
import { Loader2 } from "lucide-react";
import { ProcessedEvent } from "@/components/ActivityTimeline";
import { WelcomeScreen } from "@/components/WelcomeScreen";
import { ChatMessagesView } from "@/components/ChatMessagesView";
import { Button } from "@/components/ui/button";
import { Sidebar, type ThreadMeta } from "@/components/Sidebar";
import {
  listThreads,
  updateThread,
  deleteThread as deleteThreadApi,
} from "@/lib/api";

const ACTIVE_KEY = "lg-active-thread";

function loadActiveThreadId(): string | null {
  try {
    return localStorage.getItem(ACTIVE_KEY);
  } catch {}
  return null;
}

function saveActiveThreadId(id: string | null) {
  if (id) localStorage.setItem(ACTIVE_KEY, id);
  else localStorage.removeItem(ACTIVE_KEY);
}

function generateTitle(messages: Message[]): string {
  const firstHuman = messages.find((m) => m.type === "human");
  if (!firstHuman) return "新对话";
  const text =
    typeof firstHuman.content === "string"
      ? firstHuman.content
      : JSON.stringify(firstHuman.content);
  return text.slice(0, 24) + (text.length > 24 ? "…" : "");
}

/* ------------------------------------------------------------------ */
/* Inner component: one chat session bound to a single threadId       */
/* Remounts via key={threadId} whenever the user switches threads.    */
/* ------------------------------------------------------------------ */

interface ChatSessionProps {
  threadId: string | null;
  threadTitle?: string;
  onThreadCreated: (id: string, title: string) => void;
  onThreadUpdated: (id: string, title: string) => void;
}

function ChatSession({
  threadId,
  threadTitle,
  onThreadCreated,
  onThreadUpdated,
}: ChatSessionProps) {
  const [processedEventsTimeline, setProcessedEventsTimeline] = useState<
    ProcessedEvent[]
  >([]);
  const [historicalActivities, setHistoricalActivities] = useState<
    Record<string, ProcessedEvent[]>
  >({});
  const [messageSources, setMessageSources] = useState<
    Record<string, { rag_sources: any[]; web_sources: any[] }>
  >({});
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const hasFinalizeEventOccurredRef = useRef(false);
  const pendingSourcesRef = useRef<{
    rag_sources: any[];
    web_sources: any[];
  } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const threadCreatedRef = useRef(false);
  const [resolvedThreadId, setResolvedThreadId] = useState<string | null>(threadId);
  const lastTitleRef = useRef<string>("");

  const thread = useStream<{
    messages: Message[];
    initial_search_query_count: number;
    max_research_loops: number;
    reasoning_model: string;
  }>({
    apiUrl: import.meta.env.DEV
      ? "http://localhost:2024"
      : "http://localhost:8123",
    assistantId: "agent",
    threadId,
    onThreadId: (id) => {
      setResolvedThreadId(id);
      if (!threadCreatedRef.current && id) {
        threadCreatedRef.current = true;
        onThreadCreated(id, "新对话");
      }
    },
    messagesKey: "messages",
    onUpdateEvent: (event: any) => {
      let processedEvent: ProcessedEvent | null = null;
      if (event.generate_query) {
        processedEvent = {
          title: "Generating Search Queries",
          data: event.generate_query?.search_query?.join(", ") || "",
        };
      } else if (event.web_research) {
        const sources = event.web_research.sources_gathered || [];
        const numSources = sources.length;
        const uniqueLabels = [
          ...new Set(sources.map((s: any) => s.label).filter(Boolean)),
        ];
        const exampleLabels = uniqueLabels.slice(0, 3).join(", ");
        processedEvent = {
          title: "Web Research",
          data: `Gathered ${numSources} sources. Related to: ${
            exampleLabels || "N/A"
          }.`,
        };
      } else if (event.rag_retrieve) {
        const docs = event.rag_retrieve.rag_documents || [];
        processedEvent = {
          title: "RAG Retrieve",
          data: `Retrieved ${docs.length} documents from knowledge base.`,
        };
      } else if (event.reflection) {
        processedEvent = {
          title: "Reflection",
          data: "Analysing Web Research Results",
        };
      } else if (event.finalize_answer) {
        const fa = event.finalize_answer;
        const ragCount = (fa.rag_sources || []).length;
        const webCount = (fa.sources_gathered || []).length;
        processedEvent = {
          title: "Finalizing Answer",
          data: `Composing answer. Sources: ${ragCount} knowledge base, ${webCount} web.`,
        };
        pendingSourcesRef.current = {
          rag_sources: fa.rag_sources || [],
          web_sources: fa.sources_gathered || [],
        };
        hasFinalizeEventOccurredRef.current = true;
      }
      if (processedEvent) {
        setProcessedEventsTimeline((prevEvents) => [
          ...prevEvents,
          processedEvent!,
        ]);
      }
    },
    onError: (error: any) => {
      setError(error.message);
    },
  });

  // Auto-scroll
  useEffect(() => {
    if (scrollAreaRef.current) {
      const scrollViewport = scrollAreaRef.current.querySelector(
        "[data-radix-scroll-area-viewport]"
      );
      if (scrollViewport) {
        scrollViewport.scrollTop = scrollViewport.scrollHeight;
      }
    }
  }, [thread.messages]);

  // Capture finalize + update thread title when first human msg arrives
  useEffect(() => {
    if (
      hasFinalizeEventOccurredRef.current &&
      !thread.isLoading &&
      thread.messages.length > 0
    ) {
      const lastMessage = thread.messages[thread.messages.length - 1];
      if (lastMessage && lastMessage.type === "ai" && lastMessage.id) {
        setHistoricalActivities((prev) => ({
          ...prev,
          [lastMessage.id!]: [...processedEventsTimeline],
        }));
        if (pendingSourcesRef.current) {
          setMessageSources((prev) => ({
            ...prev,
            [lastMessage.id!]: pendingSourcesRef.current!,
          }));
          pendingSourcesRef.current = null;
        }
      }
      hasFinalizeEventOccurredRef.current = false;
    }
  }, [thread.messages, thread.isLoading, processedEventsTimeline]);

  // Update thread title from messages
  useEffect(() => {
    const id = resolvedThreadId ?? threadId;
    if (thread.messages.length > 0 && id) {
      const title = generateTitle(thread.messages);
      if (title !== lastTitleRef.current) {
        lastTitleRef.current = title;
        onThreadUpdated(id, title);
      }
    }
  }, [thread.messages, resolvedThreadId, threadId]);

  const handleSubmit = useCallback(
    (submittedInputValue: string, effort: string, model: string) => {
      if (!submittedInputValue.trim()) return;
      setProcessedEventsTimeline([]);
      hasFinalizeEventOccurredRef.current = false;

      let initial_search_query_count = 0;
      let max_research_loops = 0;
      switch (effort) {
        case "low":
          initial_search_query_count = 1;
          max_research_loops = 1;
          break;
        case "medium":
          initial_search_query_count = 3;
          max_research_loops = 3;
          break;
        case "high":
          initial_search_query_count = 5;
          max_research_loops = 10;
          break;
      }

      const newMessages: Message[] = [
        ...(thread.messages || []),
        {
          type: "human",
          content: submittedInputValue,
          id: Date.now().toString(),
        },
      ];
      thread.submit({
        messages: newMessages,
        initial_search_query_count,
        max_research_loops,
        reasoning_model: model,
      });
    },
    [thread]
  );

  const handleCancel = useCallback(() => {
    thread.stop();
  }, [thread]);

  const hasMessages = thread.messages.length > 0;

  return (
    <div className="flex h-full w-full">
      <div className="flex-1 overflow-hidden">
        {!hasMessages && !thread.isLoading && threadId == null ? (
          <WelcomeScreen
            handleSubmit={handleSubmit}
            isLoading={thread.isLoading}
            onCancel={handleCancel}
          />
        ) : !hasMessages && thread.isLoading ? (
          <div className="flex flex-col items-center justify-center h-full gap-3">
            <Loader2 className="h-8 w-8 animate-spin text-neutral-400" />
            <span className="text-neutral-400 text-sm">正在加载对话…</span>
          </div>
        ) : !hasMessages && threadId != null ? (
          <div className="flex flex-col items-center justify-center h-full gap-3">
            <span className="text-neutral-400 text-sm">
              此对话暂无消息。后端数据可能已丢失（开发模式为内存存储）。
            </span>
            <Button
              variant="outline"
              onClick={() => window.location.reload()}
            >
              刷新重试
            </Button>
          </div>
        ) : error ? (
          <div className="flex flex-col items-center justify-center h-full">
            <div className="flex flex-col items-center justify-center gap-4">
              <h1 className="text-2xl text-red-400 font-bold">Error</h1>
              <p className="text-red-400">{JSON.stringify(error)}</p>
              <Button
                variant="destructive"
                onClick={() => window.location.reload()}
              >
                Retry
              </Button>
            </div>
          </div>
        ) : (
          <ChatMessagesView
            messages={thread.messages}
            isLoading={thread.isLoading}
            scrollAreaRef={scrollAreaRef}
            onSubmit={handleSubmit}
            onCancel={handleCancel}
            liveActivityEvents={processedEventsTimeline}
            historicalActivities={historicalActivities}
            messageSources={messageSources}
            threadTitle={threadTitle}
          />
        )}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Outer App: manages sidebar + thread list + active thread switching */
/* ------------------------------------------------------------------ */

export default function App() {
  const [threads, setThreads] = useState<ThreadMeta[]>([]);
  const [activeThreadId, setActiveThreadId] = useState<string | null>(
    loadActiveThreadId()
  );
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [threadsLoading, setThreadsLoading] = useState(true);

  // Load threads from backend on mount
  useEffect(() => {
    let cancelled = false;
    listThreads()
      .then((data) => {
        if (cancelled) return;
        const loaded = data.map((t) => ({
          id: t.id,
          title: t.title,
          createdAt: t.createdAt ? new Date(t.createdAt).getTime() : Date.now(),
          updatedAt: t.updatedAt ? new Date(t.updatedAt).getTime() : Date.now(),
        }));
        setThreads(loaded);
        // If the saved active thread no longer exists on the backend,
        // clear it so we don’t show a “missing thread” blank state.
        const savedActive = loadActiveThreadId();
        if (savedActive && !loaded.find((t) => t.id === savedActive)) {
          setActiveThreadId(null);
          saveActiveThreadId(null);
        }
      })
      .catch((err) => {
        console.error("Failed to load threads:", err);
      })
      .finally(() => {
        if (!cancelled) setThreadsLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const handleNewChat = useCallback(() => {
    setActiveThreadId(null);
    saveActiveThreadId(null);
  }, []);

  const handleSelectThread = useCallback((id: string) => {
    setActiveThreadId(id);
    saveActiveThreadId(id);
  }, []);

  const handleDeleteThread = useCallback(
    async (id: string) => {
      try {
        await deleteThreadApi(id);
      } catch (err) {
        console.error("Failed to delete thread:", err);
      }
      const next = threads.filter((t) => t.id !== id);
      setThreads(next);
      if (activeThreadId === id) {
        setActiveThreadId(null);
        saveActiveThreadId(null);
      }
    },
    [threads, activeThreadId]
  );

  const handleThreadCreated = useCallback(
    async (id: string, title: string) => {
      const now = Date.now();
      const next: ThreadMeta[] = [
        { id, title, createdAt: now, updatedAt: now },
        ...threads.filter((t) => t.id !== id),
      ];
      setThreads(next);
      try {
        await updateThread(id, title);
      } catch (err) {
        console.error("Failed to create thread on backend:", err);
      }
    },
    [threads]
  );

  const handleThreadUpdated = useCallback(
    async (id: string, title: string) => {
      setThreads((prev) => {
        const exists = prev.find((t) => t.id === id);
        if (!exists) return prev;
        if (exists.title === title) return prev;
        const next = prev.map((t) =>
          t.id === id ? { ...t, title, updatedAt: Date.now() } : t
        );
        return next;
      });
      try {
        await updateThread(id, title);
      } catch (err) {
        console.error("Failed to update thread title:", err);
      }
    },
    []
  );

  return (
    <div className="flex h-screen bg-neutral-800 text-neutral-100 font-sans antialiased">
      <Sidebar
        threads={threads}
        activeThreadId={activeThreadId}
        isOpen={sidebarOpen}
        onToggle={() => setSidebarOpen((v) => !v)}
        onSelect={handleSelectThread}
        onNewChat={handleNewChat}
        onDelete={handleDeleteThread}
        loading={threadsLoading}
      />
      <main className="flex-1 h-full overflow-hidden relative">
        <ChatSession
          key={activeThreadId ?? "__new__"}
          threadId={activeThreadId}
          threadTitle={threads.find((t) => t.id === activeThreadId)?.title}
          onThreadCreated={handleThreadCreated}
          onThreadUpdated={handleThreadUpdated}
        />
      </main>
    </div>
  );
}
