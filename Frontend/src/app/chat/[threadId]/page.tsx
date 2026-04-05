"use client";

import { motion } from "framer-motion";
import { useParams, useRouter } from "next/navigation";
import { useCallback, useEffect, useRef, useState } from "react";
import { flushSync } from "react-dom";
import { ChatPillInput } from "@/components/chat/ChatPillInput";
import {
  normalizeUiEvents,
  PlanningEventsPanel,
} from "@/components/chat/PlanningEventsPanel";
import { MessageMarkdown } from "@/components/chat/MessageMarkdown";
import { API_BASE, api, getToken } from "@/lib/api";
import { CHAT_COLUMN } from "@/lib/chatConstants";
import { THREADS_REFRESH, dispatchPlanningActive } from "@/lib/appEvents";
import {
  clearPlanningJob,
  readPlanningJob,
  savePlanningJob,
} from "@/lib/planningJobStorage";

const PENDING_MESSAGE_KEY = "arkon_pending_message";

function planningWsUrl(jobId: string): string | null {
  if (typeof window === "undefined" || !jobId) return null;
  const tok = getToken();
  if (!tok) return null;
  const base = API_BASE.replace(/^http/, "ws").replace(/\/$/, "");
  return `${base}/v1/ws/jobs/${encodeURIComponent(jobId)}?token=${encodeURIComponent(tok)}`;
}

type Msg = {
  id?: string;
  role: string;
  content: string;
  agent?: string | null;
  created_at?: string;
  metadata?: Record<string, unknown>;
};

type QuickReply = { id: string; label: string; value: string; kind: string };

type ForkV = {
  has_fork_versions: boolean;
  branch_ids: string[];
  current_index: number;
  total: number;
  active_in_family?: boolean;
};

type UiEvent = Record<string, unknown>;

export default function ThreadPage() {
  const params = useParams();
  const router = useRouter();
  const threadId = params.threadId as string;
  const [messages, setMessages] = useState<Msg[]>([]);
  const [phase, setPhase] = useState("REQUIREMENTS");
  const [threadTitle, setThreadTitle] = useState("");
  const [input, setInput] = useState("");
  const [quick, setQuick] = useState<QuickReply[]>([]);
  const [loading, setLoading] = useState(false);
  const [busyHint, setBusyHint] = useState("");
  const [mode, setMode] = useState<"main" | "dev">("main");
  const [reportOpen, setReportOpen] = useState(false);
  const [reportJson, setReportJson] = useState<unknown>(null);
  const [activeBranchId, setActiveBranchId] = useState<string | null>(null);
  const [editForkId, setEditForkId] = useState<string | null>(null);
  const [editForkText, setEditForkText] = useState("");
  const [forkByMsg, setForkByMsg] = useState<Record<string, ForkV>>({});
  const [planningUiEvents, setPlanningUiEvents] = useState<UiEvent[]>([]);
  const [planningStreamOpen, setPlanningStreamOpen] = useState(false);
  const [pdfArtifactId, setPdfArtifactId] = useState<string | null>(null);
  const [reportShowPdf, setReportShowPdf] = useState(true);
  const [headerMenu, setHeaderMenu] = useState(false);
  const [renameOpen, setRenameOpen] = useState(false);
  const [renameDraft, setRenameDraft] = useState("");
  const [deleteOpen, setDeleteOpen] = useState(false);
  const [shareToast, setShareToast] = useState("");
  const [showScrollDown, setShowScrollDown] = useState(false);
  const [copyToast, setCopyToast] = useState("");
  const [planningExpandedOpen, setPlanningExpandedOpen] = useState(false);
  const [expandedPlanningEvents, setExpandedPlanningEvents] = useState<
    UiEvent[] | null
  >(null);
  const [branchPlanningTranscript, setBranchPlanningTranscript] = useState<
    UiEvent[]
  >([]);
  const bottomRef = useRef<HTMLDivElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const autoTitleAttemptedRef = useRef(false);
  const pendingBootstrapRef = useRef(false);
  const lastThreadForPendingRef = useRef<string | null>(null);
  const loadGenerationRef = useRef(0);
  const pollAbortRef = useRef<AbortController | null>(null);

  const scrollDown = () =>
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });

  const loadForkMap = useCallback(
    async (msgs: Msg[]) => {
      const keys = new Set<string>();
      for (const m of msgs) {
        if (m.role !== "user" || !m.id) continue;
        const anchor = m.metadata?.fork_anchor_message_id;
        keys.add(typeof anchor === "string" ? anchor : m.id);
      }
      const entries = await Promise.all(
        [...keys].map(async (id) => {
          try {
            const v = await api<ForkV>(
              `/v1/threads/${threadId}/messages/${id}/fork-versions`
            );
            return [id, v] as const;
          } catch {
            return [id, null] as const;
          }
        })
      );
      const next: Record<string, ForkV> = {};
      for (const [id, v] of entries) {
        if (!v || !v.has_fork_versions || v.total < 2) continue;
        if (v.active_in_family === false) continue;
        next[id] = v;
      }
      setForkByMsg(next);
    },
    [threadId]
  );

  const load = useCallback(
    async (opts?: { scrollToBottom?: boolean }) => {
      const gen = ++loadGenerationRef.current;
      const th = await api<{
        phase: string;
        active_branch_id: string;
        title: string;
      }>(`/v1/threads/${threadId}`);
      if (gen !== loadGenerationRef.current) return [];

      setPhase(th.phase);
      setActiveBranchId(th.active_branch_id);
      setThreadTitle(th.title);
      const q = th.active_branch_id
        ? `?branch_id=${encodeURIComponent(th.active_branch_id)}&limit=2000`
        : "?limit=2000";
      const data = await api<{
        items: Msg[];
        branch_id: string;
        planning_transcript?: unknown[];
      }>(`/v1/threads/${threadId}/messages${q}`);
      if (gen !== loadGenerationRef.current) return [];

      const pt = Array.isArray(data.planning_transcript)
        ? normalizeUiEvents(data.planning_transcript)
        : [];
      setBranchPlanningTranscript(pt);
      if (pt.length > 0) {
        setPlanningUiEvents(pt);
      }

      setMessages((prev) => {
        const temps = prev.filter((m) =>
          String(m.id || "").startsWith("temp-")
        );
        let next = [...data.items];
        for (const tm of temps) {
          if (tm.role !== "user") continue;
          const replaced = next.some(
            (m) => m.role === "user" && m.content === tm.content
          );
          if (!replaced) next = [...next, tm];
        }
        return next;
      });
      await loadForkMap(data.items);
      if (gen !== loadGenerationRef.current) return data.items;
      if (th.phase === "DEVELOPMENT") {
        try {
          const arts = await api<{ items: { id: string; kind: string }[] }>(
            `/v1/threads/${threadId}/artifacts`
          );
          if (gen !== loadGenerationRef.current) return data.items;
          const pdf = arts.items?.find((x) => x.kind === "pdf");
          setPdfArtifactId(pdf?.id ?? null);
        } catch {
          setPdfArtifactId(null);
        }
      } else {
        setPdfArtifactId(null);
      }
      if (opts?.scrollToBottom) {
        requestAnimationFrame(() => scrollDown());
      }
      return data.items;
    },
    [threadId, loadForkMap]
  );

  const loadRef = useRef(load);
  loadRef.current = load;

  useEffect(() => {
    setPlanningUiEvents([]);
    setBranchPlanningTranscript([]);
  }, [threadId]);

  useEffect(() => {
    load({ scrollToBottom: true }).catch(() => {});
  }, [load]);

  useEffect(() => {
    if (lastThreadForPendingRef.current !== threadId) {
      lastThreadForPendingRef.current = threadId;
      pendingBootstrapRef.current = false;
    }
    if (pendingBootstrapRef.current || !threadId || !activeBranchId || loading) {
      return;
    }
    let raw: string | null = null;
    try {
      raw = sessionStorage.getItem(PENDING_MESSAGE_KEY);
    } catch {
      return;
    }
    if (!raw) return;
    let content: string | null = null;
    try {
      const parsed = JSON.parse(raw) as { threadId?: string; content?: string };
      if (parsed.threadId !== threadId || !parsed.content?.trim()) return;
      content = parsed.content.trim();
    } catch {
      try {
        sessionStorage.removeItem(PENDING_MESSAGE_KEY);
      } catch {
        /* ignore */
      }
      return;
    }
    try {
      sessionStorage.removeItem(PENDING_MESSAGE_KEY);
    } catch {
      /* ignore */
    }
    pendingBootstrapRef.current = true;
    void send(content);
    // eslint-disable-next-line react-hooks/exhaustive-deps -- bootstrap once when branch is ready
  }, [threadId, activeBranchId, loading]);

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const onScroll = () => {
      const { scrollTop, scrollHeight, clientHeight } = el;
      const nearBottom = scrollHeight - scrollTop - clientHeight < 120;
      setShowScrollDown(!nearBottom && scrollHeight > clientHeight + 40);
    };
    el.addEventListener("scroll", onScroll, { passive: true });
    onScroll();
    return () => el.removeEventListener("scroll", onScroll);
  }, [messages.length, loading]);

  useEffect(() => {
    if (autoTitleAttemptedRef.current) return;
    if (messages.length < 6) return;
    autoTitleAttemptedRef.current = true;
    (async () => {
      try {
        const r = await api<{ updated?: boolean }>(
          `/v1/threads/${threadId}/auto-title`,
          { method: "POST" }
        );
        if (r.updated) {
          window.dispatchEvent(new Event(THREADS_REFRESH));
          const th = await api<{ title: string }>(`/v1/threads/${threadId}`);
          setThreadTitle(th.title);
        }
      } catch {
        /* optional */
      }
    })();
  }, [messages.length, threadId, threadTitle]);

  const pollJob = useCallback(
    async (
      statusPath: string,
      jobId: string | undefined,
      opts?: { signal?: AbortSignal; skipSave?: boolean }
    ) => {
      const signal = opts?.signal;
      if (signal?.aborted) return;
      if (threadId && jobId && !opts?.skipSave) {
        savePlanningJob(threadId, jobId, statusPath);
      }
      dispatchPlanningActive(threadId, true);

      const tickRunning = 700;
      const tickSlow = 2000;

      const sleep = (ms: number) =>
        new Promise<void>((resolve) => {
          const t = setTimeout(resolve, ms);
          const onAbort = () => {
            clearTimeout(t);
            resolve();
          };
          signal?.addEventListener("abort", onAbort, { once: true });
        });

      let ws: WebSocket | null = null;
      const url = jobId ? planningWsUrl(jobId) : null;
      if (url && typeof WebSocket !== "undefined") {
        try {
          ws = new WebSocket(url);
          ws.onmessage = (ev) => {
            try {
              const msg = JSON.parse(ev.data as string) as {
                type?: string;
                events?: unknown[];
              };
              if (msg.type === "ui_events" && Array.isArray(msg.events)) {
                setPlanningUiEvents(normalizeUiEvents(msg.events));
              }
            } catch {
              /* ignore non-JSON / ping */
            }
          };
        } catch {
          ws = null;
        }
      }

      const finishPersist = () => {
        if (signal?.aborted || !threadId) return;
        clearPlanningJob(threadId);
        dispatchPlanningActive(threadId, false);
      };

      try {
        while (true) {
          if (signal?.aborted) return;
          let data: {
            state: string;
            error?: string;
            phase?: string | null;
            result?: {
              phase?: string;
              ui_events?: unknown[];
              quick_replies?: QuickReply[];
            };
            ui_events?: unknown[];
          };
          try {
            data = await api<typeof data>(statusPath, { signal });
          } catch {
            if (signal?.aborted) return;
            throw new Error("Lost connection while polling planning job");
          }
          const live =
            data.state === "queued" ||
            data.state === "running" ||
            data.state === "failed";
          if (live) {
            const raw = data.ui_events;
            if (Array.isArray(raw) && raw.length) {
              setPlanningUiEvents(normalizeUiEvents(raw));
            }
          }
          if (data.state === "completed") {
            if (data.phase) setPhase(data.phase);
            else if (data.result?.phase) setPhase(data.result.phase);
            const raw = data.ui_events ?? data.result?.ui_events;
            if (Array.isArray(raw) && raw.length) {
              setPlanningUiEvents(normalizeUiEvents(raw));
            }
            const res = data.result;
            if (res && Array.isArray(res.quick_replies)) {
              setQuick(res.quick_replies);
            }
            finishPersist();
            return;
          }
          if (data.state === "failed") {
            finishPersist();
            throw new Error(data.error || "Background job failed");
          }
          const delay =
            data.state === "queued" || data.state === "running"
              ? tickRunning
              : tickSlow;
          await sleep(delay);
        }
      } finally {
        try {
          ws?.close();
        } catch {
          /* ignore */
        }
      }
    },
    [threadId]
  );

  useEffect(() => {
    pollAbortRef.current?.abort();
    pollAbortRef.current = null;
    return () => {
      pollAbortRef.current?.abort();
      pollAbortRef.current = null;
    };
  }, [threadId]);

  useEffect(() => {
    if (!threadId || !getToken()) return;
    const stored = readPlanningJob(threadId);
    if (!stored) return;

    const ac = new AbortController();
    pollAbortRef.current = ac;

    void (async () => {
      try {
        let probe: { state: string };
        try {
          probe = await api<{ state: string }>(stored.statusUrl, {
            signal: ac.signal,
          });
        } catch {
          if (ac.signal.aborted) return;
          clearPlanningJob(threadId);
          dispatchPlanningActive(threadId, false);
          return;
        }
        if (ac.signal.aborted) return;
        if (probe.state !== "queued" && probe.state !== "running") {
          clearPlanningJob(threadId);
          dispatchPlanningActive(threadId, false);
          return;
        }

        setPlanningStreamOpen(true);
        setLoading(true);
        setBusyHint("Planning in progress — reconnecting to live stream…");
        dispatchPlanningActive(threadId, true);

        await pollJob(stored.statusUrl, stored.jobId, {
          signal: ac.signal,
          skipSave: true,
        });
        await loadRef.current({ scrollToBottom: true });
        window.dispatchEvent(new Event(THREADS_REFRESH));
      } catch (e) {
        if (!ac.signal.aborted) {
          console.error(e);
        }
      } finally {
        if (pollAbortRef.current === ac) {
          pollAbortRef.current = null;
        }
        if (!ac.signal.aborted) {
          setLoading(false);
          setBusyHint("");
          setPlanningStreamOpen(false);
          setPlanningUiEvents([]);
        }
      }
    })();

    return () => {
      ac.abort();
    };
  }, [threadId, pollJob]);

  async function switchForkVersion(forkAnchorMessageId: string, delta: -1 | 1) {
    const fv = forkByMsg[forkAnchorMessageId];
    if (!fv || !fv.branch_ids.length) return;
    const next = fv.current_index + delta;
    if (next < 0 || next >= fv.branch_ids.length) return;
    const nextBranch = fv.branch_ids[next];
    setLoading(true);
    try {
      await api(`/v1/threads/${threadId}`, {
        method: "PATCH",
        body: JSON.stringify({ active_branch_id: nextBranch }),
      });
      await load({ scrollToBottom: true });
      window.dispatchEvent(new Event(THREADS_REFRESH));
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  }

  function applyUiEventsFromBody(body: Record<string, unknown>) {
    const raw = body.ui_events;
    if (Array.isArray(raw) && raw.length) {
      setPlanningUiEvents(normalizeUiEvents(raw));
    }
  }

  async function send(text: string) {
    const t = text.trim();
    if (!t || loading || !activeBranchId) return;
    if (phase === "DEVELOPMENT" && mode === "main") return;

    setQuick([]);

    /** Prefer `planning_confirmation_prompt` (engine-gated); fall back for older stored messages. */
    const lastAssistantForHandoff = [...messages]
      .reverse()
      .find((m) => m.role === "assistant");
    const meta = lastAssistantForHandoff?.metadata;
    const hasPlanningPromptKey =
      meta != null &&
      Object.prototype.hasOwnProperty.call(meta, "planning_confirmation_prompt");
    const planningConfirmationPrompt =
      meta?.planning_confirmation_prompt === true ||
      (!hasPlanningPromptKey && meta?.suggest_stream_planning_next === true);

    const useBackground =
      mode === "main" &&
      (phase === "PLANNING" ||
        (phase === "REQUIREMENTS" && planningConfirmationPrompt));

    const tempMsg: Msg = {
      id: `temp-${Date.now()}`,
      role: "user",
      content: t,
      created_at: new Date().toISOString(),
    };

    flushSync(() => {
      setMessages((prev) => [...prev, tempMsg]);
    });
    requestAnimationFrame(() => scrollDown());

    setLoading(true);
    if (useBackground) {
      setPlanningUiEvents([]);
      setPlanningStreamOpen(true);
    }
    setBusyHint(
      useBackground
        ? "Planning in progress — live updates below"
        : "Thinking…"
    );
  
    try {
      if (mode === "dev" && phase === "DEVELOPMENT") {
        await api(`/v1/threads/${threadId}/development/messages`, {
          method: "POST",
          body: JSON.stringify({ content: t, branch_id: activeBranchId }),
        });
      } else {
        const headers: HeadersInit = {
          "Content-Type": "application/json",
          Authorization: `Bearer ${getToken() || ""}`,
        };
  
        const res = await fetch(
          `${API_BASE}/v1/threads/${threadId}/messages`,
          {
            method: "POST",
            headers,
            body: JSON.stringify({
              content: t,
              branch_id: activeBranchId,
              background: useBackground,
            }),
          }
        );
  
        const body = (await res.json().catch(() => ({}))) as Record<
          string,
          unknown
        >;
  
        if (!res.ok) {
          throw new Error(
            typeof body?.detail === "string"
              ? body.detail
              : res.statusText || "Request failed"
          );
        }
  
        if (res.status === 202 && body.status_url) {
          pollAbortRef.current?.abort();
          const ac = new AbortController();
          pollAbortRef.current = ac;
          await pollJob(
            body.status_url as string,
            typeof body.job_id === "string" ? body.job_id : undefined,
            { signal: ac.signal }
          );
        } else {
          if (body.phase) setPhase(body.phase as string);
          setQuick((body.quick_replies as QuickReply[]) || []);
          applyUiEventsFromBody(body);
        }
      }

      await load({ scrollToBottom: true });
      window.dispatchEvent(new Event(THREADS_REFRESH));
    } catch (e) {
      console.error(e);

      setMessages((prev) => prev.filter((m) => m.id !== tempMsg.id));
    } finally {
      setLoading(false);
      setBusyHint("");
      setPlanningStreamOpen(false);
      setPlanningUiEvents([]);
    }
  }

  async function regenerateForMessage(msgId: string) {
    if (!activeBranchId) return;
    setLoading(true);
    try {
      await api(`/v1/threads/${threadId}/messages/${msgId}/regenerate`, {
        method: "POST",
        body: JSON.stringify({ branch_id: activeBranchId }),
      });
      await load({ scrollToBottom: true });
      window.dispatchEvent(new Event(THREADS_REFRESH));
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  }

  async function submitEditFork() {
    if (!editForkId || !activeBranchId) return;
    const t = editForkText.trim();
    if (!t) return;
    setLoading(true);
    setEditForkId(null);
    try {
      const out = await api<{
        branch_id: string;
        assistant_turn?: {
          phase?: string;
          quick_replies?: QuickReply[];
        };
      }>(`/v1/threads/${threadId}/messages/${editForkId}/edit`, {
        method: "POST",
        body: JSON.stringify({
          new_content: t,
          branch_id: activeBranchId,
        }),
      });
      if (out.assistant_turn?.phase) setPhase(out.assistant_turn.phase);
      if (out.assistant_turn?.quick_replies)
        setQuick(out.assistant_turn.quick_replies);
      await load({ scrollToBottom: true });
      window.dispatchEvent(new Event(THREADS_REFRESH));
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
      setEditForkText("");
    }
  }

  async function exportMd() {
    const base = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
    const q = activeBranchId
      ? `?fmt=markdown&branch_id=${encodeURIComponent(activeBranchId)}`
      : "?fmt=markdown";
    const res = await fetch(`${base}/v1/threads/${threadId}/export${q}`, {
      headers: { Authorization: `Bearer ${getToken()}` },
    });
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "chat.md";
    a.click();
    URL.revokeObjectURL(url);
  }

  async function openReport() {
    setReportJson(null);
    setReportShowPdf(true);
    let pdfId: string | null = null;
    try {
      const arts = await api<{ items: { id: string; kind: string }[] }>(
        `/v1/threads/${threadId}/artifacts`
      );
      const pdf = arts.items?.find((x) => x.kind === "pdf");
      if (pdf) pdfId = pdf.id;
    } catch {
      /* list failed */
    }
    if (!pdfId) pdfId = pdfArtifactId;
    if (pdfId) {
      setPdfArtifactId(pdfId);
      setReportOpen(true);
      return;
    }
    try {
      const data = await api<unknown>(`/v1/threads/${threadId}/report`);
      setReportJson(data);
      setReportShowPdf(false);
      setReportOpen(true);
    } catch {
      setReportJson(null);
      setReportShowPdf(false);
      setReportOpen(true);
    }
  }

  async function startPlanningJob() {
    if (phase !== "PLANNING" || loading) return;
    setPlanningUiEvents([]);
    setPlanningStreamOpen(true);
    setLoading(true);
    setBusyHint("Running planning cycle in background…");
    try {
      const headers: HeadersInit = {
        "Content-Type": "application/json",
        Authorization: `Bearer ${getToken() || ""}`,
      };
      const res = await fetch(
        `${API_BASE}/v1/threads/${threadId}/planning/jobs`,
        { method: "POST", headers }
      );
      const body = (await res.json().catch(() => ({}))) as Record<
        string,
        unknown
      >;
      if (!res.ok) throw new Error(String(body?.detail || res.statusText));
      if (body.status_url) {
        pollAbortRef.current?.abort();
        const ac = new AbortController();
        pollAbortRef.current = ac;
        await pollJob(
          body.status_url as string,
          typeof body.job_id === "string" ? body.job_id : undefined,
          { signal: ac.signal }
        );
      }
      await load({ scrollToBottom: true });
      window.dispatchEvent(new Event(THREADS_REFRESH));
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
      setBusyHint("");
      setPlanningStreamOpen(false);
    }
  }

  async function createShareLink() {
    if (!activeBranchId) return;
    try {
      const out = await api<{ token: string }>("/v1/share", {
        method: "POST",
        body: JSON.stringify({
          thread_id: threadId,
          branch_id: activeBranchId,
        }),
      });
      const path = `/share/${out.token}`;
      const full = `${window.location.origin}${path}`;
      await navigator.clipboard.writeText(full);
      setShareToast("Read-only link copied");
      setTimeout(() => setShareToast(""), 3500);
      setHeaderMenu(false);
    } catch (e) {
      console.error(e);
      setShareToast("Could not create link");
      setTimeout(() => setShareToast(""), 3000);
    }
  }

  async function saveRename() {
    const t = renameDraft.trim();
    if (!t) return;
    try {
      await api(`/v1/threads/${threadId}`, {
        method: "PATCH",
        body: JSON.stringify({ title: t }),
      });
      setThreadTitle(t);
      setRenameOpen(false);
      window.dispatchEvent(new Event(THREADS_REFRESH));
    } catch (e) {
      console.error(e);
    }
  }

  async function confirmDelete() {
    try {
      await api(`/v1/threads/${threadId}`, { method: "DELETE" });
      window.dispatchEvent(new Event(THREADS_REFRESH));
      router.push("/chat");
    } catch (e) {
      console.error(e);
    }
  }

  function copyText(text: string) {
    void navigator.clipboard.writeText(text);
    setCopyToast("Copied");
    window.setTimeout(() => setCopyToast(""), 2000);
  }

  const hasMessages = messages.length > 0;
  const lastAssistant = [...messages]
    .reverse()
    .find((m) => m.role === "assistant");
  const maxInlinePlanningLen = Math.max(
    0,
    ...messages.map((m) =>
      m.role === "assistant" &&
      Array.isArray(m.metadata?.planning_ui_events)
        ? m.metadata.planning_ui_events.length
        : 0
    )
  );
  const showSavedBranchPlanning =
    phase !== "REQUIREMENTS" &&
    branchPlanningTranscript.length > 0 &&
    branchPlanningTranscript.length > maxInlinePlanningLen;
  /** Prefer live buffer; during active stream with no chunks yet, don't fall back to stale branch. */
  const planningPanelEvents =
    planningUiEvents.length > 0
      ? planningUiEvents
      : planningStreamOpen && loading
        ? []
        : branchPlanningTranscript;
  const showPlanningActivityPanel =
    phase === "PLANNING" ||
    planningStreamOpen ||
    (phase === "DEVELOPMENT" && planningPanelEvents.length > 0);
  const mustUseDev = phase === "DEVELOPMENT" && mode === "main";
  const authTok = getToken();
  const pdfEmbedUrl =
    pdfArtifactId && authTok
      ? `${API_BASE.replace(/\/$/, "")}/v1/artifacts/${pdfArtifactId}/download?token=${encodeURIComponent(authTok)}`
      : null;

  return (
    <div className="flex h-full min-h-0 flex-col bg-black">
      <header className="flex shrink-0 items-center justify-between gap-2 pl-14 pr-3 py-3 md:pl-4 md:pr-4">
        <div className="min-w-0 flex-1 truncate text-[13px] font-medium text-[#707070]">
          {threadTitle || "Chat"}
        </div>
        <div className="relative flex shrink-0 items-center gap-1.5 sm:gap-2">
          <span className="hidden rounded-full border border-white/[0.08] px-2.5 py-1 text-[10px] font-semibold uppercase tracking-wider text-[#808080] sm:inline">
            {phase}
          </span>
          {phase === "PLANNING" && (
            <button
              type="button"
              disabled={loading}
              onClick={startPlanningJob}
              className="rounded-full border border-amber-500/25 bg-amber-500/[0.07] px-2.5 py-1.5 text-[10px] font-semibold text-amber-200/90 sm:px-3 sm:text-[11px]"
            >
              Plan job
            </button>
          )}
          {phase === "DEVELOPMENT" && (
            <div className="hidden rounded-full border border-white/[0.08] p-0.5 sm:flex">
              <button
                type="button"
                onClick={() => setMode("main")}
                className={`rounded-full px-2 py-1 text-[10px] font-semibold sm:px-3 sm:text-[11px] ${
                  mode === "main"
                    ? "bg-white/[0.1] text-white"
                    : "text-[#707070] hover:text-white"
                }`}
              >
                Project
              </button>
              <button
                type="button"
                onClick={() => setMode("dev")}
                className={`rounded-full px-2 py-1 text-[10px] font-semibold sm:px-3 sm:text-[11px] ${
                  mode === "dev"
                    ? "bg-white/[0.1] text-white"
                    : "text-[#707070] hover:text-white"
                }`}
              >
                Dev
              </button>
            </div>
          )}
          {phase === "DEVELOPMENT" && pdfEmbedUrl && (
            <button
              type="button"
              onClick={() => {
                setReportShowPdf(true);
                setReportOpen(true);
              }}
              className="hidden rounded-full border border-emerald-500/30 bg-emerald-500/10 px-3 py-2 text-[12px] font-semibold text-emerald-200/95 hover:bg-emerald-500/15 sm:inline sm:text-[13px]"
            >
              View PDF
            </button>
          )}
          <button
            type="button"
            onClick={createShareLink}
            className="rounded-full border border-white/[0.12] px-3 py-2 text-[12px] font-semibold text-white sm:text-[13px] hover:bg-white/[0.06]"
          >
            Share
          </button>
          <div className="relative">
            <button
              type="button"
              onClick={() => setHeaderMenu((v) => !v)}
              className="flex h-9 w-9 items-center justify-center rounded-full border border-white/[0.1] text-[#a0a0a0] hover:bg-white/[0.06]"
              aria-label="More"
            >
              ⋯
            </button>
            {headerMenu && (
              <>
                <button
                  type="button"
                  aria-label="Close menu"
                  className="fixed inset-0 z-40"
                  onClick={() => setHeaderMenu(false)}
                />
                <div className="absolute right-0 top-full z-50 mt-1 min-w-[10rem] rounded-2xl border border-white/[0.1] bg-[#141414] py-1 shadow-xl">
                  <button
                    type="button"
                    className="block w-full px-4 py-2.5 text-left text-[13px] text-white hover:bg-white/[0.06]"
                    onClick={() => {
                      setRenameDraft(threadTitle);
                      setRenameOpen(true);
                      setHeaderMenu(false);
                    }}
                  >
                    Rename
                  </button>
                  <button
                    type="button"
                    className="block w-full px-4 py-2.5 text-left text-[13px] text-white hover:bg-white/[0.06]"
                    onClick={() => {
                      openReport();
                      setHeaderMenu(false);
                    }}
                  >
                    Reports
                  </button>
                  <button
                    type="button"
                    className="block w-full px-4 py-2.5 text-left text-[13px] text-white hover:bg-white/[0.06]"
                    onClick={() => {
                      exportMd();
                      setHeaderMenu(false);
                    }}
                  >
                    Export MD
                  </button>
                  {phase === "DEVELOPMENT" && (
                    <>
                      <button
                        type="button"
                        className="block w-full px-4 py-2.5 text-left text-[13px] text-white hover:bg-white/[0.06] sm:hidden"
                        onClick={() => {
                          setMode("main");
                          setHeaderMenu(false);
                        }}
                      >
                        Mode: Project
                      </button>
                      <button
                        type="button"
                        className="block w-full px-4 py-2.5 text-left text-[13px] text-white hover:bg-white/[0.06] sm:hidden"
                        onClick={() => {
                          setMode("dev");
                          setHeaderMenu(false);
                        }}
                      >
                        Mode: Development
                      </button>
                    </>
                  )}
                  <button
                    type="button"
                    className="block w-full px-4 py-2.5 text-left text-[13px] text-red-400 hover:bg-red-500/10"
                    onClick={() => {
                      setDeleteOpen(true);
                      setHeaderMenu(false);
                    }}
                  >
                    Delete chat
                  </button>
                </div>
              </>
            )}
          </div>
        </div>
      </header>

      {shareToast && (
        <div className="pointer-events-none fixed left-1/2 top-20 z-[60] -translate-x-1/2 rounded-full border border-white/[0.12] bg-[#1a1a1a] px-4 py-2 text-[13px] text-white shadow-lg">
          {shareToast}
        </div>
      )}
      {copyToast && (
        <div className="pointer-events-none fixed left-1/2 top-28 z-[60] -translate-x-1/2 rounded-full border border-white/[0.12] bg-[#1a1a1a] px-4 py-2 text-[13px] text-white shadow-lg">
          {copyToast}
        </div>
      )}

      <div className="relative flex min-h-0 flex-1 flex-col">
        <div
          ref={scrollRef}
          className="min-h-0 flex-1 overflow-y-auto overscroll-contain"
        >
          <div className={`${CHAT_COLUMN} pb-6 pt-2`}>
            {messages.map((m, i) => (
              <article
                key={m.id ? m.id : `msg-${i}-${m.created_at || ""}`}
                className="mb-8 md:mb-8"
              >
                {m.role === "user" ? (
                  <div className="group/msg flex flex-col items-end">
                    {(() => {
                      const forkKey =
                        typeof m.metadata?.fork_anchor_message_id === "string"
                          ? m.metadata.fork_anchor_message_id
                          : m.id || "";
                      const fv = forkKey ? forkByMsg[forkKey] : undefined;
                      return (
                        <>
                          <div className="max-w-[80%] rounded-3xl rounded-br-md bg-[#1a1a1a] px-4 py-3 text-left border border-[#303030]">
                            <p className="whitespace-pre-wrap text-[15px] leading-[1.6] text-white">
                              {m.content}
                            </p>
                          </div>
                          <div className="mt-2 flex max-w-[80%] flex-wrap items-center justify-end gap-4 opacity-100 transition-opacity duration-200 sm:opacity-0 sm:group-hover/msg:opacity-100">
                            <button
                              type="button"
                              onClick={() => copyText(m.content)}
                              className="flex items-center gap-1.5 text-[12px] font-medium text-[#707070] hover:text-white"
                            >
                              <CopyIcon />
                              Copy
                            </button>
                            {mode === "main" && m.id && !mustUseDev && (
                              <button
                                type="button"
                                disabled={loading}
                                onClick={() => {
                                  setEditForkId(m.id!);
                                  setEditForkText(m.content);
                                }}
                                className="flex items-center gap-1.5 text-[12px] font-medium text-[#707070] hover:text-white disabled:opacity-40"
                              >
                                <PencilIcon />
                                Edit
                              </button>
                            )}
                          </div>
                          {fv &&
                            fv.total >= 2 &&
                            fv.active_in_family !== false && (
                              <div className="mt-2 flex max-w-[80%] items-center justify-end gap-3 text-[13px] text-[#a0a0a0]">
                                <button
                                  type="button"
                                  disabled={loading || fv.current_index <= 0}
                                  onClick={() => switchForkVersion(forkKey, -1)}
                                  className="rounded-lg p-1 hover:bg-white/[0.06] hover:text-white disabled:opacity-25"
                                  aria-label="Previous version"
                                >
                                  <ChevronLeftIcon />
                                </button>
                                <span className="min-w-[3rem] text-center font-medium tabular-nums text-white">
                                  {fv.current_index + 1} / {fv.total}
                                </span>
                                <button
                                  type="button"
                                  disabled={
                                    loading ||
                                    fv.current_index >= fv.total - 1
                                  }
                                  onClick={() => switchForkVersion(forkKey, 1)}
                                  className="rounded-lg p-1 hover:bg-white/[0.06] hover:text-white disabled:opacity-25"
                                  aria-label="Next version"
                                >
                                  <ChevronRightIcon />
                                </button>
                              </div>
                            )}
                        </>
                      );
                    })()}
                  </div>
                ) : (
                  <div className="group/asst w-full text-left">
                    {m.agent && (
                      <p className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-[#606060]">
                        {m.agent}
                      </p>
                    )}
                    <MessageMarkdown content={m.content || ""} />
                    {(() => {
                      if (phase === "REQUIREMENTS") return null;
                      const pe = m.metadata?.planning_ui_events;
                      if (!Array.isArray(pe) || pe.length === 0) return null;
                      const norm = normalizeUiEvents(pe);
                      return (
                        <PlanningEventsPanel
                          events={norm}
                          completed
                          onExpand={() => {
                            setExpandedPlanningEvents(norm);
                            setPlanningExpandedOpen(true);
                          }}
                        />
                      );
                    })()}
                    <div className="mt-3 flex flex-wrap items-center gap-4 opacity-100 transition-opacity duration-200 sm:opacity-0 sm:group-hover/asst:opacity-100">
                      {m.id &&
                        lastAssistant?.id === m.id &&
                        mode === "main" &&
                        !mustUseDev && (
                          <button
                            type="button"
                            disabled={loading}
                            onClick={() => regenerateForMessage(m.id!)}
                            className="flex items-center gap-1.5 text-[12px] font-medium text-[#707070] hover:text-white"
                          >
                            <RegenIcon />
                            Regenerate
                          </button>
                        )}
                      <button
                        type="button"
                        onClick={() => copyText(m.content)}
                        className="flex items-center gap-1.5 text-[12px] font-medium text-[#707070] hover:text-white"
                      >
                        <CopyIcon />
                        Copy
                      </button>
                    </div>
                  </div>
                )}
              </article>
            ))}

          {showSavedBranchPlanning && (
            <div className="mb-8">
              <p className="mb-3 text-[11px] font-semibold uppercase tracking-wider text-[#606060]">
                Planning session
              </p>
              <PlanningEventsPanel
                events={branchPlanningTranscript}
                completed
                onExpand={() => {
                  setExpandedPlanningEvents(branchPlanningTranscript);
                  setPlanningExpandedOpen(true);
                }}
              />
            </div>
          )}

          {loading && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="mb-6"
            >
              {planningStreamOpen && mode === "main" ? (
                busyHint && (
                  <p className="text-[13px] text-[#707070]">{busyHint}</p>
                )
              ) : (
                <div className="flex flex-wrap items-baseline gap-x-2 gap-y-1 text-[14px] text-[#909090]">
                  <motion.span
                    className="font-medium text-[#c4c4c4]"
                    animate={{ opacity: [0.35, 1, 0.35] }}
                    transition={{
                      duration: 1.6,
                      repeat: Infinity,
                      ease: "easeInOut",
                    }}
                  >
                    Thinking
                  </motion.span>
                  <span className="inline-flex translate-y-px gap-px">
                    {[0, 1, 2].map((i) => (
                      <motion.span
                        key={i}
                        className="text-[#888]"
                        animate={{ opacity: [0.15, 1, 0.15], y: [0, -3, 0] }}
                        transition={{
                          duration: 1,
                          repeat: Infinity,
                          delay: i * 0.12,
                          ease: "easeInOut",
                        }}
                      >
                        ·
                      </motion.span>
                    ))}
                  </span>
                  {busyHint && busyHint !== "Thinking…" && (
                    <span className="w-full text-[12px] text-[#606060]">
                      {busyHint}
                    </span>
                  )}
                </div>
              )}
            </motion.div>
          )}

          {showPlanningActivityPanel &&
            (planningPanelEvents.length > 0 ||
              (planningStreamOpen && loading && mode === "main")) && (
              <PlanningEventsPanel
                events={planningPanelEvents}
                showPlaceholder={
                  planningStreamOpen &&
                  loading &&
                  planningUiEvents.length === 0 &&
                  mode === "main"
                }
                completed={
                  planningPanelEvents.length > 0 &&
                  !planningStreamOpen &&
                  !loading
                }
                onExpand={
                  planningPanelEvents.length
                    ? () => {
                        setExpandedPlanningEvents(planningPanelEvents);
                        setPlanningExpandedOpen(true);
                      }
                    : undefined
                }
              />
            )}

          {quick.length > 0 && mode === "main" && !mustUseDev && (
            <div className="mb-6 flex flex-col gap-0.5">
              {quick.map((q) => (
                <button
                  key={q.id}
                  type="button"
                  disabled={loading}
                  onClick={() => send(q.value)}
                  className="group/qr flex w-fit max-w-full items-start gap-3 rounded-xl px-1 py-2.5 text-left transition-colors hover:bg-[#141414] hover:text-white disabled:cursor-not-allowed disabled:opacity-40"
                >
                  <span
                    className="mt-[3px] shrink-0 font-sans text-[15px] leading-none text-[#5c5c5c] transition-colors group-hover/qr:text-[#8a8a8a]"
                    aria-hidden
                  >
                    ↳
                  </span>
                  <span className="min-w-0 flex-1 text-[15px] font-medium leading-snug text-white/95 group-hover/qr:text-white">
                    {q.label}
                  </span>
                </button>
              ))}
            </div>
          )}

          <div ref={bottomRef} className="h-px w-full shrink-0" aria-hidden />
          </div>
        </div>

        {showScrollDown && hasMessages && (
          <button
            type="button"
            aria-label="Scroll to bottom"
            onClick={scrollDown}
            className="fixed bottom-[calc(6.5rem+env(safe-area-inset-bottom))] right-4 z-30 flex h-10 w-10 items-center justify-center rounded-full border border-white/[0.12] bg-[#1a1a1a] text-white shadow-lg md:bottom-[calc(5.5rem+env(safe-area-inset-bottom))] md:right-8"
          >
            ↓
          </button>
        )}

        <div className="z-20 shrink-0 bg-black px-3 pb-[max(0.75rem,env(safe-area-inset-bottom))]">
          <div className={CHAT_COLUMN}>
            <div>
              {mustUseDev ? (
                <div className="rounded-2xl border border-amber-500/25 bg-gradient-to-b from-amber-950/30 to-black/40 p-4">
                  <p className="text-[15px] font-semibold text-white">
                    Planning finished — switch to Implementation
                  </p>
                  <p className="mt-1.5 text-[13px] leading-relaxed text-[#a0a0a0]">
                    Project chat is paused so you review the validated
                    architecture PDF first. Open the report, then continue in
                    Development for build questions.
                  </p>
                  <div className="mt-4 flex flex-wrap items-center gap-2">
                    {pdfEmbedUrl && (
                      <>
                        <button
                          type="button"
                          onClick={() => {
                            setReportShowPdf(true);
                            setReportOpen(true);
                          }}
                          className="rounded-full border border-emerald-500/35 bg-emerald-500/15 px-4 py-2.5 text-[13px] font-semibold text-emerald-100 hover:bg-emerald-500/25"
                        >
                          View PDF
                        </button>
                        <a
                          href={pdfEmbedUrl}
                          download
                          target="_blank"
                          rel="noreferrer"
                          className="rounded-full border border-white/[0.12] px-4 py-2.5 text-[13px] font-semibold text-white hover:bg-white/[0.06]"
                        >
                          Download PDF
                        </a>
                      </>
                    )}
                    <button
                      type="button"
                      onClick={() => setMode("dev")}
                      className="rounded-full bg-white px-5 py-2.5 text-[13px] font-semibold text-black hover:bg-[#e8e8e8]"
                    >
                      Continue in Implementation →
                    </button>
                  </div>
                </div>
              ) : (
                <ChatPillInput
                  variant="dock"
                  value={input}
                  onChange={setInput}
                  disabled={loading || !activeBranchId}
                  placeholder={
                    mode === "dev"
                      ? "Ask implementation questions..."
                      : "Ask anything"
                  }
                  onSubmit={() => {
                    send(input);
                    setInput("");
                  }}
                />
              )}
            </div>
          </div>
        </div>
      </div>

      {planningExpandedOpen && (
        <div className="fixed inset-0 z-[75] flex items-center justify-center bg-black/85 p-4">
          <div className="flex max-h-[92vh] w-full max-w-4xl flex-col overflow-hidden rounded-3xl border border-emerald-500/25 bg-[#0a0a0a] shadow-2xl">
            <div className="flex items-center justify-between border-b border-white/[0.08] px-4 py-3">
              <p className="text-[14px] font-semibold text-white">
                Planning — expanded view
              </p>
              <button
                type="button"
                onClick={() => {
                  setPlanningExpandedOpen(false);
                  setExpandedPlanningEvents(null);
                }}
                className="rounded-full px-3 py-1.5 text-[13px] text-[#a0a0a0] hover:bg-white/[0.06] hover:text-white"
              >
                Close
              </button>
            </div>
            <div className=" flex-1 overflow-y-auto p-4">
              <PlanningEventsPanel
                events={
                  expandedPlanningEvents?.length
                    ? expandedPlanningEvents
                    : planningUiEvents
                }
                showPlaceholder={false}
                compactMaxHeight={false}
              />
            </div>
          </div>
        </div>
      )}

      {renameOpen && (
        <div className="fixed inset-0 z-[70] flex items-center justify-center bg-black/75 p-4">
          <div className="w-full max-w-md rounded-3xl border border-white/[0.1] bg-[#111] p-5">
            <h2 className="text-[15px] font-semibold">Rename chat</h2>
            <input
              value={renameDraft}
              onChange={(e) => setRenameDraft(e.target.value)}
              className="mt-3 w-full rounded-2xl border border-white/[0.1] bg-black px-3 py-2.5 text-[15px] text-white"
            />
            <div className="mt-4 flex justify-end gap-2">
              <button
                type="button"
                onClick={() => setRenameOpen(false)}
                className="rounded-full px-4 py-2 text-[13px] text-[#909090]"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={saveRename}
                className="rounded-full bg-white px-5 py-2 text-[13px] font-semibold text-black"
              >
                Save
              </button>
            </div>
          </div>
        </div>
      )}

      {deleteOpen && (
        <div className="fixed inset-0 z-[70] flex items-center justify-center bg-black/75 p-4">
          <div className="w-full max-w-md rounded-3xl border border-white/[0.1] bg-[#111] p-5">
            <h2 className="text-[15px] font-semibold">Delete this chat?</h2>
            <p className="mt-2 text-[13px] text-[#909090]">{threadTitle}</p>
            <div className="mt-4 flex justify-end gap-2">
              <button
                type="button"
                onClick={() => setDeleteOpen(false)}
                className="rounded-full px-4 py-2 text-[13px] text-[#909090]"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={confirmDelete}
                className="rounded-full bg-red-600 px-5 py-2 text-[13px] font-semibold text-white"
              >
                Delete
              </button>
            </div>
          </div>
        </div>
      )}

      {editForkId && (
        <div className="fixed inset-0 z-50 flex items-end justify-center bg-black/75 p-4 sm:items-center">
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            className="w-full max-w-lg rounded-3xl border border-white/[0.1] bg-[#111] p-5 shadow-2xl"
          >
            <h2 className="text-[15px] font-semibold text-white">
              Edit message
            </h2>
            <p className="mt-1 text-[12px] text-[#707070]">
              Creates a new branch from this point.
            </p>
            <textarea
              value={editForkText}
              onChange={(e) => setEditForkText(e.target.value)}
              rows={5}
              className="mt-4 w-full resize-none rounded-2xl border border-white/[0.08] bg-black px-4 py-3 text-[15px] text-white placeholder:text-[#505050] focus:border-white/[0.16] focus:outline-none"
            />
            <div className="mt-4 flex justify-end gap-2">
              <button
                type="button"
                onClick={() => {
                  setEditForkId(null);
                  setEditForkText("");
                }}
                className="rounded-full px-4 py-2 text-[13px] font-medium text-[#909090] transition-colors hover:bg-white/[0.06] hover:text-white"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={submitEditFork}
                className="rounded-full bg-white px-5 py-2 text-[13px] font-semibold text-black transition-colors hover:bg-[#e8e8e8]"
              >
                Save &amp; fork
              </button>
            </div>
          </motion.div>
        </div>
      )}

      {reportOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-4">
          <motion.div
            initial={{ opacity: 0, scale: 0.98 }}
            animate={{ opacity: 1, scale: 1 }}
            className="flex max-h-[90vh] w-full max-w-5xl flex-col overflow-hidden rounded-3xl border border-white/[0.1] bg-[#0a0a0a] p-5"
          >
            <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
              <h2 className="text-[15px] font-semibold text-white">
                Validated architecture report
              </h2>
              <div className="flex flex-wrap items-center gap-2">
                {pdfEmbedUrl && reportShowPdf && (
                  <button
                    type="button"
                    onClick={async () => {
                      try {
                        const data = await api<unknown>(
                          `/v1/threads/${threadId}/report`
                        );
                        setReportJson(data);
                        setReportShowPdf(false);
                      } catch {
                        setReportJson(null);
                        setReportShowPdf(false);
                      }
                    }}
                    className="rounded-full px-3 py-1.5 text-[12px] text-[#a0a0a0] hover:bg-white/[0.06] hover:text-white"
                  >
                    Raw JSON
                  </button>
                )}
                {!reportShowPdf && pdfEmbedUrl && (
                  <button
                    type="button"
                    onClick={() => setReportShowPdf(true)}
                    className="rounded-full px-3 py-1.5 text-[12px] text-[#a0a0a0] hover:bg-white/[0.06] hover:text-white"
                  >
                    PDF view
                  </button>
                )}
                <button
                  type="button"
                  onClick={() => setReportOpen(false)}
                  className="rounded-full px-3 py-1 text-[13px] text-[#808080] hover:bg-white/[0.06] hover:text-white"
                >
                  Close
                </button>
              </div>
            </div>
            {reportShowPdf && pdfEmbedUrl ? (
              <div className="flex min-h-0 flex-1 flex-col gap-3">
                <iframe
                  title="Architecture PDF"
                  src={pdfEmbedUrl}
                  className="min-h-[70vh] w-full flex-1 rounded-2xl border border-white/[0.08] bg-black"
                />
                <a
                  href={pdfEmbedUrl}
                  download
                  target="_blank"
                  rel="noreferrer"
                  className="inline-flex w-fit rounded-full border border-white/[0.12] px-4 py-2 text-[13px] font-semibold text-white hover:bg-white/[0.06]"
                >
                  Download PDF
                </a>
              </div>
            ) : (
              <pre className="max-h-[75vh] overflow-auto whitespace-pre-wrap wrap-break-word text-[12px] text-[#a0a0a0]">
                {reportJson
                  ? JSON.stringify(reportJson, null, 2).slice(0, 80000)
                  : "No structured report JSON loaded. If planning just finished, use PDF view or refresh."}
              </pre>
            )}
          </motion.div>
        </div>
      )}
    </div>
  );
}

function CopyIcon() {
  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
    >
      <rect x="9" y="9" width="13" height="13" rx="2" />
      <path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1" />
    </svg>
  );
}

function PencilIcon() {
  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
    >
      <path d="M12 20h9M16.5 3.5a2.12 2.12 0 013 3L7 19l-4 1 1-4L16.5 3.5z" />
    </svg>
  );
}

function RegenIcon() {
  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
    >
      <path d="M23 4v6h-6M1 20v-6h6" />
      <path d="M3.51 9a9 9 0 0114.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0020.49 15" />
    </svg>
  );
}

function ChevronLeftIcon() {
  return (
    <svg
      width="18"
      height="18"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
    >
      <path d="M15 18l-6-6 6-6" strokeLinecap="round" />
    </svg>
  );
}

function ChevronRightIcon() {
  return (
    <svg
      width="18"
      height="18"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
    >
      <path d="M9 18l6-6-6-6" strokeLinecap="round" />
    </svg>
  );
}
