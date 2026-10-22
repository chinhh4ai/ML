import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { marked } from "marked";
import {
  ArrowLeft,
  ArrowRight,
  ArrowUpRight,
  Check,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  CircleHelp,
  Clock3,
  Cpu,
  Film,
  Layers3,
  LoaderCircle,
  MessageSquare,
  Play,
  Plus,
  RefreshCw,
  ScanLine,
  Search,
  Send,
  Settings2,
  ShieldCheck,
  Sparkles,
  UploadCloud,
  Video,
  X,
} from "lucide-react";
import "./style.css";

function MarkdownView({ content, className = "summary-markdown" }: { content: string; className?: string }) {
  const html = useMemo(() => {
    if (!content) return "";
    try {
      return marked.parse(content, { breaks: true, gfm: true }) as string;
    } catch {
      return content;
    }
  }, [content]);

  return <div className={className} dangerouslySetInnerHTML={{ __html: html }} />;
}

type Obj = {
  label: string;
  confidence: number;
  box: number[];
  polygons?: number[][][];
  track_id?: number;
  debris_candidate?: boolean;
};
type Segment = {
  id: number;
  start: number;
  end: number;
  labels: string[];
  description: string;
  tags: string[];
  ai: boolean;
  thumbnail: string;
};
type Item = {
  id: string;
  name: string;
  status: string;
  stage: string;
  progress: number;
  duration: number;
  created_at: string;
  width: number;
  height: number;
  size: number;
  summary?: string;
  warning?: string;
  error?: string;
  object_counts?: Record<string, number>;
  segments?: Segment[];
  observations?: { time: number; objects: Obj[] }[];
  processed_frames?: number;
  elapsed_seconds?: number;
  sample_fps?: number;
  detection_model?: string;
  detection_device?: string;
  track_count?: number;
};
type Health = {
  device?: string;
  provider: string;
  model: string;
  ai_configured: boolean;
  detection_ready: boolean;
  max_upload_mb: number;
  max_duration_seconds: number;
  sample_fps: number;
  detection_model: string;
  segmentation: boolean;
  tracking: string;
};
type ProviderConfig = {
  model: string;
  has_key: boolean;
  key_masked: string;
  base_url: string;
  preset_models: string[];
};
type DetectionModelOption = {
  id: string;
  name: string;
  classes: number | string;
  type?: string;
};
type FullConfig = {
  provider: string;
  model: string;
  base_url: string;
  has_key: boolean;
  detection_model?: string;
  detection_confidence?: number;
  available_detection_models?: DetectionModelOption[];
  providers: Record<string, ProviderConfig>;
};
type Message = {
  role: "user" | "assistant";
  text: string;
  sources?: Segment[];
  scope?: string;
};
const time = (n = 0) =>
  `${Math.floor(n / 60)
    .toString()
    .padStart(2, "0")}:${Math.floor(n % 60)
    .toString()
    .padStart(2, "0")}`;
const labels: Record<string, string> = {
  Plastic: "Plastic Debris",
  Metal: "Metal / Can Debris",
  Glass: "Glass Container",
  Paper: "Paper / Cardboard",
  Other: "Other Floating Debris",
  plastic: "Plastic Debris",
  metal: "Metal / Can Debris",
  glass: "Glass Container",
  paper: "Paper / Cardboard",
  other: "Other Floating Debris",
  person: "Person",
  car: "Car",
  motorcycle: "Motorcycle",
  bicycle: "Bicycle",
  truck: "Truck",
  bus: "Bus",
  dog: "Dog",
  cat: "Cat",
  chair: "Chair",
  bottle: "Bottle",
  laptop: "Laptop",
  book: "Book",
  "cell phone": "Cell Phone",
  handbag: "Handbag",
  backpack: "Backpack",
  bird: "Bird",
  boat: "Boat",
  tv: "TV",
  cup: "Cup",
};
const label = (s: string) => labels[s] || s;
async function api<T>(path: string, opts?: RequestInit): Promise<T> {
  const r = await fetch("/api" + path, opts);
  const data = await r.json();
  if (!r.ok)
    throw new Error(
      typeof data.detail === "string"
        ? data.detail
        : "Request failed. Please try again.",
    );
  return data;
}
function App() {
  const [items, setItems] = useState<Item[]>([]),
    [selected, setSelected] = useState<string | null>(null),
    [detail, setDetail] = useState<Item | null>(null);
  const [health, setHealth] = useState<Health | null>(null),
    [error, setError] = useState(""),
    [settings, setSettings] = useState(false),
    [help, setHelp] = useState(false);
  const [uploading, setUploading] = useState(false),
    [uploadProgress, setUploadProgress] = useState(0),
    [drag, setDrag] = useState(false);
  const [filter, setFilter] = useState(""),
    [query, setQuery] = useState(""),
    [results, setResults] = useState<Segment[] | null>(null),
    [searching, setSearching] = useState(false);
  const [tab, setTab] = useState("summary"),
    [overlay, setOverlay] = useState(true),
    [masks, setMasks] = useState(true),
    [trails, setTrails] = useState(true),
    [current, setCurrent] = useState(0),
    [objectFilter, setObjectFilter] = useState("all");
  const [question, setQuestion] = useState(""),
    [messages, setMessages] = useState<Message[]>([]),
    [asking, setAsking] = useState(false);
  const [testingAi, setTestingAi] = useState(false);
  const [aiTestResult, setAiTestResult] = useState<{ ok: boolean; msg: string } | null>(null);
  const [fullConfig, setFullConfig] = useState<FullConfig | null>(null);
  const [cfgProvider, setCfgProvider] = useState("gemini");
  const [cfgModel, setCfgModel] = useState("gemini-3.6-flash");
  const [cfgKey, setCfgKey] = useState("");
  const [cfgBaseUrl, setCfgBaseUrl] = useState("");
  const [cfgDetectionModel, setCfgDetectionModel] = useState("yolo11_litter_seg");
  const [cfgDetectionConfidence, setCfgDetectionConfidence] = useState(0.15);
  const [expandedSegmentId, setExpandedSegmentId] = useState<number | null>(null);
  const [savingConfig, setSavingConfig] = useState(false);
  const [saveResult, setSaveResult] = useState<{ ok: boolean; msg: string } | null>(null);

  const input = useRef<HTMLInputElement>(null),
    player = useRef<HTMLVideoElement>(null),
    chatBottom = useRef<HTMLDivElement>(null);
  const activeId = useRef<string | null>(null);
  activeId.current = selected;

  const load = async () => {
    try {
      const [list, h] = await Promise.all([
        api<Item[]>("/videos"),
        api<Health>("/health"),
      ]);
      setItems(list);
      setHealth(h);
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const openSettings = async () => {
    setSettings(true);
    setSaveResult(null);
    setAiTestResult(null);
    try {
      const cfg = await api<FullConfig>("/config");
      setFullConfig(cfg);
      setCfgProvider(cfg.provider);
      setCfgModel(cfg.model);
      setCfgBaseUrl(cfg.providers[cfg.provider]?.base_url || "");
      setCfgDetectionModel(cfg.detection_model || "yolo11_litter_seg");
      setCfgDetectionConfidence(cfg.detection_confidence ?? 0.15);
      setCfgKey("");
    } catch {
      if (health) {
        setCfgProvider(health.provider);
        setCfgModel(health.model);
        setCfgDetectionModel(health.detection_model || "yolo11_litter_seg");
      }
    }
  };

  const handleProviderSelect = (p: string) => {
    setCfgProvider(p);
    setSaveResult(null);
    setAiTestResult(null);
    if (fullConfig?.providers[p]) {
      setCfgModel(fullConfig.providers[p].model || fullConfig.providers[p].preset_models[0] || "");
      setCfgBaseUrl(fullConfig.providers[p].base_url || "");
    }
  };

  const handleSaveConfig = async (e: React.FormEvent) => {
    e.preventDefault();
    setSavingConfig(true);
    setSaveResult(null);
    setAiTestResult(null);
    try {
      const res = await api<{ status: string; config: FullConfig }>("/config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          provider: cfgProvider,
          model: cfgModel,
          key: cfgKey.trim() || undefined,
          base_url: cfgBaseUrl.trim() || undefined,
          detection_model: cfgDetectionModel,
          detection_confidence: Number(cfgDetectionConfidence),
        }),
      });
      setFullConfig(res.config);
      setCfgKey("");
      setSaveResult({ ok: true, msg: "Configuration saved and active!" });
      await load();
    } catch (err) {
      setSaveResult({ ok: false, msg: (err as Error).message });
    } finally {
      setSavingConfig(false);
    }
  };

  useEffect(() => {
    void load();
    const timer = setInterval(() => void load(), 4000);
    return () => clearInterval(timer);
  }, []);
  useEffect(() => {
    setDetail(null);
    setMessages([]);
    setResults(null);
    setQuery("");
    setCurrent(0);
    setTab("summary");
    setObjectFilter("all");
    setQuestion("");
    setAsking(false);
    if (!selected) return;
    let live = true;
    const poll = () =>
      api<Item>("/videos/" + selected)
        .then((d) => {
          if (live) setDetail(d);
        })
        .catch((e) => {
          if (live) setError(e.message);
        });
    void poll();
    const timer = setInterval(() => void poll(), 2500);
    return () => {
      live = false;
      clearInterval(timer);
    };
  }, [selected]);
  useEffect(() => {
    chatBottom.current?.scrollIntoView({
      behavior: "smooth",
      block: "nearest",
    });
  }, [messages, asking]);
  const upload = async (file?: File) => {
    if (!file || uploading) return;
    if (health && file.size > health.max_upload_mb * 1024 * 1024) {
      setError(`Max file size is ${health.max_upload_mb} MB.`);
      return;
    }
    setUploading(true);
    setUploadProgress(0);
    setError("");
    try {
      const row = await new Promise<Item>((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        xhr.open("POST", "/api/videos");
        xhr.timeout = 120000;
        xhr.upload.onprogress = (e) => {
          if (e.lengthComputable)
            setUploadProgress(Math.round((e.loaded / e.total) * 100));
        };
        xhr.onerror = () => reject(new Error("Connection lost during video upload."));
        xhr.ontimeout = () => reject(new Error("Video upload timed out."));
        xhr.onload = () => {
          try {
            const data = JSON.parse(xhr.responseText);
            xhr.status < 300
              ? resolve(data)
              : reject(new Error(data.detail || "Failed to upload video."));
          } catch {
            reject(new Error("Invalid upload response."));
          }
        };
        const form = new FormData();
        form.append("file", file);
        xhr.send(form);
      });
      await load();
      setSelected(row.id);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setUploading(false);
      if (input.current) input.current.value = "";
    }
  };
  const seek = (s: number) => {
    if (player.current) {
      player.current.currentTime = s;
      void player.current.play().catch(() => {});
      setCurrent(s);
    }
  };
  const search = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selected) return;
    const id = selected;
    setSearching(true);
    try {
      const d = await api<{ results: Segment[] }>(
        `/videos/${id}/search?q=${encodeURIComponent(query)}`,
      );
      if (activeId.current === id) setResults(d.results);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSearching(false);
    }
  };
  const ask = async (text = question) => {
    if (!selected || !text.trim() || asking) return;
    const id = selected;
    setQuestion("");
    setMessages((m) => [...m, { role: "user", text }]);
    setAsking(true);
    try {
      const data = await api<{
        answer: string;
        sources: Segment[];
        scope: string;
      }>(`/videos/${id}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: text }),
      });
      if (activeId.current === id)
        setMessages((m) => [
          ...m,
          {
            role: "assistant",
            text: data.answer,
            sources: data.sources,
            scope: data.scope,
          },
        ]);
    } catch (e) {
      if (activeId.current === id)
        setMessages((m) => [
          ...m,
          { role: "assistant", text: (e as Error).message },
        ]);
    } finally {
      if (activeId.current === id) setAsking(false);
    }
  };
  const retry = async () => {
    if (!selected) return;
    try {
      const d = await api<Item>(`/videos/${selected}/retry`, {
        method: "POST",
      });
      setDetail((old) => (old ? { ...old, ...d } : d));
      await load();
    } catch (e) {
      setError((e as Error).message);
    }
  };
  const ready = detail?.status === "ready";
  const observation = detail?.observations?.reduce<
    { time: number; objects: Obj[] } | undefined
  >(
    (best, o) =>
      Math.abs(o.time - current) < Math.abs((best?.time ?? Infinity) - current)
        ? o
        : best,
    undefined,
  );
  const boxes =
    observation &&
    Math.abs(observation.time - current) <= 1 / (detail?.sample_fps || 1)
      ? observation.objects.filter(
          (o) => objectFilter === "all" || o.label === objectFilter,
        )
      : [];
  const segments = results ?? detail?.segments ?? [];
  const filtered = items.filter((i) =>
    i.name.toLowerCase().includes(filter.toLowerCase()),
  );
  return (
    <div className="app">
      <input
        ref={input}
        type="file"
        accept="video/*,.mkv"
        hidden
        onChange={(e) => void upload(e.target.files?.[0])}
      />
      <aside className="sidebar">
        <button className="brand" onClick={() => setSelected(null)}>
          <span className="brand-mark">
            <ScanLine size={24} />
          </span>
          riverlens<span className="brand-dot">.</span>
        </button>
        <div className="workspace-label">WORKSPACE</div>
        <button
          className={"nav-item " + (!settings ? "active" : "")}
          onClick={() => {
            setSelected(null);
            setSettings(false);
          }}
        >
          <Layers3 size={19} />
          Video Library<span className="nav-count">{items.length}</span>
        </button>
        <button className="nav-item" onClick={() => void openSettings()}>
          <Settings2 size={19} />
          System Settings
          <ArrowUpRight size={15} />
        </button>
        <div className="side-note">
          <div className="side-note-icon">
            <Cpu size={21} />
          </div>
          <strong>Smart & Lightweight</strong>
          <p>
            Track aquatic debris on CPU.
            <br />
            Connect AI on your terms.
          </p>
          <div className="cpu-pill">
            <span />
            CPU ONLY
          </div>
        </div>
        <div className="side-bottom">
          <button onClick={() => setHelp(true)}>
            <CircleHelp size={17} />
            User Guide
            <ArrowUpRight size={14} />
          </button>
          <div className="profile">
            <div className="avatar">L</div>
            <div>
              <strong>Local workspace</strong>
              <small>Development environment</small>
            </div>
            <span className="online-dot" />
          </div>
        </div>
      </aside>
      <div className="main-shell">
        <header>
          <div className="breadcrumb">
            <span>Workspace</span>
            <ChevronRight size={14} />
            <button onClick={() => setSelected(null)}>Video Library</button>
            {selected && (
              <>
                <ChevronRight size={14} />
                <span className="breadcrumb-video">Analysis</span>
              </>
            )}
          </div>
          <div className="header-right">
            <span className={"connection " + (health ? "" : "offline")}>
              <i />
              {health ? "System Ready" : "Connecting..."}
            </span>
            <span className="header-divider" />
            <span className="version">MVP / 0.1</span>
          </div>
        </header>
        <main>
          {error && (
            <div className="alert error" role="alert">
              {error}
              <button aria-label="Dismiss alert" onClick={() => setError("")}>
                <X size={16} />
              </button>
            </div>
          )}
          {!selected ? (
            <>
              <div className="page-heading">
                <div>
                  <div className="eyebrow">
                    <span /> AQUATIC DEBRIS INTELLIGENCE
                  </div>
                  <h1>
                    Clear insight into every waterway<span>.</span>
                  </h1>
                  <p>
                    Monitor floating debris across rivers, lakes, and waterways. Detect, track, and analyze with AI.
                  </p>
                </div>
                <button
                  className="primary"
                  disabled={uploading}
                  onClick={() => input.current?.click()}
                >
                  <Plus size={18} />
                  Add Video
                </button>
              </div>
              <section className="stats">
                <div>
                  <span className="stat-icon">
                    <Video size={21} />
                  </span>
                  <div>
                    <small>Videos in library</small>
                    <strong>
                      {items.length}
                      <span>videos</span>
                    </strong>
                  </div>
                </div>
                <div>
                  <span className="stat-icon">
                    <CheckCircle2 size={21} />
                  </span>
                  <div>
                    <small>Analyzed</small>
                    <strong>
                      {items.filter((i) => i.status === "ready").length}
                      <span>completed</span>
                    </strong>
                  </div>
                </div>
                <div>
                  <span className="stat-icon">
                    <Clock3 size={21} />
                  </span>
                  <div>
                    <small>Total duration</small>
                    <strong>
                      {time(items.reduce((s, i) => s + i.duration, 0))}
                      <span>min : sec</span>
                    </strong>
                  </div>
                </div>
                <div>
                  <span className="stat-icon">
                    <Sparkles size={21} />
                  </span>
                  <div>
                    <small>AI Provider</small>
                    <strong className="provider-name">
                      {health?.provider || "—"}
                      <span
                        className={
                          health?.ai_configured ? "configured" : "pending"
                        }
                      >
                        {health?.ai_configured ? "Configured" : "No API key"}
                      </span>
                    </strong>
                  </div>
                </div>
              </section>
              <section
                className={"drop-zone " + (drag ? "dragging" : "")}
                onDragOver={(e) => {
                  e.preventDefault();
                  setDrag(true);
                }}
                onDragLeave={() => setDrag(false)}
                onDrop={(e) => {
                  e.preventDefault();
                  setDrag(false);
                  void upload(e.dataTransfer.files[0]);
                }}
              >
                <div className="upload-art">
                  <div />
                  <div />
                  <span>
                    {uploading ? (
                      <LoaderCircle className="spin" size={27} />
                    ) : (
                      <UploadCloud size={29} />
                    )}
                  </span>
                </div>
                <div className="drop-copy">
                  <h2>
                    {uploading
                      ? `Uploading video · ${uploadProgress}%`
                      : "Start with a waterway view"}
                  </h2>
                  <p>
                    {uploading
                      ? "Keep this window open while video uploads."
                      : "Drag and drop video here, or select a file from your computer."}
                  </p>
                  <span>
                    MP4, MOV, WebM · Up to {health?.max_upload_mb ?? 256} MB ·{" "}
                    {Math.round((health?.max_duration_seconds ?? 1800) / 60)}{" "}
                    min
                  </span>
                </div>
                <button
                  className="secondary"
                  onClick={() => input.current?.click()}
                  disabled={uploading}
                >
                  Select Video
                  <ArrowUpRight size={16} />
                </button>
                {uploading && (
                  <div
                    className="upload-bar"
                    style={{ width: uploadProgress + "%" }}
                  />
                )}
              </section>
              <div className="section-heading">
                <div>
                  <h2>
                    Your Library <span>{items.length}</span>
                  </h2>
                  <p>All perspectives, in a single unified workspace.</p>
                </div>
                <div className="library-search">
                  <Search size={17} />
                  <input
                    aria-label="Search video library"
                    placeholder="Search by video name..."
                    value={filter}
                    onChange={(e) => setFilter(e.target.value)}
                  />
                </div>
              </div>
              {filtered.length ? (
                <section className="video-grid">
                  {filtered.map((item) => (
                    <button
                      className="video-card"
                      key={item.id}
                      onClick={() => setSelected(item.id)}
                    >
                      <div className="card-image">
                        {item.status === "ready" ? (
                          <img
                            src={`/api/videos/${item.id}/frames/0-0.jpg`}
                            alt=""
                          />
                        ) : (
                          <div className="card-placeholder">
                            <Film size={38} />
                          </div>
                        )}
                        <span className="duration">{time(item.duration)}</span>
                        <span className={"status-badge " + item.status}>
                          {item.status === "ready" ? (
                            <Check size={11} />
                          ) : item.status === "failed" ? (
                            <X size={11} />
                          ) : (
                            <LoaderCircle size={11} className="spin" />
                          )}
                          {item.status === "ready"
                            ? "Analyzed"
                            : item.status === "failed"
                              ? "Failed"
                              : item.status === "queued"
                                ? "Queued"
                                : "Processing"}
                        </span>
                        <span className="card-play">
                          <Play size={18} fill="currentColor" />
                        </span>
                      </div>
                      <div className="card-body">
                        <h3>{item.name}</h3>
                        <div className="card-meta">
                          <span>
                            {new Date(item.created_at).toLocaleDateString(
                              "en-US",
                            )}{" "}
                            · {(item.size / 1024 / 1024).toFixed(1)} MB
                          </span>
                          <ArrowUpRight size={16} />
                        </div>
                        {["queued", "processing"].includes(item.status) && (
                          <div className="mini-progress">
                            <i style={{ width: item.progress + "%" }} />
                          </div>
                        )}
                      </div>
                    </button>
                  ))}
                </section>
              ) : (
                <section className="empty-library">
                  <div className="empty-graphic">
                    <span />
                    <Film size={30} />
                    <span />
                  </div>
                  <h3>
                    {filter
                      ? "No videos found"
                      : "A library full of potential"}
                  </h3>
                  <p>
                    {filter
                      ? "Try searching with a different name."
                      : "Add your first video to discover what might have been missed."}
                  </p>
                  {!filter && (
                    <div className="feature-chips">
                      <span>
                        <ScanLine size={14} />
                        Segmentation & tracking
                      </span>
                      <span>
                        <Sparkles size={14} />
                        Content Summarization
                      </span>
                      <span>
                        <MessageSquare size={14} />
                        Q&A with AI
                      </span>
                    </div>
                  )}
                </section>
              )}
              <div className="privacy-footer">
                <ShieldCheck size={15} />
                <span>
                  Videos stored locally. Sampled frames sent to your chosen AI provider only when enabled.
                </span>
                <span className="footer-badge">BUILT FOR CURIOSITY</span>
              </div>
            </>
          ) : !detail ? (
            <div className="loading">
              <LoaderCircle className="spin" />
              Opening video...
            </div>
          ) : (
            <>
              <button className="back-link" onClick={() => setSelected(null)}>
                <ArrowLeft size={16} />
                Back to Library
              </button>
              <div className="detail-heading">
                <div>
                  <h1>{detail.name}</h1>
                  <p>
                    {time(detail.duration)} · {detail.width} × {detail.height} ·{" "}
                    {(detail.size / 1024 / 1024).toFixed(1)} MB{" "}
                    <span className="dot-separator">•</span>{" "}
                    {detail.detection_model || "YOLO11n-seg"} {detail.detection_device === "cloud-vlm" ? "· Cloud VLM" : "· CPU"}
                  </p>
                </div>
                <button
                  className="secondary"
                  disabled={!["ready", "failed"].includes(detail.status)}
                  onClick={() => void retry()}
                >
                  <RefreshCw size={15} />
                  Re-analyze
                </button>
              </div>
              {!ready && detail.status !== "failed" && (
                <div className="processing-banner">
                  <LoaderCircle className="spin" size={20} />
                  <div>
                    <strong>{detail.stage}</strong>
                    <p>CPU worker is processing. You can return to library in the meantime.</p>
                  </div>
                  <div className="processing-progress">
                    <span>{detail.progress}%</span>
                    <div>
                      <i style={{ width: detail.progress + "%" }} />
                    </div>
                  </div>
                </div>
              )}
              {detail.status === "failed" && (
                <div className="alert error" role="alert">
                  {detail.error} · Click “Re-analyze” to retry.
                </div>
              )}
              {detail.warning && (
                <div className="alert warning">
                  <CircleHelp size={17} />
                  <span>{detail.warning}</span>
                  <button onClick={() => void openSettings()}>
                    Settings
                    <ArrowUpRight size={14} />
                  </button>
                </div>
              )}
              <div className="domain-notice">
                <ShieldCheck size={15} />
                <span>
                  Aquatic Debris Monitoring Mode · Orange masks indicate candidate debris objects; does not confirm surface status or trash nature. COCO pretrained models are not specialized for waterways.
                </span>
              </div>
              <div className="analysis-layout">
                <div className="analysis-left">
                  <section className="player-panel">
                    <div className="panel-heading">
                      <h2>
                        <Play size={16} />
                        Video Player
                      </h2>
                      <span className="cpu-tag">
                        <Cpu size={13} />
                        CPU inference
                      </span>
                    </div>
                    <div className="player-area">
                      {ready ? (
                        <div
                          className="video-stage"
                          style={{
                            aspectRatio: `${detail.width}/${detail.height}`,
                          }}
                        >
                          <video
                            ref={player}
                            src={`/api/videos/${detail.id}/media`}
                            controls
                            onTimeUpdate={(e) =>
                              setCurrent(e.currentTarget.currentTime)
                            }
                            preload="metadata"
                          />
                          <svg
                            className="box-overlay"
                            viewBox="0 0 1000 1000"
                            preserveAspectRatio="none"
                            aria-hidden="true"
                          >
                            {trails &&
                              Object.entries(
                                (detail.observations || [])
                                  .filter(
                                    (o) =>
                                      o.time <= current &&
                                      o.time >= current - 4,
                                  )
                                  .reduce<Record<string, string[]>>(
                                    (acc, o) => {
                                      o.objects
                                        .filter(
                                          (b) =>
                                            objectFilter === "all" ||
                                            b.label === objectFilter,
                                        )
                                        .forEach((b) => {
                                          const key = String(b.track_id);
                                          (acc[key] ??= []).push(
                                            `${(b.box[0] + b.box[2]) * 500},${(b.box[1] + b.box[3]) * 500}`,
                                          );
                                        });
                                      return acc;
                                    },
                                    {},
                                  ),
                              ).map(([id, points]) => (
                                <polyline
                                  key={id}
                                  points={points.join(" ")}
                                  fill="none"
                                  stroke="#9beaff"
                                  strokeWidth="2"
                                  vectorEffect="non-scaling-stroke"
                                />
                              ))}
                            {boxes.map((b, i) => (
                              <g
                                key={i}
                                className={
                                  b.debris_candidate ? "debris-candidate" : ""
                                }
                              >
                                {masks &&
                                  b.polygons?.map((p, j) => (
                                    <polygon
                                      key={j}
                                      points={p
                                        .map(
                                          ([x, y]) => `${x * 1000},${y * 1000}`,
                                        )
                                        .join(" ")}
                                      className="instance-mask"
                                    />
                                  ))}
                                {overlay && (
                                  <rect
                                    x={b.box[0] * 1000}
                                    y={b.box[1] * 1000}
                                    width={(b.box[2] - b.box[0]) * 1000}
                                    height={(b.box[3] - b.box[1]) * 1000}
                                  />
                                )}{" "}
                                {overlay && (
                                  <text
                                    x={Math.max(2, b.box[0] * 1000 + 4)}
                                    y={Math.max(24, b.box[1] * 1000 - 7)}
                                  >
                                    #{b.track_id} {label(b.label)}{" "}
                                    {Math.round(b.confidence * 100)}%
                                  </text>
                                )}
                              </g>
                            ))}
                          </svg>
                        </div>
                      ) : (
                        <div className="player-wait">
                          <ScanLine size={44} />
                          <p>
                            {detail.status === "failed"
                              ? "Video is not ready"
                              : "Preparing video for analysis..."}
                          </p>
                        </div>
                      )}
                    </div>
                    <div className="player-toolbar">
                      <label>
                        <input
                          type="checkbox"
                          checked={overlay}
                          onChange={(e) => setOverlay(e.target.checked)}
                        />
                        Box
                      </label>
                      <label>
                        <input
                          type="checkbox"
                          checked={masks}
                          onChange={(e) => setMasks(e.target.checked)}
                          disabled={!health?.segmentation}
                        />
                        Mask
                      </label>
                      <label>
                        <input
                          type="checkbox"
                          checked={trails}
                          onChange={(e) => setTrails(e.target.checked)}
                        />
                        Track
                      </label>
                      <select
                        aria-label="Filter objects on video"
                        value={objectFilter}
                        onChange={(e) => setObjectFilter(e.target.value)}
                      >
                        <option value="all">All Objects</option>
                        {Object.keys(detail.object_counts || {}).map((k) => (
                          <option key={k} value={k}>
                            {label(k)}
                          </option>
                        ))}
                      </select>
                      <span>
                        Sampling {detail.sample_fps ?? health?.sample_fps ?? 1}{" "}
                        fps
                      </span>
                    </div>
                  </section>
                  <section className="timeline-panel">
                    <div className="panel-heading">
                      <h2>
                        <Layers3 size={16} />
                        Timeline
                      </h2>
                      <span>{detail.segments?.length || 0} segments</span>
                    </div>
                    <div className="timeline-track">
                      {(detail.segments || []).map((s) => (
                        <button
                          key={s.id}
                          className={
                            (current >= s.start && current < s.end
                              ? "playing "
                              : "") + (s.ai ? "ai-segment" : "")
                          }
                          style={{ flex: Math.max(0.1, s.end - s.start) }}
                          onClick={() => {
                            seek(s.start);
                            setExpandedSegmentId(s.id);
                          }}
                          title={`${time(s.start)}–${time(s.end)} · ${s.labels.map(label).join(", ")}`}
                          aria-label={`Open segment at ${time(s.start)}`}
                        />
                      ))}
                    </div>
                    <div className="timeline-labels">
                      <span>00:00</span>
                      <span>
                        {time(current)} / {time(detail.duration)}
                      </span>
                      <span>{time(detail.duration)}</span>
                    </div>
                  </section>
                  <section className="segments-panel">
                    <div className="panel-heading">
                      <h2>
                        <Search size={16} />
                        Find Moments
                      </h2>
                      <span className="subtle">By labels & descriptions</span>
                    </div>
                    <form
                      className="segment-search"
                      onSubmit={(e) => void search(e)}
                    >
                      <Search size={17} />
                      <input
                        placeholder="Try 'debris', 'bottle', 'plastic', 'bag'..."
                        aria-label="Search video segments"
                        value={query}
                        maxLength={500}
                        onChange={(e) => setQuery(e.target.value)}
                      />
                      <button
                        disabled={searching || !ready}
                        aria-label="Search"
                      >
                        {searching ? (
                          <LoaderCircle className="spin" size={16} />
                        ) : (
                          <ArrowRight size={18} />
                        )}
                      </button>
                    </form>
                    <div className="segment-list">
                      {segments.map((s) => {
                        const isExpanded = expandedSegmentId === s.id;
                        return (
                          <div
                            className={`segment-card ${isExpanded ? "expanded" : ""}`}
                            key={s.id}
                          >
                            <div
                              className="segment-row"
                              onClick={() => {
                                setExpandedSegmentId(isExpanded ? null : s.id);
                                seek(s.start);
                              }}
                              role="button"
                              tabIndex={0}
                              aria-expanded={isExpanded}
                            >
                              <div
                                className="segment-thumb"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  seek(s.start);
                                }}
                                title={`Play from ${time(s.start)}`}
                              >
                                <img src={s.thumbnail} alt="" loading="lazy" />
                                <span>
                                  <Play size={12} />
                                </span>
                              </div>
                              <div className="segment-body">
                                <div className="segment-header-line">
                                  <span className="segment-time">
                                    {time(s.start)} — {time(s.end)}
                                    {s.ai && (
                                      <span className="ai-pill" title="Analyzed by VLM">
                                        <Sparkles size={11} /> AI
                                      </span>
                                    )}
                                  </span>
                                  <span className="segment-duration">
                                    {(s.end - s.start).toFixed(0)}s
                                  </span>
                                </div>
                                <p className={isExpanded ? "desc-full" : "desc-clamped"}>
                                  {s.description ||
                                    (s.labels.length
                                      ? `Detected: ${s.labels.map(label).join(", ")}`
                                      : "No objects flagged in this interval.")}
                                </p>
                                {!isExpanded && (
                                  <div className="tags">
                                    {s.labels.map((l) => (
                                      <span key={l} className="tag-detector">{label(l)}</span>
                                    ))}
                                    {(s.tags || []).slice(0, 4).map((t) => (
                                      <span key={t} className="tag-vlm">{t}</span>
                                    ))}
                                  </div>
                                )}
                              </div>
                              <div className="segment-toggle">
                                {isExpanded ? <ChevronDown size={17} /> : <ChevronRight size={17} />}
                              </div>
                            </div>

                            {isExpanded && (
                              <div className="segment-expanded-details">
                                <div className="detail-grid">
                                  <div className="detail-item">
                                    <span className="detail-label">Interval</span>
                                    <span className="detail-val">
                                      {time(s.start)} – {time(s.end)} ({(s.end - s.start).toFixed(1)} seconds)
                                    </span>
                                  </div>
                                  <div className="detail-item">
                                    <span className="detail-label">Local Detector</span>
                                    <span className="detail-val">
                                      {s.labels.length
                                        ? s.labels.map(label).join(", ")
                                        : "None detected"}
                                    </span>
                                  </div>
                                </div>

                                {s.tags && s.tags.length > 0 && (
                                  <div className="expanded-tags-section">
                                    <span className="detail-label">VLM Scene Tags</span>
                                    <div className="tags">
                                      {s.tags.map((t) => (
                                        <span key={t} className="tag-vlm">
                                          {t}
                                        </span>
                                      ))}
                                    </div>
                                  </div>
                                )}

                                <div className="segment-actions">
                                  <button
                                    type="button"
                                    className="segment-action-btn"
                                    onClick={() => seek(s.start)}
                                  >
                                    <Play size={13} />
                                    Play from start ({time(s.start)})
                                  </button>
                                  <button
                                    type="button"
                                    className="segment-action-btn"
                                    onClick={() => seek(Math.max(0, s.end - 1))}
                                  >
                                    <Clock3 size={13} />
                                    Jump to end ({time(s.end)})
                                  </button>
                                  <button
                                    type="button"
                                    className="segment-action-btn"
                                    onClick={() => {
                                      const term = (s.tags && s.tags[0]) || s.labels[0] || "debris";
                                      setQuery(term);
                                    }}
                                  >
                                    <Search size={13} />
                                    Filter similar
                                  </button>
                                </div>
                              </div>
                            )}
                          </div>
                        );
                      })}
                      {!segments.length && (
                        <p className="inline-empty">
                          {results
                            ? "No matching segments found. Try a different query."
                            : "Video segments will appear after processing."}
                        </p>
                      )}
                    </div>
                  </section>
                </div>
                <section className="insight-panel">
                  <div className="insight-title">
                    <span>
                      <Sparkles size={18} />
                    </span>
                    <div>
                      <h2>Smart Insights</h2>
                      <small>
                        {health?.ai_configured
                          ? `${health.provider} · ${health.model}`
                          : "Connect AI for deeper analysis"}
                      </small>
                    </div>
                  </div>
                  <div className="tabs">
                    {[
                      ["summary", "Summary"],
                      ["chat", "Q&A"],
                      ["objects", "Objects"],
                    ].map(([id, t]) => (
                      <button
                        key={id}
                        className={tab === id ? "active" : ""}
                        onClick={() => setTab(id)}
                      >
                        {t}
                      </button>
                    ))}
                  </div>
                  {tab === "summary" && (
                    <div className="insight-content">
                      {detail.summary ? (
                        <>
                          <div className="section-kicker">
                            <Sparkles size={13} />
                            AI SUMMARY
                          </div>
                          <MarkdownView content={detail.summary} className="summary-markdown" />
                        </>
                      ) : (
                        <div className="ai-empty">
                          <div>
                            <Sparkles size={26} />
                          </div>
                          <h3>
                            {health?.ai_configured
                              ? "No AI Summary Yet"
                              : "Add an AI Perspective"}
                          </h3>
                          <p>
                            {health?.ai_configured
                              ? "Summary appears once AI analysis completes. If an error occurred, check settings and re-analyze."
                              : "Connect Gemini, OpenAI, or LiteLLM to summarize and chat about this video."}
                          </p>
                          <button
                            className="secondary"
                            onClick={() => void openSettings()}
                          >
                            View Configuration
                            <ArrowUpRight size={14} />
                          </button>
                        </div>
                      )}
                      <div className="facts">
                        <h3>Processing Details</h3>
                        <div>
                          <span>Tracks Created</span>
                          <strong>{detail.track_count ?? 0}</strong>
                        </div>
                        <div>
                          <span>Frames Analyzed</span>
                          <strong>{detail.processed_frames ?? "—"}</strong>
                        </div>
                        <div>
                          <span>Object Classes</span>
                          <strong>
                            {Object.keys(detail.object_counts || {}).length}
                          </strong>
                        </div>
                        <div>
                          <span>Processing Time</span>
                          <strong>
                            {detail.elapsed_seconds
                              ? `${detail.elapsed_seconds}s`
                              : "—"}
                          </strong>
                        </div>
                        <div>
                          <span>Inference Device</span>
                          <strong>CPU</strong>
                        </div>
                      </div>
                      <div className="evidence-note">
                        <ShieldCheck size={17} />
                        <p>
                          COCO detects general common objects, not verified debris. VLM analyzes sampled frames and may make errors; always cross-reference the original video.
                        </p>
                      </div>
                    </div>
                  )}
                  {tab === "objects" && (
                    <div className="insight-content">
                      <div className="section-kicker">
                        <ScanLine size={13} />
                        {detail.detection_model || health?.detection_model || "YOLO11-Litter-Seg"} {detail.detection_device === "cloud-vlm" ? "· Cloud VLM" : "· CPU"}
                      </div>
                      <p className="object-explain">
                        Detections across sampled frames, not unique individual counts.
                      </p>
                      {Object.entries(detail.object_counts || {})
                        .sort((a, b) => b[1] - a[1])
                        .map(([k, v]) => (
                          <button
                            className="object-row"
                            key={k}
                            onClick={() => {
                              setObjectFilter(k);
                              setOverlay(true);
                            }}
                          >
                            <span>
                              <ScanLine size={15} />
                              {label(k)}
                              <small>{k}</small>
                            </span>
                            <strong>{v}</strong>
                          </button>
                        ))}
                      {!Object.keys(detail.object_counts || {}).length && (
                        <p className="inline-empty">
                          No objects detected.
                        </p>
                      )}
                      <div className="evidence-note">
                        <CircleHelp size={17} />
                        <p>
                          Masks and track IDs are generated on CPU. IDs may shift upon occlusion, camera jitter, or large sampling intervals; not a unique entity count.
                        </p>
                      </div>
                    </div>
                  )}
                  {tab === "chat" && (
                    <div className="chat-layout">
                      <div className="chat-messages">
                        {!messages.length && (
                          <div className="chat-welcome">
                            <MessageSquare size={26} />
                            <h3>What would you like to know?</h3>
                            <p>
                              Ask questions about video contents. Answers include video segment timestamps as evidence.
                            </p>
                            {[
                              "Are there signs of floating debris in this video?",
                              "Distinguish trash from aquatic plants and water reflections.",
                              "Where do candidate debris objects appear?",
                            ].map((q) => (
                              <button
                                key={q}
                                disabled={!ready || !health?.ai_configured}
                                onClick={() => void ask(q)}
                              >
                                {q}
                                <ArrowUpRight size={14} />
                              </button>
                            ))}
                          </div>
                        )}
                        {messages.map((m, i) => (
                          <div className={"message " + m.role} key={i}>
                            <span className="message-label">
                              {m.role === "user" ? "YOU" : "RIVERLENS AI"}
                            </span>
                            {m.role === "assistant" ? (
                              <MarkdownView content={m.text} className="chat-markdown" />
                            ) : (
                              <p>{m.text}</p>
                            )}
                            {m.sources?.length ? (
                              <div className="sources">
                                {m.sources.map((s) => (
                                  <button
                                    key={s.id}
                                    onClick={() => seek(s.start)}
                                  >
                                    <Play size={11} />
                                    {time(s.start)} — {time(s.end)}
                                  </button>
                                ))}
                              </div>
                            ) : null}
                            {m.scope && <small>{m.scope}</small>}
                          </div>
                        ))}
                        {asking && (
                          <div className="thinking">
                            <LoaderCircle className="spin" size={15} />
                            Cross-referencing video evidence...
                          </div>
                        )}
                        <div ref={chatBottom} />
                      </div>
                      <form
                        className="chat-form"
                        onSubmit={(e) => {
                          e.preventDefault();
                          void ask();
                        }}
                      >
                        <input
                          aria-label="Question about video"
                          value={question}
                          maxLength={2000}
                          onChange={(e) => setQuestion(e.target.value)}
                          placeholder={
                            health?.ai_configured
                              ? "Ask about this video..."
                              : "API key required for Q&A"
                          }
                          disabled={!ready || !health?.ai_configured}
                        />
                        <button
                          aria-label="Send question"
                          disabled={
                            asking ||
                            !question.trim() ||
                            !ready ||
                            !health?.ai_configured
                          }
                        >
                          <Send size={17} />
                        </button>
                      </form>
                      <p className="chat-footnote">
                        AI can make mistakes. Always verify with original video evidence.
                      </p>
                    </div>
                  )}
                </section>
              </div>
            </>
          )}
        </main>
      </div>
      {(settings || help) && (
        <div
          className="modal-backdrop"
          onClick={() => {
            setSettings(false);
            setHelp(false);
          }}
        >
          <section
            className="modal"
            role="dialog"
            aria-modal="true"
            aria-label={settings ? "System Settings" : "User Guide"}
            onClick={(e) => e.stopPropagation()}
          >
            <button
              className="modal-close"
              aria-label="Close"
              onClick={() => {
                setSettings(false);
                setHelp(false);
              }}
            >
              <X size={20} />
            </button>
            <div className="modal-icon">
              {settings ? <Settings2 size={25} /> : <CircleHelp size={25} />}
            </div>
            <h2>
              {settings
                ? "AI & System Configuration"
                : "From Video to Insight"}
            </h2>
            {settings ? (
              <>
                <p>
                  Switch AI providers, select models, or update credentials. Changes apply immediately and are permanently saved to <code>.env</code>.
                </p>

                <form className="cfg-form" onSubmit={(e) => void handleSaveConfig(e)}>
                  <div className="cfg-field">
                    <label>AI PROVIDER</label>
                    <div className="cfg-provider-pills">
                      {[
                        { id: "gemini", name: "Google Gemini" },
                        { id: "litellm", name: "LiteLLM (Local / Proxy)" },
                        { id: "openai", name: "OpenAI" },
                      ].map((p) => (
                        <button
                          key={p.id}
                          type="button"
                          className={"cfg-provider-pill " + (cfgProvider === p.id ? "active" : "")}
                          onClick={() => handleProviderSelect(p.id)}
                        >
                          <Sparkles size={13} />
                          {p.name}
                        </button>
                      ))}
                    </div>
                  </div>

                  <div className="cfg-field">
                    <label>
                      <span>MODEL</span>
                      {fullConfig?.providers[cfgProvider]?.model && (
                        <span className="cfg-hint">Saved: {fullConfig.providers[cfgProvider].model}</span>
                      )}
                    </label>
                    {fullConfig?.providers[cfgProvider]?.preset_models?.length ? (
                      <select
                        value={
                          fullConfig.providers[cfgProvider].preset_models.includes(cfgModel)
                            ? cfgModel
                            : "custom"
                        }
                        onChange={(e) => {
                          if (e.target.value !== "custom") {
                            setCfgModel(e.target.value);
                          }
                        }}
                      >
                        {fullConfig.providers[cfgProvider].preset_models.map((m) => (
                          <option key={m} value={m}>
                            {m}
                          </option>
                        ))}
                        <option value="custom">Custom model name...</option>
                      </select>
                    ) : null}
                    <input
                      placeholder="Enter model name (e.g. gemini-3.6-flash)..."
                      value={cfgModel}
                      onChange={(e) => setCfgModel(e.target.value)}
                      required
                    />
                  </div>

                  {(cfgProvider === "litellm" || cfgProvider === "openai") && (
                    <div className="cfg-field">
                      <label>
                        <span>BASE URL</span>
                        <span className="cfg-hint">
                          {cfgProvider === "litellm" ? "Default: http://172.18.60.121:4000" : "Optional proxy or base URL"}
                        </span>
                      </label>
                      <input
                        placeholder={
                          cfgProvider === "litellm"
                            ? "http://172.18.60.121:4000"
                            : "https://api.openai.com/v1"
                        }
                        value={cfgBaseUrl}
                        onChange={(e) => setCfgBaseUrl(e.target.value)}
                      />
                    </div>
                  )}

                  <div className="cfg-field">
                    <label>
                      <span>API KEY</span>
                      <span className="cfg-hint">
                        {fullConfig?.providers[cfgProvider]?.has_key
                          ? `Saved (${fullConfig.providers[cfgProvider].key_masked})`
                          : "Not configured"}
                      </span>
                    </label>
                    <input
                      type="password"
                      placeholder={
                        fullConfig?.providers[cfgProvider]?.has_key
                          ? "Leave blank to keep current saved key"
                          : "Enter API key..."
                      }
                      value={cfgKey}
                      onChange={(e) => setCfgKey(e.target.value)}
                    />
                  </div>

                  <div className="cfg-field">
                    <label>
                      <span>DETECTION ENGINE / MODEL</span>
                      <span className="cfg-hint">
                        {cfgDetectionModel === "gemini" ? "Cloud Multimodal VLM (Zero-shot)" : "CPU Instance Segmentation"}
                      </span>
                    </label>
                    <select
                      value={cfgDetectionModel}
                      onChange={(e) => setCfgDetectionModel(e.target.value)}
                    >
                      {(fullConfig?.available_detection_models || [
                        { id: "yolo11_litter_seg", name: "YOLO11-Litter-Seg (Trained for Aquatic Litter: Plastic, Metal, Glass, Paper)" },
                        { id: "yolo11n-seg", name: "YOLO11n-seg (General COCO 80: bottle, cup, etc.)" },
                        { id: "gemini", name: "Gemini Vision Detector (Multimodal VLM · Zero-shot 2D Bounding Boxes)" }
                      ]).map((m) => (
                        <option key={m.id} value={m.id}>
                          {m.name}
                        </option>
                      ))}
                    </select>
                    {cfgDetectionModel === "gemini" && (
                      <p style={{ fontSize: "11px", color: "#64748b", marginTop: "4px", lineHeight: "1.4" }}>
                        ⚡ <strong>Gemini Vision Mode</strong>: Runs multimodal 2D bounding box localization directly using Gemini. Automatically applies adaptive keyframe sampling (0.25 FPS) and synchronizes detections with segment reasoning.
                      </p>
                    )}
                  </div>

                  <div className="cfg-field">
                    <label>
                      <span>DETECTION CONFIDENCE THRESHOLD</span>
                      <span className="cfg-hint">{cfgDetectionConfidence}</span>
                    </label>
                    <input
                      type="range"
                      min="0.05"
                      max="0.60"
                      step="0.05"
                      value={cfgDetectionConfidence}
                      onChange={(e) => setCfgDetectionConfidence(parseFloat(e.target.value))}
                    />
                  </div>

                  <div className="cfg-actions">
                    <button className="primary" type="submit" disabled={savingConfig}>
                      {savingConfig ? <LoaderCircle size={15} className="spin" /> : <Check size={15} />}
                      {savingConfig ? "Saving..." : "Save & Apply"}
                    </button>

                    <button
                      className="secondary"
                      type="button"
                      disabled={testingAi}
                      onClick={async () => {
                        setTestingAi(true);
                        setAiTestResult(null);
                        try {
                          const res = await api<{ status: string; provider: string; model: string; response: string }>("/ai/test", { method: "POST" });
                          setAiTestResult({ ok: true, msg: `Verified ✓ (${res.provider} / ${res.model})` });
                        } catch (err) {
                          setAiTestResult({ ok: false, msg: (err as Error).message });
                        } finally {
                          setTestingAi(false);
                        }
                      }}
                    >
                      {testingAi ? <LoaderCircle size={15} className="spin" /> : <Sparkles size={15} />}
                      {testingAi ? "Testing..." : "Test AI Connection"}
                    </button>
                  </div>

                  {saveResult && (
                    <div style={{ fontSize: "11px", fontWeight: 500, color: saveResult.ok ? "#3e7b48" : "#c2410c", padding: "4px 0" }}>
                      {saveResult.msg}
                    </div>
                  )}
                  {aiTestResult && (
                    <div style={{ fontSize: "11px", fontWeight: 500, color: aiTestResult.ok ? "#3e7b48" : "#c2410c", padding: "4px 0" }}>
                      {aiTestResult.msg}
                    </div>
                  )}
                </form>

                <div className="config-rows">
                  <div>
                    <span>Detection Engine</span>
                    <strong>
                      {health?.detection_model || "YOLO11-Litter-Seg"}{" "}
                      {health?.detection_ready ? "✓" : "· Missing key/model"}
                    </strong>
                  </div>
                  <div>
                    <span>Active Provider</span>
                    <strong>{health?.provider}</strong>
                  </div>
                  <div>
                    <span>Active Model</span>
                    <strong>{health?.model}</strong>
                  </div>
                </div>

                <p className="modal-note">
                  AI is only invoked when you enable it. Sampled JPEG frames are sent to your chosen provider. Audio is not processed. Pretrained COCO weights are not specialized for aquatic debris; field validation is recommended.
                </p>
              </>
            ) : (
              <>
                <p>Three simple steps to explore your videos.</p>
                <ol className="help-steps">
                  <li>
                    <span>01</span>
                    <div>
                      <strong>Add Video</strong>
                      <p>
                        Upload a video. The CPU worker normalizes the stream and runs object detection on sampled frames.
                      </p>
                    </div>
                  </li>
                  <li>
                    <span>02</span>
                    <div>
                      <strong>Explore Moments</strong>
                      <p>
                        Inspect bounding boxes, filter detected classes, and search segments by English or Vietnamese keywords.
                      </p>
                    </div>
                  </li>
                  <li>
                    <span>03</span>
                    <div>
                      <strong>Analyze with AI</strong>
                      <p>
                        Configure an API key to generate video summaries and interact via multimodal Q&A with timestamped evidence.
                      </p>
                    </div>
                  </li>
                </ol>
                <p className="modal-note">
                  Local build for internal analysis. No authentication layer; do not expose this port directly to public networks.
                </p>
              </>
            )}
            <button
              className="primary modal-done"
              onClick={() => {
                setSettings(false);
                setHelp(false);
              }}
            >
              Got it
              <Check size={16} />
            </button>
          </section>
        </div>
      )}
    </div>
  );
}
createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
