import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  ArrowLeft,
  ArrowRight,
  ArrowUpRight,
  Check,
  CheckCircle2,
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
  track_count?: number;
};
type Health = {
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
  person: "Người",
  car: "Ô tô",
  motorcycle: "Xe máy",
  bicycle: "Xe đạp",
  truck: "Xe tải",
  bus: "Xe buýt",
  dog: "Chó",
  cat: "Mèo",
  chair: "Ghế",
  bottle: "Chai",
  laptop: "Laptop",
  book: "Sách",
  "cell phone": "Điện thoại",
  handbag: "Túi xách",
  backpack: "Ba lô",
  bird: "Chim",
  boat: "Thuyền",
  tv: "TV",
  cup: "Cốc",
};
const label = (s: string) => labels[s] || s;
async function api<T>(path: string, opts?: RequestInit): Promise<T> {
  const r = await fetch("/api" + path, opts);
  const data = await r.json();
  if (!r.ok)
    throw new Error(
      typeof data.detail === "string"
        ? data.detail
        : "Yêu cầu không thành công. Vui lòng thử lại.",
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
      setError(`Dung lượng tối đa ${health.max_upload_mb} MB.`);
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
        xhr.onerror = () => reject(new Error("Mất kết nối khi tải video."));
        xhr.ontimeout = () => reject(new Error("Tải video quá thời gian."));
        xhr.onload = () => {
          try {
            const data = JSON.parse(xhr.responseText);
            xhr.status < 300
              ? resolve(data)
              : reject(new Error(data.detail || "Không tải được video."));
          } catch {
            reject(new Error("Phản hồi tải video không hợp lệ."));
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
        <div className="workspace-label">KHÔNG GIAN LÀM VIỆC</div>
        <button
          className={"nav-item " + (!settings ? "active" : "")}
          onClick={() => {
            setSelected(null);
            setSettings(false);
          }}
        >
          <Layers3 size={19} />
          Thư viện video<span className="nav-count">{items.length}</span>
        </button>
        <button className="nav-item" onClick={() => setSettings(true)}>
          <Settings2 size={19} />
          Cấu hình hệ thống
          <ArrowUpRight size={15} />
        </button>
        <div className="side-note">
          <div className="side-note-icon">
            <Cpu size={21} />
          </div>
          <strong>Thông minh. Nhẹ nhàng.</strong>
          <p>
            Theo dõi rác nổi trên CPU.
            <br />
            Kết nối AI theo cách của bạn.
          </p>
          <div className="cpu-pill">
            <span />
            CPU ONLY
          </div>
        </div>
        <div className="side-bottom">
          <button onClick={() => setHelp(true)}>
            <CircleHelp size={17} />
            Hướng dẫn sử dụng
            <ArrowUpRight size={14} />
          </button>
          <div className="profile">
            <div className="avatar">L</div>
            <div>
              <strong>Local workspace</strong>
              <small>Môi trường phát triển</small>
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
            <button onClick={() => setSelected(null)}>Thư viện video</button>
            {selected && (
              <>
                <ChevronRight size={14} />
                <span className="breadcrumb-video">Phân tích</span>
              </>
            )}
          </div>
          <div className="header-right">
            <span className={"connection " + (health ? "" : "offline")}>
              <i />
              {health ? "Hệ thống sẵn sàng" : "Đang kết nối"}
            </span>
            <span className="header-divider" />
            <span className="version">MVP / 0.1</span>
          </div>
        </header>
        <main>
          {error && (
            <div className="alert error" role="alert">
              {error}
              <button aria-label="Đóng thông báo" onClick={() => setError("")}>
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
                    Nhìn rõ từng dòng nước<span>.</span>
                  </h1>
                  <p>
                    Quan trắc rác nổi trên sông, hồ, ao. Nhận diện, theo dõi và
                    phân tích cùng AI.
                  </p>
                </div>
                <button
                  className="primary"
                  disabled={uploading}
                  onClick={() => input.current?.click()}
                >
                  <Plus size={18} />
                  Thêm video
                </button>
              </div>
              <section className="stats">
                <div>
                  <span className="stat-icon">
                    <Video size={21} />
                  </span>
                  <div>
                    <small>Video trong thư viện</small>
                    <strong>
                      {items.length}
                      <span>video</span>
                    </strong>
                  </div>
                </div>
                <div>
                  <span className="stat-icon">
                    <CheckCircle2 size={21} />
                  </span>
                  <div>
                    <small>Đã phân tích</small>
                    <strong>
                      {items.filter((i) => i.status === "ready").length}
                      <span>hoàn tất</span>
                    </strong>
                  </div>
                </div>
                <div>
                  <span className="stat-icon">
                    <Clock3 size={21} />
                  </span>
                  <div>
                    <small>Tổng thời lượng</small>
                    <strong>
                      {time(items.reduce((s, i) => s + i.duration, 0))}
                      <span>phút : giây</span>
                    </strong>
                  </div>
                </div>
                <div>
                  <span className="stat-icon">
                    <Sparkles size={21} />
                  </span>
                  <div>
                    <small>AI provider</small>
                    <strong className="provider-name">
                      {health?.provider || "—"}
                      <span
                        className={
                          health?.ai_configured ? "configured" : "pending"
                        }
                      >
                        {health?.ai_configured ? "Đã cấu hình" : "Chưa có key"}
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
                      ? `Đang tải video · ${uploadProgress}%`
                      : "Bắt đầu từ một góc nhìn mặt nước"}
                  </h2>
                  <p>
                    {uploading
                      ? "Giữ trang mở trong khi video được tải lên."
                      : "Kéo thả video vào đây, hoặc chọn một file từ máy tính."}
                  </p>
                  <span>
                    MP4, MOV, WebM · Tối đa {health?.max_upload_mb ?? 256} MB ·{" "}
                    {Math.round((health?.max_duration_seconds ?? 1800) / 60)}{" "}
                    phút
                  </span>
                </div>
                <button
                  className="secondary"
                  onClick={() => input.current?.click()}
                  disabled={uploading}
                >
                  Chọn video
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
                    Thư viện của bạn <span>{items.length}</span>
                  </h2>
                  <p>Mọi góc nhìn, trong cùng một không gian.</p>
                </div>
                <div className="library-search">
                  <Search size={17} />
                  <input
                    aria-label="Tìm video trong thư viện"
                    placeholder="Tìm theo tên video..."
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
                            ? "Đã phân tích"
                            : item.status === "failed"
                              ? "Có lỗi"
                              : item.status === "queued"
                                ? "Đang chờ"
                                : "Đang xử lý"}
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
                              "vi-VN",
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
                      ? "Không tìm thấy video"
                      : "Một thư viện đầy tiềm năng"}
                  </h3>
                  <p>
                    {filter
                      ? "Thử tìm với một tên khác."
                      : "Thêm video đầu tiên để khám phá những gì bạn có thể đã bỏ lỡ."}
                  </p>
                  {!filter && (
                    <div className="feature-chips">
                      <span>
                        <ScanLine size={14} />
                        Segmentation & tracking
                      </span>
                      <span>
                        <Sparkles size={14} />
                        Tóm tắt nội dung
                      </span>
                      <span>
                        <MessageSquare size={14} />
                        Hỏi đáp với AI
                      </span>
                    </div>
                  )}
                </section>
              )}
              <div className="privacy-footer">
                <ShieldCheck size={15} />
                <span>
                  Video lưu trên máy chủ local. Ảnh mẫu được gửi đến provider
                  bạn chọn khi bật AI.
                </span>
                <span className="footer-badge">BUILT FOR CURIOSITY</span>
              </div>
            </>
          ) : !detail ? (
            <div className="loading">
              <LoaderCircle className="spin" />
              Đang mở video...
            </div>
          ) : (
            <>
              <button className="back-link" onClick={() => setSelected(null)}>
                <ArrowLeft size={16} />
                Quay lại thư viện
              </button>
              <div className="detail-heading">
                <div>
                  <h1>{detail.name}</h1>
                  <p>
                    {time(detail.duration)} · {detail.width} × {detail.height} ·{" "}
                    {(detail.size / 1024 / 1024).toFixed(1)} MB{" "}
                    <span className="dot-separator">•</span>{" "}
                    {detail.detection_model || "YOLO11n-seg"} / CPU
                  </p>
                </div>
                <button
                  className="secondary"
                  disabled={!["ready", "failed"].includes(detail.status)}
                  onClick={() => void retry()}
                >
                  <RefreshCw size={15} />
                  Phân tích lại
                </button>
              </div>
              {!ready && detail.status !== "failed" && (
                <div className="processing-banner">
                  <LoaderCircle className="spin" size={20} />
                  <div>
                    <strong>{detail.stage}</strong>
                    <p>Worker CPU đang xử lý. Bạn có thể quay lại thư viện.</p>
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
                  {detail.error} · Nhấn “Phân tích lại” để thử lại.
                </div>
              )}
              {detail.warning && (
                <div className="alert warning">
                  <CircleHelp size={17} />
                  <span>{detail.warning}</span>
                  <button onClick={() => setSettings(true)}>
                    Cấu hình
                    <ArrowUpRight size={14} />
                  </button>
                </div>
              )}
              <div className="domain-notice">
                <ShieldCheck size={15} />
                <span>
                  Chế độ quan trắc rác nổi · Mask màu cam là vật thể có thể liên
                  quan đến rác, chưa xác nhận vị trí trên mặt nước hoặc bản chất
                  rác. COCO pretrained chưa chuyên biệt cho môi trường sông hồ.
                </span>
              </div>
              <div className="analysis-layout">
                <div className="analysis-left">
                  <section className="player-panel">
                    <div className="panel-heading">
                      <h2>
                        <Play size={16} />
                        Trình phát video
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
                              ? "Video chưa sẵn sàng"
                              : "Đang chuẩn bị video để phân tích"}
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
                        aria-label="Lọc đối tượng trên video"
                        value={objectFilter}
                        onChange={(e) => setObjectFilter(e.target.value)}
                      >
                        <option value="all">Tất cả đối tượng</option>
                        {Object.keys(detail.object_counts || {}).map((k) => (
                          <option key={k} value={k}>
                            {label(k)}
                          </option>
                        ))}
                      </select>
                      <span>
                        Lấy mẫu {detail.sample_fps ?? health?.sample_fps ?? 1}{" "}
                        fps
                      </span>
                    </div>
                  </section>
                  <section className="timeline-panel">
                    <div className="panel-heading">
                      <h2>
                        <Layers3 size={16} />
                        Dòng thời gian
                      </h2>
                      <span>{detail.segments?.length || 0} đoạn</span>
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
                          onClick={() => seek(s.start)}
                          title={`${time(s.start)}–${time(s.end)} · ${s.labels.map(label).join(", ")}`}
                          aria-label={`Mở đoạn ${time(s.start)}`}
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
                        Tìm khoảnh khắc
                      </h2>
                      <span className="subtle">Theo nhãn & mô tả</span>
                    </div>
                    <form
                      className="segment-search"
                      onSubmit={(e) => void search(e)}
                    >
                      <Search size={17} />
                      <input
                        placeholder="Thử “rác”, “chai”, “bottle”, “nhựa”..."
                        aria-label="Tìm đoạn video"
                        value={query}
                        maxLength={500}
                        onChange={(e) => setQuery(e.target.value)}
                      />
                      <button
                        disabled={searching || !ready}
                        aria-label="Tìm kiếm"
                      >
                        {searching ? (
                          <LoaderCircle className="spin" size={16} />
                        ) : (
                          <ArrowRight size={18} />
                        )}
                      </button>
                    </form>
                    <div className="segment-list">
                      {segments.map((s) => (
                        <button
                          className="segment-row"
                          key={s.id}
                          onClick={() => seek(s.start)}
                        >
                          <div className="segment-thumb">
                            <img src={s.thumbnail} alt="" loading="lazy" />
                            <span>
                              <Play size={12} />
                            </span>
                          </div>
                          <div>
                            <span className="segment-time">
                              {time(s.start)} — {time(s.end)}
                              {s.ai && <Sparkles size={12} />}
                            </span>
                            <p>
                              {s.description ||
                                (s.labels.length
                                  ? `Phát hiện: ${s.labels.map(label).join(", ")}`
                                  : "Chưa phát hiện đối tượng ở ngưỡng cấu hình.")}
                            </p>
                            <div className="tags">
                              {s.labels.slice(0, 5).map((l) => (
                                <span key={l}>{label(l)}</span>
                              ))}
                            </div>
                          </div>
                          <ChevronRight size={16} />
                        </button>
                      ))}
                      {!segments.length && (
                        <p className="inline-empty">
                          {results
                            ? "Không tìm thấy đoạn phù hợp. Thử một từ khóa khác."
                            : "Các đoạn video sẽ xuất hiện sau khi xử lý."}
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
                      <h2>Góc nhìn thông minh</h2>
                      <small>
                        {health?.ai_configured
                          ? `${health.provider} · ${health.model}`
                          : "Kết nối AI để hiểu sâu hơn"}
                      </small>
                    </div>
                  </div>
                  <div className="tabs">
                    {[
                      ["summary", "Tóm tắt"],
                      ["chat", "Hỏi đáp"],
                      ["objects", "Đối tượng"],
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
                            TÓM TẮT BỞI AI
                          </div>
                          <div className="summary-text">{detail.summary}</div>
                        </>
                      ) : (
                        <div className="ai-empty">
                          <div>
                            <Sparkles size={26} />
                          </div>
                          <h3>
                            {health?.ai_configured
                              ? "Chưa có bản tóm tắt AI"
                              : "Thêm một góc nhìn từ AI"}
                          </h3>
                          <p>
                            {health?.ai_configured
                              ? "Bản tóm tắt xuất hiện khi phân tích AI hoàn tất. Nếu có lỗi, kiểm tra cấu hình rồi phân tích lại."
                              : "Kết nối Gemini, OpenAI hoặc LiteLLM để tóm tắt và trò chuyện về video."}
                          </p>
                          <button
                            className="secondary"
                            onClick={() => setSettings(true)}
                          >
                            Xem cấu hình
                            <ArrowUpRight size={14} />
                          </button>
                        </div>
                      )}
                      <div className="facts">
                        <h3>Thông tin xử lý</h3>
                        <div>
                          <span>Track ID đã tạo</span>
                          <strong>{detail.track_count ?? 0}</strong>
                        </div>
                        <div>
                          <span>Khung hình đã phân tích</span>
                          <strong>{detail.processed_frames ?? "—"}</strong>
                        </div>
                        <div>
                          <span>Loại đối tượng</span>
                          <strong>
                            {Object.keys(detail.object_counts || {}).length}
                          </strong>
                        </div>
                        <div>
                          <span>Thời gian xử lý</span>
                          <strong>
                            {detail.elapsed_seconds
                              ? `${detail.elapsed_seconds}s`
                              : "—"}
                          </strong>
                        </div>
                        <div>
                          <span>Thiết bị suy luận</span>
                          <strong>CPU</strong>
                        </div>
                      </div>
                      <div className="evidence-note">
                        <ShieldCheck size={17} />
                        <p>
                          COCO nhận diện vật phổ thông, chưa xác nhận đó là rác.
                          VLM phân tích ảnh mẫu và có thể sai; luôn đối chiếu
                          video gốc.
                        </p>
                      </div>
                    </div>
                  )}
                  {tab === "objects" && (
                    <div className="insight-content">
                      <div className="section-kicker">
                        <ScanLine size={13} />
                        YOLO11n-seg · COCO 80
                      </div>
                      <p className="object-explain">
                        Số lần phát hiện trên các frame lấy mẫu, không phải số
                        cá thể duy nhất.
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
                          Chưa có đối tượng được phát hiện.
                        </p>
                      )}
                      <div className="evidence-note">
                        <CircleHelp size={17} />
                        <p>
                          Mask và track ID được tạo trên CPU. ID có thể đổi khi
                          vật bị che, camera rung hoặc khoảng lấy mẫu lớn; không
                          phải số vật duy nhất.
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
                            <h3>Bạn muốn biết điều gì?</h3>
                            <p>
                              Hỏi về nội dung video. Câu trả lời có thể kèm đoạn
                              video làm bằng chứng.
                            </p>
                            {[
                              "Có dấu hiệu rác nổi trong video không?",
                              "Phân biệt rác với bèo và phản xạ nước.",
                              "Rác nghi vấn xuất hiện ở vị trí nào?",
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
                              {m.role === "user" ? "BẠN" : "RIVERLENS AI"}
                            </span>
                            <p>{m.text}</p>
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
                            Đang đối chiếu nội dung...
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
                          aria-label="Câu hỏi về video"
                          value={question}
                          maxLength={2000}
                          onChange={(e) => setQuestion(e.target.value)}
                          placeholder={
                            health?.ai_configured
                              ? "Hỏi về video này..."
                              : "Cần API key để hỏi đáp"
                          }
                          disabled={!ready || !health?.ai_configured}
                        />
                        <button
                          aria-label="Gửi câu hỏi"
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
                        AI có thể sai. Đối chiếu bằng chứng trước khi sử dụng.
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
            aria-label={settings ? "Cấu hình hệ thống" : "Hướng dẫn sử dụng"}
            onClick={(e) => e.stopPropagation()}
          >
            <button
              className="modal-close"
              aria-label="Đóng"
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
                ? "Kết nối theo cách của bạn"
                : "Từ video đến thông tin"}
            </h2>
            {settings ? (
              <>
                <p>
                  Cấu hình tại file <code>.env</code> ở backend. Khóa API không
                  được gửi đến trình duyệt. Khởi động lại backend sau khi thay
                  đổi.
                </p>
                <div className="config-rows">
                  <div>
                    <span>Inference local</span>
                    <strong>
                      {health?.detection_model || "YOLO11n-seg"} · CPU{" "}
                      {health?.detection_ready ? "✓" : "· Thiếu model"}
                    </strong>
                  </div>
                  <div>
                    <span>AI provider</span>
                    <strong>{health?.provider}</strong>
                  </div>
                  <div>
                    <span>Model</span>
                    <strong>{health?.model}</strong>
                  </div>
                  <div>
                    <span>API key</span>
                    <strong
                      className={
                        health?.ai_configured ? "configured" : "pending"
                      }
                    >
                      {health?.ai_configured
                        ? "Đã cấu hình · chưa xác minh kết nối"
                        : "Chưa cấu hình"}
                    </strong>
                  </div>
                </div>
                <pre>{`AI_PROVIDER=gemini\nAI_MODEL=gemini-2.5-flash\nAI_API_KEY=<khóa của bạn>\n\n# openai: AI_PROVIDER=openai\n# litellm: AI_PROVIDER=litellm\n# AI_BASE_URL=https://your-proxy/v1`}</pre>
                <p className="modal-note">
                  Chỉ bật AI khi bạn đồng ý gửi ảnh mẫu đến provider đã chọn.
                  Không có chuyển provider tự động. Chưa hỗ trợ âm thanh. Model
                  COCO chưa được huấn luyện chuyên cho rác nổi; cần dữ liệu thực
                  địa để đánh giá và fine-tune.
                </p>
              </>
            ) : (
              <>
                <p>Ba bước để bắt đầu khám phá video của bạn.</p>
                <ol className="help-steps">
                  <li>
                    <span>01</span>
                    <div>
                      <strong>Thêm video</strong>
                      <p>
                        Tải video lên. Worker CPU chuẩn hóa video và chạy nhận
                        diện đối tượng trên ảnh lấy mẫu.
                      </p>
                    </div>
                  </li>
                  <li>
                    <span>02</span>
                    <div>
                      <strong>Khám phá khoảnh khắc</strong>
                      <p>
                        Xem bounding box, lọc đối tượng và tìm đoạn theo nhãn
                        tiếng Việt hoặc tiếng Anh.
                      </p>
                    </div>
                  </li>
                  <li>
                    <span>03</span>
                    <div>
                      <strong>Trò chuyện cùng AI</strong>
                      <p>
                        Cấu hình API key để tạo tóm tắt và hỏi đáp kèm mốc thời
                        gian. Không cần GPU local.
                      </p>
                    </div>
                  </li>
                </ol>
                <p className="modal-note">
                  Bản local dành cho review. Chưa có xác thực người dùng; không
                  công khai cổng ra Internet.
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
              Đã hiểu
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
