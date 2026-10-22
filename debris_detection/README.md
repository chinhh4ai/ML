# Riverlens — Aquatic Debris Intelligence

Ứng dụng FastAPI + React để xem, tìm kiếm và phân tích video rác nổi trên sông, hồ, ao. **Chạy inference local bằng CPU**, không cần CUDA/GPU. Gemini, OpenAI hoặc LiteLLM được dùng cho phân tích hình ảnh và hỏi đáp khi bạn cấu hình API key.

## Những gì đã triển khai

- Upload MP4/MOV/WebM, giới hạn dung lượng/thời lượng, chuẩn hóa bằng FFmpeg.
- Worker nền một luồng, SQLite lưu thư viện/kết quả/tiến độ; tác vụ đang chạy được xếp lại hàng đợi sau restart.
- YOLO11n-seg ONNX (416px): box, instance mask dạng polygon, confidence, nhãn COCO.
- Tracking class-aware IoU, track ID và đường di chuyển ngắn; bật/tắt box/mask/track khi xem video.
- Nhãn ứng viên rác cho bottle/cup/bowl/handbag/backpack/umbrella; **đây chỉ là heuristic theo lớp**, chưa xác nhận vật nằm trên nước hoặc là rác. Các lớp khác vẫn được lưu làm ngữ cảnh.
- Chia đoạn video, lưu thumbnail, tìm kiếm theo nhãn/mô tả; hỗ trợ các từ khóa tiếng Việt phổ biến. **Search hiện là keyword, chưa phải semantic/vector search.**
- AI mô tả đoạn, tổng hợp video, hỏi đáp với ảnh bằng chứng và link timestamp. Prompt chuyên cho rác nổi, phân biệt bèo/lá cây/bọt/phản xạ.
- API adapter cho Gemini native và OpenAI-compatible (OpenAI/LiteLLM), retry có giới hạn, tối đa hai request AI đồng thời, không tự chuyển provider.
- UI tiếng Việt, responsive, cùng origin với API ở port **5076**.

## Chạy trên Windows tại `D:\work_space\2026\HIP`

Phiên phát triển cloud không thể ghi trực tiếp vào ổ D: của bạn. Gói `riverlens-HIP.zip` chứa thư mục `HIP`, source, frontend đã build và hai model ONNX; **không chứa API key, video đã tải lên hoặc database**. `.env` trong ZIP được tạo từ template trắng, không lấy từ file cấu hình thật.

1. Cài **Python 3.12 x64**, **FFmpeg** (bao gồm `ffprobe`). Cần **Node.js 22+** nếu build lại giao diện. Để dùng bản frontend đã build trong ZIP, không bắt buộc cài Node.
2. Giải nén ZIP vào `D:\work_space\2026` để có `D:\work_space\2026\HIP`.
3. Mở PowerShell và chạy:

```powershell
Set-Location D:\work_space\2026\HIP
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
notepad .env
powershell -ExecutionPolicy Bypass -File scripts\start.ps1
```

Mở trình duyệt tại `http://127.0.0.1:5076`. API docs: `http://127.0.0.1:5076/docs`.

Nếu FFmpeg chưa có, có thể cài bằng `winget install Gyan.FFmpeg`, sau đó mở lại PowerShell. Script setup dừng nếu thiếu FFmpeg; không tự thay đổi PATH hay cài phần mềm hệ thống.

**Windows chưa được chạy kiểm thử trong phiên cloud Linux.** Các script Windows được chuẩn bị để bạn chạy local, không phải bằng chứng ứng dụng đã chạy trên máy D: của bạn.

## Chọn provider trong `.env`

Mỗi provider có key và model riêng, chỉ cần đổi `AI_PROVIDER`. Dùng một model hỗ trợ ảnh.

```dotenv
AI_PROVIDER=gemini
GEMINI_API_KEY=
GEMINI_MODEL=gemini-2.5-flash

OPENAI_API_KEY=
OPENAI_MODEL=gpt-4.1-mini

LITELLM_API_KEY=
LITELLM_MODEL=gpt-4.1-mini
LITELLM_BASE_URL=http://127.0.0.1:4000/v1
```

Điền key cho provider bạn chọn. Tên model phải còn khả dụng và được tài khoản của bạn cho phép; các tên trong template chỉ là giá trị khởi đầu. Với LiteLLM, `LITELLM_MODEL` phải là alias đã cấu hình trên proxy. App là **client của LiteLLM**, không tự chạy một LiteLLM proxy.

`AI_API_KEY`, `AI_MODEL`, `AI_BASE_URL` là override dùng chung; nên để trống khi dùng biến riêng từng provider. Thứ tự ưu tiên: override → biến provider → mặc định. Gemini cũng nhận `GOOGLE_API_KEY` khi không có `GEMINI_API_KEY`.

Khởi động lại backend sau khi sửa `.env`. Không cần build lại frontend. Video đã xử lý trước đó cần bấm **Phân tích lại** để tạo mô tả AI và tóm tắt mới.

**Không có API key:** segmentation/tracking, phát video, tìm theo nhãn vẫn chạy. UI không tạo tóm tắt hoặc câu trả lời giả. **Có API key:** ảnh JPEG lấy mẫu và mô tả sẽ được gửi đến provider đã chọn; video gốc vẫn lưu local, audio không được gửi/xử lý.

## Chạy trên Linux/macOS

Đã xác minh trên Linux x86_64 với Python 3.12. macOS chưa kiểm thử, và quá trình export model bằng bộ dependency CPU PyTorch này chỉ nhắm Linux/Windows x86_64.

```bash
bash scripts/setup.sh
# Sửa .env
bash scripts/start.sh
```

Setup cài runtime từ `requirements.lock.txt`, build frontend bằng `npm ci` và kiểm tra checksum model được đóng gói. Nếu không có YOLO11n-seg ONNX (ví dụ fresh clone), setup tạo `.model-venv`, tải dependencies export CPU, kiểm tra SHA256 trọng số chính thức rồi export ONNX. Runtime không cần PyTorch/Ultralytics. Model tự export có metadata thời điểm build nên có checksum riêng, được ghi trong file provenance local sau khi xác minh trọng số nguồn. Script kiểm tra hỗ trợ cả model đóng gói và model tự export; không sửa checksum artifact công bố để bỏ qua lỗi.

Mỗi task cloud đã có checkout cô lập: sử dụng checkout hiện hữu, không tạo Git worktree trừ khi được yêu cầu.

## Ubuntu Server — cài đặt và chạy dịch vụ

Script cài hệ thống tự động nhắm **Ubuntu 24.04 x86_64** (Python 3.12). Node.js >=18.19 là đủ với Vite 6 hiện tại; Node 22+ cũng được hỗ trợ. Chạy bằng user thường có quyền sudo, trong thư mục ứng dụng do user đó sở hữu:

```bash
cd /path/to/HIP
bash scripts/install-ubuntu.sh
nano .env
bash scripts/start.sh
```

`install-ubuntu.sh` cài Python venv, FFmpeg, Node/npm qua apt có kiểm tra chữ ký, rồi cài dependency Python/JS và kiểm tra checksum model. Nếu đã chuẩn bị đủ công cụ (hoặc dùng Ubuntu phiên bản khác với Python 3.12 đã cài), dùng `bash scripts/install-ubuntu.sh --skip-system`. Script không tự thêm PPA hoặc thay Python hệ thống.

Mặc định bind `127.0.0.1:5076`. Từ máy cá nhân kết nối qua SSH tunnel:

```bash
ssh -L 5076:127.0.0.1:5076 your-user@your-server
```

Sau đó mở `http://127.0.0.1:5076` trên máy cá nhân. Nếu thật sự cần truy cập trong LAN tin cậy, chạy `APP_HOST=0.0.0.0 bash scripts/start.sh`; app chưa có xác thực nên không mở public trực tiếp. Script không tự mở firewall.

Để tự chạy sau reboot (systemd), dừng foreground server trước rồi:

```bash
bash scripts/install-service.sh
bash scripts/status.sh
sudo journalctl -u riverlens -f
# Sau khi chỉnh .env:
sudo systemctl restart riverlens
# Dừng dịch vụ:
bash scripts/stop-service.sh
```

Service chạy dưới user hiện tại, một Uvicorn process, port 5076; không chạy app bằng root. Mặc định loopback. Có thể đặt `APP_HOST=0.0.0.0` khi chạy script cài service nếu đã có chính sách mạng riêng. Unit chỉ cho ghi `.data`; nếu đổi `VSS_DATA_DIR` sang nơi khác, phải cập nhật `ReadWritePaths` của unit tương ứng.

Script systemd đã được kiểm tra cú pháp/generate unit trong môi trường phát triển; **chưa chạy systemd hoặc apt trên một Ubuntu Server thực**. Việc cài package hệ thống và đăng ký service chỉ xảy ra khi bạn chạy các script tương ứng.

## Đánh giá file MOV gốc tại máy bạn

Ứng dụng local nhận upload tới 256 MB mặc định (có thể chỉnh `.env`). Giới hạn **32 MiB** là của công cụ chuyển file đính kèm vào phiên cloud này, không phải giới hạn CLI hoặc giới hạn cố định của ứng dụng.

```bash
# Đánh giá 60 giây đầu trên CPU, không gửi dữ liệu đến AI:
.venv/bin/python scripts/evaluate_video.py /path/to/20260915_010416000_iOS.MOV

# Toàn bộ video, lưu vào thư mục mới do bạn chọn:
.venv/bin/python scripts/evaluate_video.py /path/to/video.MOV \
  --max-seconds 0 --fps 2 --output .data/evaluations/river-test-01

# Chỉ bật sau khi đã điền API key; gửi tối đa 4 ảnh mẫu đến provider:
.venv/bin/python scripts/evaluate_video.py /path/to/video.MOV \
  --output .data/evaluations/river-test-with-ai --with-ai
```

Kết quả gồm `report.json`, `normalized.mp4`, `overlay.mp4`, ảnh gốc và ảnh overlay ở bốn thời điểm đại diện. `overlay.mp4` có FPS lấy mẫu, không phải bản tracking đủ FPS gốc. Báo cáo có thời gian suy luận, số detection/mask/track và ứng viên rác; **không báo precision/recall/mAP khi chưa có ground truth**. Số ứng viên rác không phải số rác được xác nhận.

Video iOS có thể có rotation metadata/HDR. FFmpeg chuẩn hóa hướng và mã hóa 8-bit yuv420p để phát, nhưng hiện chưa có pipeline tone mapping HDR được kiểm định; cần xem ảnh xuất ra để kiểm tra màu/độ sáng. CLI mặc định không thay đổi file video gốc.

## Cấu hình CPU và giới hạn

| Biến | Mặc định | Ý nghĩa |
|---|---:|---|
| `DETECTION_MODEL` | `yolo11n-seg` | Mask + box. `yolox-nano` chỉ có box; YOLOX không có segmentation sẵn. |
| `DETECTION_CONFIDENCE` | `0.25` | Ngưỡng detector, không phải xác suất một vật là rác |
| `VIDEO_SAMPLE_FPS` | `2` | Tăng để giảm khoảng trống tracking, đổi lại tốn CPU |
| `CPU_THREADS` | `2` | Số thread ONNX Runtime |
| `VIDEO_CHUNK_SECONDS` | `30` | Mỗi đoạn có tối đa 4 ảnh gửi AI |
| `AI_MAX_SEGMENTS` | `60` | Giới hạn đoạn gửi AI mỗi lần xử lý |
| `VIDEO_MAX_DURATION_SECONDS` | `1800` | Thời lượng tối đa video |
| `VIDEO_MAX_UPLOAD_MB` | `256` | Dung lượng file tối đa |

Máy 4–8 CPU core, RAM 8–16 GB là điểm khởi đầu để thử; chưa cam kết thông lượng với video thực. Worker xử lý từng video một. Chỉ chạy **một process Uvicorn**, không dùng nhiều `--workers` vì worker/queue nằm trong process. Production cần hàng đợi ngoài (Redis/Celery), xác thực, quota và storage policy.

## Giới hạn độ chính xác

YOLO11n-seg pretrained COCO **không phải model chuyên rác sông hồ**. Chai nhỏ, túi nilon, mảnh xốp, rác trong suốt hoặc bị che có thể không được nhận ra. “Không detection” không có nghĩa là không có rác.

Tracking là IoU nhẹ, không phải ReID/ByteTrack và chưa bù chuyển động camera. Track ID có thể bị tách/đổi khi vật di chuyển nhanh, bị che hoặc ra khỏi khung. Số track không phải số vật duy nhất. Đường track được vẽ từ các frame lấy mẫu, không phải quỹ đạo vật lý đã hiệu chỉnh.

Mask chỉ áp dụng cho các vật model nhận ra; VLM không tạo segmentation mask. Cần bộ video/ảnh sông hồ có ground truth và fine-tune để đánh giá, cải thiện độ chính xác rác thực địa. Không suy ra khối lượng rác, diện tích m², độc tính hoặc chất lượng nước từ kết quả này.

MVP chưa có RTSP, xử lý audio, semantic embeddings, bộ benchmark rác thực địa hoặc hệ thống nhiều người dùng. Chỉ bind loopback khi chạy trên máy cá nhân. Chưa có xác thực; không công khai trực tiếp lên Internet.

## Kiểm thử và cấu trúc

```bash
.venv/bin/python -m pip install pytest
.venv/bin/python -m pytest -q
cd frontend && npm ci && npm run build
```

Trên Windows dùng `.venv\Scripts\python.exe` thay `.venv/bin/python`.

- `backend/`: FastAPI, store, pipeline, ONNX segmentation, tracking, prompts, AI adapters.
- `frontend/`: React + TypeScript + Vite; build được FastAPI phục vụ.
- `scripts/`: setup/start cho Windows/Linux, export model và đóng gói.
- `.data/`: video, thumbnail, database, models (gitignored).
- `tests/`: test workflow có inference thật, adapter tests với HTTP mock. Xem `VALIDATION.md` để phân biệt kết quả thực và mô phỏng.

Xem `THIRD_PARTY_NOTICES.md` về giấy phép model và tài sản kiểm thử. API key, video, database, dependency cache và model sinh ra local được gitignore; bản clone từ GitHub sẽ tải trọng số chính thức và export model khi chạy setup lần đầu.
