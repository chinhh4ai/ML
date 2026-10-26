import os
import sys
import logging
from pathlib import Path
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

DEBRIS_BACKEND_URL = os.environ.get("DEBRIS_BACKEND_URL", "http://127.0.0.1:5076")
GAUGE_BACKEND_URL = os.environ.get("GAUGE_BACKEND_URL", "http://127.0.0.1:5077")

client: httpx.AsyncClient = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global client
    client = httpx.AsyncClient(timeout=300.0)
    yield
    await client.aclose()

app = FastAPI(title="HIP Detection", lifespan=lifespan)

STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

INDEX_HTML_PATH = STATIC_DIR / "index.html"

@app.get("/", response_class=HTMLResponse)
async def get_index():
    if INDEX_HTML_PATH.exists():
        return INDEX_HTML_PATH.read_text(encoding="utf-8")
    return "<h1>HIP Detection</h1>"

# Forward iframe shells
@app.get("/view/debris", response_class=HTMLResponse)
async def view_debris():
    try:
        r = await client.get(f"{DEBRIS_BACKEND_URL}/")
        html = r.text
        # Rewrite root-relative asset and API references for clean routing
        html = html.replace('src="/assets/', 'src="/debris/assets/')
        html = html.replace('href="/assets/', 'href="/debris/assets/')
        return HTMLResponse(content=html, status_code=r.status_code)
    except Exception as e:
        return HTMLResponse(content=f"<h3>Debris service unavailable: {e}</h3>", status_code=502)

@app.get("/view/gauge", response_class=HTMLResponse)
async def view_gauge():
    try:
        r = await client.get(f"{GAUGE_BACKEND_URL}/")
        html = r.text
        html = html.replace('href="/static/', 'href="/gauge/static/')
        html = html.replace('src="/static/', 'src="/gauge/static/')
        return HTMLResponse(content=html, status_code=r.status_code)
    except Exception as e:
        return HTMLResponse(content=f"<h3>Gauge service unavailable: {e}</h3>", status_code=502)

@app.get("/gauge/static/app.js")
async def gauge_app_js():
    try:
        r = await client.get(f"{GAUGE_BACKEND_URL}/static/app.js")
        js_code = r.text
        # Replace all API endpoints to /gauge/api
        js_code = js_code.replace("'/api/", "'/gauge/api/")
        js_code = js_code.replace('"/api/', '"/gauge/api/')
        # Ensure sample thumbnails point to /gauge/api/samples
        js_code = js_code.replace('${sample.thumbnail_url}', '${sample.thumbnail_url.startsWith("/gauge") ? sample.thumbnail_url : "/gauge" + sample.thumbnail_url}')
        return Response(content=js_code, media_type="application/javascript")
    except Exception as e:
        return Response(content=str(e), status_code=502)

# --- PROXY DEBRIS ---
@app.api_route("/debris/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"])
async def debris_proxy(path: str, request: Request):
    target_url = f"{DEBRIS_BACKEND_URL}/{path}"
    if request.url.query:
        target_url += f"?{request.url.query}"
    
    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("content-length", None)
    body = await request.body()
    try:
        req = client.build_request(request.method, target_url, headers=headers, content=body)
        resp = await client.send(req, stream=True)
        return StreamingResponse(
            resp.aiter_raw(),
            status_code=resp.status_code,
            headers=dict(resp.headers),
            media_type=resp.headers.get("content-type")
        )
    except Exception as e:
        return Response(content=str(e), status_code=502)

# Also proxy root /api and /assets fallback for frontend SPAs if any root-relative calls occur
@app.api_route("/api/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"])
async def api_proxy(path: str, request: Request):
    if path.startswith("samples"):
        target_url = f"{GAUGE_BACKEND_URL}/api/{path}"
    else:
        target_url = f"{DEBRIS_BACKEND_URL}/api/{path}"
        
    if request.url.query:
        target_url += f"?{request.url.query}"
    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("content-length", None)
    body = await request.body()
    try:
        req = client.build_request(request.method, target_url, headers=headers, content=body)
        resp = await client.send(req, stream=True)
        return StreamingResponse(
            resp.aiter_raw(),
            status_code=resp.status_code,
            headers=dict(resp.headers),
            media_type=resp.headers.get("content-type")
        )
    except Exception as e:
        return Response(content=str(e), status_code=502)

@app.api_route("/assets/{path:path}", methods=["GET", "HEAD", "OPTIONS"])
async def assets_proxy(path: str, request: Request):
    target_url = f"{DEBRIS_BACKEND_URL}/assets/{path}"
    try:
        req = client.build_request(request.method, target_url)
        resp = await client.send(req, stream=True)
        return StreamingResponse(
            resp.aiter_raw(),
            status_code=resp.status_code,
            headers=dict(resp.headers),
            media_type=resp.headers.get("content-type")
        )
    except Exception as e:
        return Response(content=str(e), status_code=502)

# --- PROXY GAUGE ---
@app.api_route("/gauge/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"])
async def gauge_proxy(path: str, request: Request):
    target_url = f"{GAUGE_BACKEND_URL}/{path}"
    if request.url.query:
        target_url += f"?{request.url.query}"
    
    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("content-length", None)
    body = await request.body()
    try:
        req = client.build_request(request.method, target_url, headers=headers, content=body)
        resp = await client.send(req, stream=True)
        return StreamingResponse(
            resp.aiter_raw(),
            status_code=resp.status_code,
            headers=dict(resp.headers),
            media_type=resp.headers.get("content-type")
        )
    except Exception as e:
        return Response(content=str(e), status_code=502)
