"""FastAPI backend for the replay dashboard (T8.2; architecture.md §2.7).

    uvicorn crowdsafe.server.app:app --port 8000

REST: /api/runs, /api/runs/{id}/layout, /api/runs/{id}/frame (camera image), /api/runs/{id}/summary.
WebSocket: /ws/replay/{id}?speed=10&start=0 streams the per-second messages (speed = replay seconds per real second).
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):   # low-RAM laptop: one BLAS thread
    os.environ.setdefault(_v, "1")

import asyncio

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response

from crowdsafe.server import replay

app = FastAPI(title="CrowdSafe replay")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
                   allow_methods=["*"], allow_headers=["*"])


def _check(run_id: str) -> None:
    if run_id not in {r["id"] for r in replay.list_runs()}:
        raise HTTPException(404, f"unknown run {run_id}")


@app.get("/api/runs")
def runs() -> list[dict]:
    return replay.list_runs()


@app.get("/api/runs/{run_id}/layout")
def layout(run_id: str) -> dict:
    _check(run_id)
    return replay.layout(run_id)


@app.get("/api/runs/{run_id}/frame")
def frame(run_id: str) -> Response:
    _check(run_id)
    return Response(replay.frame_jpeg(run_id), media_type="image/jpeg")


@app.get("/api/runs/{run_id}/video")
def video(run_id: str) -> FileResponse:
    """MP4 of the run (supports range requests, so the browser can seek)."""
    _check(run_id)
    p = replay.video_path(run_id)
    if p is None:
        raise HTTPException(404, "no video for this run")
    return FileResponse(p, media_type="video/mp4")


@app.get("/api/runs/{run_id}/summary")
def summary(run_id: str) -> dict:
    """Duration, peak density and all alerts of the run (for the timeline bar)."""
    _check(run_id)
    tl = replay.timeline(run_id)
    peaks = [max((z["density"] or 0) for z in m["zones"]) for m in tl]
    seen, alerts = set(), []
    for m in tl:
        for a in m["alerts"]:
            key = (a["zone_id"], a["since"])
            if key not in seen:
                seen.add(key)
                alerts.append(a)
    return {"start": tl[0]["timestamp"], "end": tl[-1]["timestamp"], "n": len(tl), "peak_density": round(max(peaks), 2),
            "peaks": [round(p, 2) for p in peaks], "times": [m["timestamp"] for m in tl], "alerts": alerts}


@app.websocket("/ws/replay/{run_id}")
async def ws_replay(ws: WebSocket, run_id: str, speed: float = 10.0, start: float = 0.0) -> None:
    await ws.accept()
    try:
        _check(run_id)
        tl = replay.timeline(run_id)
        delay = 1.0 / max(speed, 0.1)
        for m in tl:
            if m["timestamp"] < start:
                continue
            await ws.send_json(m)
            await asyncio.sleep(delay)
        await ws.send_json({"done": True})
        await ws.close()
    except WebSocketDisconnect:
        pass
    except HTTPException as e:
        await ws.send_json({"error": e.detail})
        await ws.close()
