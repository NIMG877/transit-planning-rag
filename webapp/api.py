import json
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from webapp.schemas import ChatRequest, ChatResponse
from webapp.service import run_chat, run_chat_stream

APP_TITLE = "轨道交通规划与政策智能问答助手"
STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title=APP_TITLE, version="1.0.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": APP_TITLE,
        "time": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/api/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    clean_question = payload.question.strip()
    if not clean_question:
        raise HTTPException(status_code=400, detail="问题不能为空")

    normalized_payload = payload.model_copy(update={"question": clean_question})

    try:
        return run_chat(normalized_payload)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"问答失败: {exc}") from exc


@app.post("/api/chat/stream")
def chat_stream(payload: ChatRequest) -> StreamingResponse:
    clean_question = payload.question.strip()
    if not clean_question:
        raise HTTPException(status_code=400, detail="问题不能为空")

    normalized_payload = payload.model_copy(update={"question": clean_question})

    def event_stream():
        try:
            yield from run_chat_stream(normalized_payload)
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
