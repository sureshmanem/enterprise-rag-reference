"""FastAPI service: POST /ask, GET /health, and a tiny chat page at /."""

from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .config import get_settings
from .models import Answer
from .pipeline import RAG

app = FastAPI(title="rag-mini", version="0.1.0")


@lru_cache
def get_rag() -> RAG:
    return RAG(get_settings())


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


@app.get("/health")
def health() -> dict:
    rag = get_rag()
    return {"status": "ok", "chunks": rag.store.count(), "embedding_model": rag.embedder.model_id}


@app.post("/ask", response_model=Answer)
def ask(req: AskRequest) -> Answer:
    try:
        return get_rag().ask(req.question)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


CHAT_HTML = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>rag-mini</title>
<style>body{font-family:system-ui,sans-serif;max-width:720px;margin:2rem auto;padding:0 1rem}
#log div{margin:.75rem 0;padding:.75rem;border-radius:8px;background:#f3f3f3;white-space:pre-wrap}
#log .q{background:#e6f1fb}form{display:flex;gap:.5rem}input{flex:1;padding:.6rem}
small{color:#666}</style></head><body><h2>rag-mini chat</h2><div id="log"></div>
<form id="f"><input id="q" placeholder="e.g. How many leave days do I get?" autofocus>
<button>Ask</button></form><script>
const log=document.getElementById('log');
document.getElementById('f').onsubmit=async e=>{e.preventDefault();
const q=document.getElementById('q');const text=q.value.trim();if(!text)return;
log.insertAdjacentHTML('beforeend','<div class="q"></div>');log.lastChild.textContent=text;q.value='';
const r=await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},
body:JSON.stringify({question:text})});const a=await r.json();
const cites=(a.citations||[]).map(c=>`[${c.ref}] ${c.source}${c.page?', page '+c.page:''}`).join('\\n');
const d=document.createElement('div');d.textContent=a.answer+(cites?'\\n\\n'+cites:'');log.appendChild(d);};
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return CHAT_HTML
