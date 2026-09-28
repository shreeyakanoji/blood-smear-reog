"""FastAPI backend.   Run from project root:  uvicorn backend.api:app --reload
Interactive docs at http://localhost:8000/docs
"""
import hashlib
import io
import os
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from sqlalchemy import select

import core as sc
from . import registry
from .db import Audit, Scan, Session, init_db

DISCLAIMER = "Research use only. Not for diagnostic use."
DATA_DIR = Path(os.getenv("DATA_DIR", "data/scans"))
DATA_DIR.mkdir(parents=True, exist_ok=True)


def _keys():
   
    raw = os.getenv("SMEAR_API_KEYS", "dev:dev-key")
    return {k: u for u, k in (p.split(":", 1) for p in raw.split(",") if ":" in p)}


KEYS = _keys()


def actor(x_api_key: str = Header(...)):
    if x_api_key not in KEYS:
        raise HTTPException(401, "invalid API key")
    return KEYS[x_api_key]


def audit(s, who, action, scan_id=None, **detail):
    s.add(Audit(actor=who, action=action, scan_id=scan_id, detail=detail))


app = FastAPI(title="Multispectral smear scanner API", version="0.1")
init_db()


@app.get("/health")
def health():
    return {"status": "ok", "models": [m["version"] for m in registry.list_versions()]}


@app.get("/models")
def models(who: str = Depends(actor)):
    return registry.list_versions()


@app.post("/scans")
async def create_scan(file: UploadFile = File(...), task: str = Form("malaria"),
                      model_version: Optional[str] = Form(None), who: str = Depends(actor)):
    """Upload an .npz capture: stack (N,H,W), ref (N,H,W), wl (N), optional M (N,K)."""
    raw = await file.read()
    if len(raw) > 200_000_000:
        raise HTTPException(413, "file too large")
    try:
        z = np.load(io.BytesIO(raw))   # allow_pickle stays False: never enable it for uploaded files
        stack, ref, wl = z["stack"].astype(float), z["ref"].astype(float), z["wl"].astype(float)
        M = z["M"] if "M" in z.files else sc.stain_matrix(wl)
        assert stack.ndim == 3 and ref.shape == stack.shape and len(wl) == len(stack), "shape mismatch"
        assert M.shape == (len(wl), sc.K), f"M must be ({len(wl)}, {sc.K})"
    except Exception as e:
        raise HTTPException(422, f"bad capture file: {e}")

    try:
        mdl, meta = registry.load_model(model_version, task)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    if tuple(int(w) for w in wl) != tuple(meta["wavelengths"]):
        raise HTTPException(422, f"capture wavelengths {list(map(int, wl))} differ from model's {meta['wavelengths']}")

    P = sc.run_pipeline(dict(stack=stack, ref=ref, wl=wl), M, meta["weighted"])
    lab, cells = sc.segment(P["od"], wl)
    med = np.median(P["res"]); mad = 1.4826 * np.median(np.abs(P["res"] - med))
    flag = P["res"] > med + 5 * mad                      # residual anomaly map (self-referenced)
    ch = sc.channels(P, wl, "unmix")
    out = []
    for c in cells:
        m = lab == c["id"]
        p = float(mdl["unmix"].predict_proba(sc.cell_feats(ch, m)[None])[0, 1])
        ff = float(flag[m].mean())
        out.append(dict(cell=int(c["id"]), x=round(float(c["x"]), 1), y=round(float(c["y"]), 1),
                        p_positive=round(p, 4), label=meta["classes"][int(p >= .5)],
                        flagged_fraction=round(ff, 4), needs_review=ff > 0.2))

    sha = hashlib.sha256(raw).hexdigest()
    result = dict(cells=out, model_version=meta["version"], model_data_source=meta["data_source"],
                  wavelengths=[int(w) for w in wl], disclaimer=DISCLAIMER)
    with Session() as s:
        scan = Scan(actor=who, task=task, model_version=meta["version"], input_sha256=sha,
                    n_cells=len(out), flagged_fraction=float(flag.mean()), result=result)
        s.add(scan); s.flush()
        (DATA_DIR / f"{scan.id}.npz").write_bytes(raw)
        audit(s, who, "scan.create", scan.id, sha256=sha, model=meta["version"])
        s.commit()
        return dict(id=scan.id, **result)


@app.get("/scans/{scan_id}")
def get_scan(scan_id: str, who: str = Depends(actor)):
    with Session() as s:
        scan = s.get(Scan, scan_id)
        if scan is None:
            raise HTTPException(404, "scan not found")
        audit(s, who, "scan.read", scan_id); s.commit()
        return dict(id=scan.id, created=scan.created, actor=scan.actor, input_sha256=scan.input_sha256, **scan.result)


@app.get("/scans")
def list_scans(limit: int = 50, who: str = Depends(actor)):
    with Session() as s:
        rows = s.scalars(select(Scan).order_by(Scan.created.desc()).limit(min(limit, 200))).all()
        return [dict(id=r.id, created=r.created, task=r.task, model_version=r.model_version,
                     n_cells=r.n_cells, flagged_fraction=r.flagged_fraction) for r in rows]


@app.get("/audit")
def get_audit(limit: int = 100, who: str = Depends(actor)):
    with Session() as s:
        rows = s.scalars(select(Audit).order_by(Audit.id.desc()).limit(min(limit, 1000))).all()
        return [dict(ts=r.ts, actor=r.actor, action=r.action, scan_id=r.scan_id, detail=r.detail) for r in rows]
