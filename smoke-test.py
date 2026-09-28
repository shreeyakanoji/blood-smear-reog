"""End-to-end test of the backend (no server needed).   Run:  python smoke_test.py
Requires a trained model:  python -m backend.train


"""
import hashlib
import io
import os
import shutil
import sys

os.environ["DATABASE_URL"] = "sqlite:///smoke_test.db"     
os.environ["DATA_DIR"] = "data/smoke"
os.environ["SMEAR_API_KEYS"] = "tester:test-key"

import numpy as np
from fastapi.testclient import TestClient

import core as sc
from backend.api import app

H = {"X-API-Key": "test-key"}
client = TestClient(app)
results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  - {detail}" if detail else ""))


def npz_bytes(**arrs):
    b = io.BytesIO(); np.savez(b, **arrs); return b.getvalue()


def make_capture(seed, cond=sc.Cond(), wl=sc.EXT_WL):
    wl = np.array(wl, float)
    a = sc.acquire(sc.make_field(seed, hard=bool(cond.hard)), wl, cond, seed)
    return a, npz_bytes(stack=a["stack"], ref=a["ref"], wl=wl, M=sc.get_M("calibrated", cond, wl))


def score(cells, truth):
    """Match each API cell to the nearest true cell. Returns (correct, n) for red cells the API classified,
    and (routed_out, n) for white cells (which the malaria model must decline to classify)."""
    ok = n = wbc_out = wbc_n = 0
    for c in cells:
        d = [np.hypot(c["x"] - t["x"], c["y"] - t["y"]) for t in truth]
        t = truth[int(np.argmin(d))]
        if min(d) > t["r"] + 3:
            continue
        if t["ctype"] in sc.WBC_TYPES or t["ctype"] in ("lymph", "neut"):
            wbc_n += 1; wbc_out += (not c["in_scope"])
        elif t["ctype"] in ("rbc", "rbc_inf") and c["in_scope"]:
            n += 1; ok += (c["label"] == "infected") == (t["ctype"] == "rbc_inf")
    return ok, n, wbc_out, wbc_n


r = client.get("/health")
check("health endpoint lists a trained model", r.status_code == 200 and len(r.json()["models"]) > 0, str(r.json()))
check("missing API key rejected", client.get("/scans").status_code in (401, 422))
check("wrong API key rejected (401)", client.get("/scans", headers={"X-API-Key": "nope"}).status_code == 401)

# one scan: full round trip + audit trail
acq, raw = make_capture(seed=7)
r = client.post("/scans", headers=H, files={"file": ("cap.npz", raw)}, data={"task": "malaria"})
check("upload capture -> 200", r.status_code == 200, r.text[:200] if r.status_code != 200 else "")
if r.status_code == 200:
    body = r.json(); sid = body["id"]
    check("response carries model version, data source and disclaimer",
          all(k in body for k in ("model_version", "model_data_source")) and "Not for diagnostic" in body["disclaimer"], body["model_data_source"])
    g = client.get(f"/scans/{sid}", headers=H)
    check("stored SHA-256 matches the uploaded bytes", g.status_code == 200 and g.json()["input_sha256"] == hashlib.sha256(raw).hexdigest())
    actions = [a["action"] for a in client.get("/audit", headers=H).json()]
    check("audit log recorded create + read", "scan.create" in actions and "scan.read" in actions)
    check("scan appears in list", any(s["id"] == sid for s in client.get("/scans", headers=H).json()))

# aggregate accuracy per condition over several fields (single fields are too small to mean anything)
CONDS = [("baseline", sc.Cond()), ("dim -40%", sc.Cond(brightness=.6)), ("aged stain", sc.Cond(batch="aged")),
         ("camera B + rig B", sc.Cond(camera="B", rig="B", brightness=.85))]
print("\naccuracy of the API on red cells, 6 fields per condition (zero-shot; model trained on baseline only):")
for label, cond in CONDS:
    ok = n = wo = wn = 0; status_ok = True
    for seed in range(2000, 2006):
        a2, raw2 = make_capture(seed, cond)
        r2 = client.post("/scans", headers=H, files={"file": ("c.npz", raw2)}, data={"task": "malaria"})
        status_ok &= r2.status_code == 200
        if r2.status_code == 200:
            o, m, w1, w2 = score(r2.json()["cells"], a2["cells"]); ok += o; n += m; wo += w1; wn += w2
    print(f"      {label:18s} {ok}/{n} = {ok / max(n, 1):.2f}   |  white cells declined: {wo}/{wn}")
    check(f"scans under '{label}' succeed", status_ok and n > 0)
    if label == "baseline":
        check("baseline accuracy well above chance (>0.70)", n > 0 and ok / n > 0.70, f"{ok / max(n, 1):.2f}")
        check("gate declines most white cells (>=90%)", wn > 0 and wo / wn >= 0.90, f"{wo}/{wn}")

# bad input handling
check("garbage file -> 422", client.post("/scans", headers=H, files={"file": ("x.npz", b"not an npz")}, data={"task": "malaria"}).status_code == 422)
_, raw4 = make_capture(seed=3, wl=sc.BASE_WL)
check("wavelength mismatch -> 422", client.post("/scans", headers=H, files={"file": ("c.npz", raw4)}, data={"task": "malaria"}).status_code == 422)
check("unknown scan -> 404", client.get("/scans/does-not-exist", headers=H).status_code == 404)
check("unknown model version -> 404", client.post("/scans", headers=H, files={"file": ("c.npz", raw)}, data={"task": "malaria", "model_version": "nope"}).status_code == 404)

print(f"\n{sum(results)}/{len(results)} checks passed")
shutil.rmtree("data/smoke", ignore_errors=True)
if os.path.exists("smoke_test.db"): os.remove("smoke_test.db")
sys.exit(0 if all(results) else 1)
