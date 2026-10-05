import hashlib
import io
import os
import shutil
import sys

os.environ["DATABASE_URL"] = "sqlite:///smoke_test.db"     
os.environ["DATA_DIR"] = "data/smoke"
os.environ.setdefault("SMEAR_MODEL_DIR_NOTE", "uses whatever is in ./models")
os.environ["SMEAR_API_KEYS"] = "tester:test-key"

import numpy as np
from fastapi.testclient import TestClient

import core as sc
from api import app

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


check("garbage file -> 422", client.post("/scans", headers=H, files={"file": ("x.npz", b"not an npz")}, data={"task": "malaria"}).status_code == 422)
_, raw4 = make_capture(seed=3, wl=sc.BASE_WL)
check("wavelength mismatch -> 422", client.post("/scans", headers=H, files={"file": ("c.npz", raw4)}, data={"task": "malaria"}).status_code == 422)
check("unknown scan -> 404", client.get("/scans/does-not-exist", headers=H).status_code == 404)
check("unknown model version -> 404", client.post("/scans", headers=H, files={"file": ("c.npz", raw)}, data={"task": "malaria", "model_version": "nope"}).status_code == 404)

print(f"\n{sum(results)}/{len(results)} checks passed (part 1)")



print("\n-- counts + particle detection + robustness --")

acqc, rawc = make_capture(seed=7)
rc = client.post("/scans", headers=H, files={"file": ("c.npz", rawc)}, data={"task": "malaria"})
if rc.status_code == 200:
    b = rc.json()
    check("response includes counts", "counts" in b, str(b.get("counts")))
    c = b.get("counts", {})
    check("red+white cells <= total", c.get("red_cells", 0) + c.get("white_cells", 0) <= c.get("total_cells", -1))
    true_rbc = sum(1 for t in acqc["cells"] if t["ctype"] in ("rbc", "rbc_inf"))
    true_inf = sum(1 for t in acqc["cells"] if t["ctype"] == "rbc_inf")
    true_par = round(100 * true_inf / true_rbc, 1) if true_rbc else None
    got_par = c.get("parasitemia_percent")
    check("parasitemia within 15 points of ground truth (noisy on one field, by design)",
          got_par is not None and abs(got_par - true_par) < 15, f"got {got_par}, true {true_par}")
    check("small_particles field present (list, possibly empty)", isinstance(b.get("small_particles"), list))
    check("small_particles_note explicitly disclaims bacteria-count validity",
          "NOT" in b.get("small_particles_note", "") and "bacteria" in b.get("small_particles_note", "").lower())
else:
    check("counts scan succeeded", False, rc.text[:200])

# a capture with literally nothing in it (zeros) - must fail cleanly, not 500
blank = npz_bytes(stack=np.zeros((7, 160, 160)), ref=np.full((7, 160, 160), 0.5), wl=np.array(sc.EXT_WL, float), M=sc.get_M("calibrated", sc.Cond(), np.array(sc.EXT_WL, float)))
rb = client.post("/scans", headers=H, files={"file": ("blank.npz", blank)}, data={"task": "malaria"})
check("blank/empty-field capture fails with a clean 4xx (not 500)", 400 <= rb.status_code < 500, f"got {rb.status_code}: {rb.text[:150]}")


nanstack = acqc["stack"].copy(); nanstack[0, 0, 0] = np.nan
badnan = npz_bytes(stack=nanstack, ref=acqc["ref"], wl=np.array(sc.EXT_WL, float), M=sc.get_M("calibrated", sc.Cond(), np.array(sc.EXT_WL, float)))
rn = client.post("/scans", headers=H, files={"file": ("nan.npz", badnan)}, data={"task": "malaria"})
check("NaN-contaminated capture handled without a 500", rn.status_code != 500, f"got {rn.status_code}")


rt = client.post("/scans", headers=H, files={"file": ("c.npz", rawc)}, data={"task": "not_a_real_task"})
check("unknown task name -> clean 4xx, not 500", 400 <= rt.status_code < 500, f"got {rt.status_code}: {rt.text[:150]}")


re_ = client.post("/scans", headers=H, files={"file": ("empty.npz", b"")}, data={"task": "malaria"})
check("completely empty file -> clean 4xx, not 500", 400 <= re_.status_code < 500, f"got {re_.status_code}")


badM = npz_bytes(stack=acqc["stack"], ref=acqc["ref"], wl=np.array(sc.EXT_WL, float), M=np.ones((3, 2)))
rm = client.post("/scans", headers=H, files={"file": ("badm.npz", badM)}, data={"task": "malaria"})
check("malformed M shape -> clean 4xx, not 500", 400 <= rm.status_code < 500, f"got {rm.status_code}")

print(f"\n{sum(results)}/{len(results)} checks passed (full run)")


shutil.rmtree("data/smoke", ignore_errors=True)
if os.path.exists("smoke_test.db"): os.remove("smoke_test.db")
sys.exit(0 if all(results) else 1)
