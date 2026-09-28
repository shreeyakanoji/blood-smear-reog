"""Multispectral Blood Smear Scanner """
import io, time, zipfile
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import core as sc

st.set_page_config(page_title="Multispectral Smear Scanner", layout="wide")

ARDUINO_SKETCH = r'''// LED sequencer. Serial 115200. Commands: "L<i>[,duty]\n" = only LED i on (duty 0-255), "X\n" = all off.
const int PINS[] = {3, 5, 6, 9, 10, 11, 4};   // 7 LEDs via NPN transistors (pin 4 = on/off only)
const int N = 7;
void allOff(){ for(int i=0;i<N;i++){ digitalWrite(PINS[i], LOW);} }
void setup(){ Serial.begin(115200); for(int i=0;i<N;i++) pinMode(PINS[i], OUTPUT); allOff(); }
void loop(){
  if(!Serial.available()) return;
  String s = Serial.readStringUntil('\n'); s.trim();
  if(s == "X"){ allOff(); Serial.println("OK"); }
  else if(s.charAt(0) == 'L'){
    int comma = s.indexOf(','); int i = s.substring(1, comma < 0 ? s.length() : comma).toInt();
    int duty = comma < 0 ? 255 : s.substring(comma + 1).toInt();
    allOff(); if(i >= 0 && i < N){ if(i == 6) digitalWrite(PINS[i], duty > 0); else analogWrite(PINS[i], duty); }
    Serial.println("OK");
  }
}
'''


def show(fig):
    st.pyplot(fig); plt.close(fig)


def strip(imgs, titles, cmap="gray", vmin=None, vmax=None, h=2.4):
    fig, ax = plt.subplots(1, len(imgs), figsize=(2.3 * len(imgs), h))
    ax = np.atleast_1d(ax)
    for a, im, t in zip(ax, imgs, titles):
        a.imshow(im, cmap=cmap, vmin=vmin, vmax=vmax); a.set_title(t, fontsize=8); a.axis("off")
    fig.tight_layout(); return fig


def cond_widget(key, default=None):
    d = default or sc.Cond()
    c = st.columns(5)
    b = c[0].slider("LED brightness", .3, 1.6, d.brightness, .05, key=key + "b")
    cam = c[1].selectbox("Camera", ["A", "B"], index=["A", "B"].index(d.camera), key=key + "c",
                         help="A = NoIR sensor. B = IR-cut filter, noisier, different spectral response")
    bt = c[2].selectbox("Stain batch", ["fresh", "aged"], index=["fresh", "aged"].index(d.batch), key=key + "s")
    rig = c[3].selectbox("LED rig", ["A", "B"], index=["A", "B"].index(d.rig), key=key + "r",
                         help="B = different LED peak wavelengths / brightness spread")
    nz = c[4].slider("Noise x", .5, 4., d.noise, .25, key=key + "n")
    return sc.Cond(b, cam, bt, rig, nz)


# ------------------------------- sidebar -------------------------------
st.sidebar.title("Scanner configuration")
preset = st.sidebar.radio("LED set", ["Extended 7-LED", "Base 4-LED", "Custom"])
if preset == "Custom":
    wl = sorted(st.sidebar.multiselect("Wavelengths (nm)", sc.EXT_WL, default=sc.BASE_WL) or sc.BASE_WL)
else:
    wl = sc.EXT_WL if preset.startswith("Ext") else sc.BASE_WL
WL_T = tuple(int(w) for w in wl)
mmode = st.sidebar.radio("Unmixing matrix M", ["literature", "calibrated"],
                         help="calibrated = fitted by NNLS from known-stain reference patches imaged on the same rig/batch")
weighted = st.sidebar.checkbox("SNR-weighted unmixing", True, help="Weights each LED by its measured signal (inverse noise variance)")
kthr = st.sidebar.slider("Anomaly threshold (std above clean-slide mean)", 3., 10., 5., .5)
Mlit = sc.stain_matrix(wl)
st.sidebar.metric("cond(M)", f"{np.linalg.cond(Mlit):.1f}", help="Lower = better-conditioned unmixing")
st.sidebar.caption(f"N = {len(wl)} wavelengths vs K = {sc.K} components -> "
                   + ("exactly determined: residual is ~0, no anomaly signal" if len(wl) == sc.K else
                      "over-determined: residual usable" if len(wl) > sc.K else "UNDER-determined"))

tabs = st.tabs(["Overview", "1 Acquire", "2 Pipeline + anomalies", "3 Calibration", "4 Classifier + invariance",
                "5 Deep learning + tasks", "6 Digital re-staining", "7 Live demo"])


with tabs[0]:
    st.title("Multispectral blood smear scanner")
    st.markdown("""
Sequential narrow-band LED captures -> registration -> flat-field -> optical density (Beer Lambert' law is followed here.) -> linear stain unmixing ->
residual anomaly map + classifiers + digital re-staining. Every feature works on a **built in physics- I have listed the laws used in repo's readme-  simulator

| Feature | Where |
|---|---|
| Hardware capture + Arduino sketch | 1 Acquire |
| Registration, flat-field, OD, unmixing, residual | 2 Pipeline |
| F1 Diagnostic classifier vs raw-RGB baseline | 4 |
| F2 Invariance experiment | 4 |
| F3 Extra channels (change LED set in sidebar) | 2, sidebar |
| F4 Calibrated M | 3 |
| F5 Residual anomaly detection | 2 |
| F6 Cross-device zero-shot transfer (Rig B) | 4 |
| F7 Live demo | 7 |
| F8 Digital re-staining | 6 |
| F9 CNNs + extensible multi-task registry | 5 |
""")
    st.warning("Simulator results demonstrate the *method's mechanics*, not the clinical performance. The simulator's physics "
               "(spectra, noise, drift) is a model I wrote; real slides will behave worse in ways it cannot show. "
               "Use prepared, fixed educational slides only.")
    st.caption("Registered tasks (add a dict entry in core.TASKS + a dataset to add a task; upstream stages are untouched): "
               + ", ".join(f"{k} {v['classes']}" for k, v in sc.TASKS.items()))

# ------------------------------- acquire -------------------------------
with tabs[1]:
    src = st.radio("Data source", ["Simulator", "Upload capture", "Live hardware"], horizontal=True)
    if src == "Simulator":
        c0, c1, c2 = st.columns([1, 1, 1])
        seed = c0.number_input("Field seed", 0, 9999, 3)
        defects = c1.checkbox("Add dust / precipitate defects", True)
        drift = c2.checkbox("Simulate slide drift between frames", True)
        cond = cond_widget("acq")
        acq = sc.acquire(sc.make_field(int(seed), defects=defects), wl, cond, int(seed), drift=drift); acq["seed"] = int(seed)
    elif src == "Upload capture":
        st.markdown("**Option A** - `.npz` with `stack (N,H,W)`, `wl (N)`, `ref (N,H,W)`, optional `M (N,K)` (your calibrated matrix).  \n"
                    "**Option B** - one grayscale image per wavelength plus a reference (grey card) image per wavelength.")
        f = st.file_uploader("NPZ capture", type=["npz"])
        imgs = st.file_uploader("Sample images (in wavelength order)", type=["png", "tif", "tiff", "jpg"], accept_multiple_files=True)
        refs = st.file_uploader("Reference images (same order)", type=["png", "tif", "tiff", "jpg"], accept_multiple_files=True)
        wtxt = st.text_input("Wavelengths for Option B", ",".join(map(str, wl)))

        def load(fs):
            out = []
            for u in fs:
                a = np.array(Image.open(u)); a = a.mean(2) if a.ndim == 3 else a
                out.append(a / (65535. if a.max() > 255 else 255.))
            return np.array(out)
        try:
            if f is not None:
                z = np.load(f); st.session_state.real = dict(stack=z["stack"].astype(float), ref=z["ref"].astype(float), wl=z["wl"].astype(float),
                                                            M=z["M"] if "M" in z else None, sim=False, cond=None)
            elif imgs and refs and len(imgs) == len(refs):
                st.session_state.real = dict(stack=load(imgs), ref=load(refs), wl=np.array([float(x) for x in wtxt.split(",")]), M=None, sim=False, cond=None)
        except Exception as e:
            st.error(f"Could not load capture: {e}")
        acq = st.session_state.get("real")
        if acq is None:
            st.info("Upload a capture to continue (or switch to Simulator)."); acq = None
    else:
        st.markdown("Wire each LED through an NPN transistor to the Arduino pins in the sketch. Lock camera exposure/gain - auto-exposure silently breaks flat-fielding.")
        st.download_button("Download Arduino sketch", ARDUINO_SKETCH, "led_sequencer.ino")
        h1, h2, h3, h4 = st.columns(4)
        port = h1.text_input("Serial port", "/dev/ttyACM0"); cam_idx = h2.number_input("Camera index", 0, 9, 0)
        settle = h3.number_input("LED settle (ms)", 5, 2000, 150); expo = h4.number_input("Exposure (us)", 100, 200000, 20000)
        use_pi = st.checkbox("Use picamera2 (Pi NoIR module)", False); navg = st.slider("Frames averaged per LED", 1, 16, 4)

        def capture():
            import serial
            ser = serial.Serial(port, 115200, timeout=3); time.sleep(2)
            if use_pi:
                from picamera2 import Picamera2
                cam = Picamera2(); cam.configure(cam.create_still_configuration(main={"format": "RGB888"}))
                cam.set_controls({"AeEnable": False, "ExposureTime": int(expo), "AnalogueGain": 1.0}); cam.start()
                grab = lambda: cam.capture_array().astype(float).mean(2) / 255.
            else:
                import cv2
                cam = cv2.VideoCapture(int(cam_idx)); cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1); cam.set(cv2.CAP_PROP_EXPOSURE, float(expo))
                grab = lambda: cam.read()[1].astype(float).mean(2) / 255.
            frames, bar = [], st.progress(0.)
            for i, w in enumerate(wl):
                ser.write(f"L{i}\n".encode()); ser.readline(); time.sleep(settle / 1000)
                grab(); frames.append(np.mean([grab() for _ in range(navg)], 0)); bar.progress((i + 1) / len(wl), f"{w} nm")
            ser.write(b"X\n"); ser.close(); return np.array(frames)
        b1, b2 = st.columns(2)
        if b1.button("Capture reference (grey card / blank field)"):
            try: st.session_state.hw_ref = capture(); st.success("Reference stored")
            except Exception as e: st.error(f"Hardware capture failed: {e}")
        if b2.button("Capture sample stack"):
            try: st.session_state.hw_stack = capture(); st.success("Stack stored")
            except Exception as e: st.error(f"Hardware capture failed: {e}")
        if "hw_ref" in st.session_state and "hw_stack" in st.session_state:
            st.session_state.real = dict(stack=st.session_state.hw_stack, ref=st.session_state.hw_ref, wl=np.array(wl, float), M=None, sim=False, cond=None)
            buf = io.BytesIO(); np.savez(buf, stack=st.session_state.hw_stack, ref=st.session_state.hw_ref, wl=np.array(wl))
            st.download_button("Save capture (.npz)", buf.getvalue(), "capture.npz")
        acq = st.session_state.get("real")
        if acq is None: st.info("Capture a reference and a sample stack to continue.")
    if acq is not None:
        fig = strip(list(acq["stack"]), [f"{int(w)} nm" for w in acq["wl"]], vmin=0, vmax=1)
        st.caption("Raw frames (one per LED)"); show(fig)

# shared pipeline result
P = M = cond_t = None
if acq is not None:
    if acq["sim"]:
        cond_t = sc.Cond(*acq["cond"])
        M = sc.get_M(mmode, cond_t, acq["wl"])
    else:
        M = acq["M"] if (acq.get("M") is not None and mmode == "calibrated") else sc.stain_matrix(acq["wl"])
    use_nnls = st.session_state.get("nnls", False)
    P = sc.run_pipeline(acq, M, weighted, "nnls" if use_nnls else "lstsq")

# pipeline 
with tabs[2]:
    if P is None:
        st.info("Load data in tab 1.")
    else:
        twl = acq["wl"]; nl = [f"{int(w)} nm" for w in twl]
        st.checkbox("Use per-pixel NNLS (slower strictly nonnegative)", key="nnls")
        st.subheader("Registration")
        if acq["sim"]:
            st.dataframe(pd.DataFrame({"wavelength": nl, "true drift (y,x)": [tuple(np.round(-s, 2)) for s in acq["true_shift"]],
                                       "estimated (y,x)": [tuple(np.round(s, 2)) for s in P["shifts"]]}), hide_index=True)
        else:
            st.dataframe(pd.DataFrame({"wavelength": nl, "estimated shift (y,x)": [tuple(np.round(s, 2)) for s in P["shifts"]]}), hide_index=True)
        st.caption("Low-contrast frames (far red / NIR) can fail to register; implausible shifts (>6 px) are rejected and left unshifted.")
        st.subheader("Optical density  OD = -log(I / I0)")
        show(strip(list(P["od"]), nl, cmap="magma", vmin=0, vmax=1.2))
        st.subheader("Unmixed stain concentrations")
        show(strip(list(P["C"]), sc.STAINS, cmap="viridis", vmin=0, vmax=1))
        if acq["sim"]:
            r = [np.corrcoef(P["C"][k].ravel(), acq["true_C"][k].ravel())[0, 1] for k in range(sc.K)]
            st.caption("Pearson r vs ground-truth concentration: " + ", ".join(f"{s} {v:.2f}" for s, v in zip(sc.STAINS, r)))
        st.subheader("Residual anomaly map  ||OD - M c||")
        if acq["sim"]:
            cr = sc.clean_residual(acq["cond"], tuple(int(w) for w in twl), mmode, weighted)
            thr = cr.mean() + kthr * cr.std()
        else:
            thr = np.median(P["res"]) + kthr * 1.4826 * np.median(np.abs(P["res"] - np.median(P["res"])))
            st.caption("No clean-slide capture available: threshold from this image's own robust statistics.")
        flag = P["res"] > thr
        rgb = sc.transmittance_rgb(P, twl); ov = rgb.copy(); ov[flag] = [1, 0, 0]
        c1, c2, c3 = st.columns(3)
        c1.image(rgb, "Flat-fielded transmittance (visible LEDs)", clamp=True)
        f2, a2 = plt.subplots(figsize=(3.2, 3.2)); a2.imshow(P["res"], cmap="inferno"); a2.axis("off"); c2.pyplot(f2); plt.close(f2)
        c3.image(ov, f"Flagged: {flag.mean() * 100:.2f}% of pixels", clamp=True)
        if acq["sim"] and acq["dmask"].any():
            d = acq["dmask"]
            st.metric("Defect recall / false-positive rate", f"{(flag & d).sum() / d.sum():.2f} / {(flag & ~d).sum() / (~d).sum():.3f}")
            if st.button("Compare 4-LED vs 7-LED on this exact field"):
                cols = st.columns(2)
                for col, wset in zip(cols, (sc.BASE_WL, sc.EXT_WL)):
                    a_ = sc.acquire(acq["field"], wset, cond_t, acq["seed"])
                    Mx = sc.get_M(mmode, cond_t, wset); Px = sc.run_pipeline(a_, Mx, weighted)
                    crx = sc.clean_residual(acq["cond"], tuple(wset), mmode, weighted); fx = Px["res"] > crx.mean() + kthr * crx.std()
                    col.image(np.clip(Px["res"] / max(thr, 1e-6) / 2, 0, 1), f"{len(wset)} LEDs | cond(M)={np.linalg.cond(sc.stain_matrix(wset)):.1f} | "
                              f"recall {(fx & d).sum() / d.sum():.2f}, FPR {(fx & ~d).sum() / (~d).sum():.3f}", clamp=True)
        if not acq["sim"]:
            lab, cells = sc.segment(P["od"], twl)
            rows = [dict(cell=c["id"], x=round(c["x"]), y=round(c["y"]), **{f"c_{s}": float(P["C"][k][lab == c["id"]].mean()) for k, s in enumerate(sc.STAINS)},
                         flagged_frac=float(flag[lab == c["id"]].mean())) for c in cells]
            df = pd.DataFrame(rows); st.subheader("Per-cell stain features (Otsu segmentation)"); st.dataframe(df)
            st.download_button("Download CSV", df.to_csv(index=False), "cells.csv")
            st.caption("Classifier training needs labeled captures; export this table and train on your own labels or benchmark on public data in tab 4.")

#  calibration 
with tabs[3]:
    st.subheader("Calibrated vs literature unmixing matrix (Feature 4)")
    cc = cond_widget("cal")
    Mc = sc.calibrate_M(tuple(cc), WL_T); Mt = sc.true_M_effective(cc, wl)
    fig, ax = plt.subplots(1, sc.K, figsize=(11, 2.6))
    for k in range(sc.K):
        ax[k].plot(wl, Mlit[:, k], "o--", label="literature"); ax[k].plot(wl, Mc[:, k], "s-", label="calibrated")
        ax[k].plot(wl, Mt[:, k], "k:", label="truth (sim)"); ax[k].set_title(sc.STAINS[k], fontsize=8); ax[k].set_xlabel("nm", fontsize=7)
    ax[0].legend(fontsize=6); fig.tight_layout(); show(fig)
    st.caption("Patches = pure eosin, pure azure, hemoglobin, blank, and 4 mixtures, imaged under all LEDs; M fitted row-by-row with NNLS. "
               "Patches must come from the same stain batch as the slides. The 'truth' curve exists only in the simulator.")
    if st.button("Compare literature vs calibrated M on downstream results"):
        rows = []
        for mm in ("literature", "calibrated"):
            df, res = sc.run_experiment("malaria", WL_T, mm, weighted, {"selected condition": cc}, 16, 6)
            r = df[df.method == sc.METHOD_NAMES["unmix"]].iloc[0]
            rows.append(dict(M=mm, macro_f1=r.macro_f1, accuracy=r.accuracy, mean_residual=float(res["selected condition"].mean())))
        st.dataframe(pd.DataFrame(rows), hide_index=True)

# classifier + invariance 
with tabs[4]:
    st.subheader("Invariance validation and cross-device transfer (Features 1, 2, 6)")
    c1, c2, c3 = st.columns(3)
    task = c1.selectbox("Task", list(sc.TASKS))
    ntr = c2.slider("Training fields (baseline rig only)", 8, 48, 24); nte = c3.slider("Test fields per condition", 4, 16, 8)
    sel = st.multiselect("Conditions", list(sc.COND_SET), default=list(sc.COND_SET))
    if st.button("Run experiment", type="primary"):
        with st.spinner("Simulating, unmixing, training..."):
            df, resid = sc.run_experiment(task, WL_T, mmode, weighted, {k: sc.COND_SET[k] for k in sel}, ntr, nte)
        st.session_state.exp = (df, resid, task)
    if "exp" in st.session_state:
        df, resid, tk = st.session_state.exp
        st.caption(f"Task: {tk} | LEDs: {WL_T} | M: {mmode} | trained ONLY on Baseline; every other condition is zero-shot (no retraining).")
        fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
        for a, met in zip(ax, ("accuracy", "macro_f1")):
            pv = df.pivot(index="condition", columns="method", values=met).loc[[s for s in sel if s in set(df.condition)]]
            pv.plot.bar(ax=a, rot=25, legend=a is ax[0]); a.set_ylim(0, 1.05); a.set_title(met); a.set_xlabel("")
        fig.tight_layout(); show(fig)
        fig, a = plt.subplots(figsize=(6, 2.6)); a.boxplot(list(resid.values()), labels=list(resid), vert=True); a.set_ylabel("mean residual"); a.tick_params(axis="x", rotation=25, labelsize=7)
        a.set_title("Unmixing residual per condition (stable = calibration working)", fontsize=9); fig.tight_layout(); show(fig)
        u = df[df.method == sc.METHOD_NAMES["unmix"]].set_index("condition").macro_f1
        bad = [k for k in u.index if u[k] < u.get("Baseline", 1) - .05]
        st.info("Where the unmixed method still degrades (>0.05 macro-F1 below baseline): " + (", ".join(f"{k} ({u[k]:.2f})" for k in bad) if bad else "none in this run")
                + ". Try the 4-LED set, camera B or literature M to see the failure modes; report them honestly.")
        st.dataframe(df.round(3), hide_index=True)
    with st.expander("Benchmark on public data before hardware exists (NIH malaria cell images, RGB only)"):
        z = st.file_uploader("ZIP containing Parasitized/ and Uninfected/ folders", type=["zip"])
        if z is not None and st.button("Run 5-fold benchmark"):
            from sklearn.model_selection import cross_val_predict
            X, y = [], []
            with zipfile.ZipFile(z) as zf:
                for n in zf.namelist():
                    l = n.lower()
                    lab = 0 if "uninfect" in l else 1 if "parasit" in l else None
                    if lab is None or not l.endswith((".png", ".jpg", ".jpeg")) or len(y) >= 1500: continue
                    im = np.asarray(Image.open(zf.open(n)).convert("RGB").resize((48, 48)), float).transpose(2, 0, 1) / 255
                    X.append(sc.cell_feats(im, im.sum(0) > .15)); y.append(lab)
            p = cross_val_predict(sc.RandomForestClassifier(300, random_state=0, n_jobs=-1, class_weight="balanced"), np.array(X), np.array(y), cv=5)
            a_, f_ = sc.scores(np.array(y), p); st.success(f"{len(y)} images | accuracy {a_:.3f} | macro-F1 {f_:.3f} (raw-RGB baseline only)")

# ------------------------------- deep learning -------------------------------
with tabs[5]:
    st.subheader("CNNs on unmixed channels + extensible tasks (Feature 9)")
    st.markdown("Upstream stages (capture -> register -> flat-field -> OD -> unmix) are shared; a task is only a label rule + a small classifier head. "
                "Prove one task rigorously (malaria) before adding another (`wbc_diff`).")
    st.code("# core.py - add a task in one line, nothing upstream changes\nTASKS['my_task'] = dict(classes=['a','b'], types=('ctype_a','ctype_b'))", "python")
    if not sc.torch_ok():
        st.warning("PyTorch not installed - only the classical model will run. `pip install torch torchvision` for the scratch CNN and adapted ResNet18.")
    d1, d2, d3, d4 = st.columns(4)
    dtask = d1.selectbox("Task", list(sc.TASKS), key="dt"); dep = d2.slider("Epochs", 5, 60, 20)
    dtr = d3.slider("Training fields", 12, 48, 24, key="dtr"); dte = d4.slider("Test fields", 4, 12, 6, key="dte")
    dsel = st.multiselect("Test conditions", list(sc.COND_SET), default=["Baseline", "Dim -40%", "Aged stain batch", "Rig B (cross-device)"], key="dsel")
    if st.button("Train + evaluate (three-way comparison)", type="primary"):
        bar = st.progress(0.)
        df, notes = sc.run_dl(dtask, WL_T, mmode, weighted, {k: sc.COND_SET[k] for k in dsel}, dtr, dte, dep, sc.torch_ok(), lambda f, t: bar.progress(f, t))
        bar.empty(); st.session_state.dl = (df, notes)
    if "dl" in st.session_state:
        df, notes = st.session_state.dl
        for n in notes: st.warning(n)
        df["series"] = df.model + " | " + df.input
        fig, a = plt.subplots(figsize=(11, 3.8))
        df.pivot(index="condition", columns="series", values="macro_f1").plot.bar(ax=a, rot=20); a.set_ylim(0, 1.05); a.legend(fontsize=6, ncol=2)
        fig.tight_layout(); show(fig); st.dataframe(df.drop(columns="series").round(3), hide_index=True)

#  re-staining 
with tabs[6]:
    st.subheader("Digital re-staining (Feature 8)")
    if acq is None:
        st.info("Load data in tab 1 (simulator: choose 'aged' stain batch and dim LEDs for a dramatic result).")
    else:
        refl = sc.reference_levels(WL_T, mmode, weighted) if acq["sim"] else np.array([1., 1.])
        if not acq["sim"]:
            refl = np.array([st.number_input("Reference p95 eosin", .1, 3., 1.), st.number_input("Reference p95 azure", .1, 3., 1.)])
        before = sc.transmittance_rgb(P, acq["wl"]); after = sc.restain(P, refl)
        cols = st.columns(3 if acq["sim"] else 2)
        cols[0].image(before, "As captured (flat-fielded)", clamp=True); cols[1].image(after, "Re-stained from unmixed concentrations", clamp=True)
        if acq["sim"]:
            ref_acq = sc.acquire(acq["field"], acq["wl"], sc.Cond(), acq["seed"])
            Pr = sc.run_pipeline(ref_acq, sc.get_M(mmode, sc.Cond(), acq["wl"]), weighted)
            cols[2].image(sc.transmittance_rgb(Pr, acq["wl"]), "Same field, well-stained (ground truth)", clamp=True)
        st.caption("Eosin and azure concentrations are rescaled to reference 95th percentiles, then reprojected through the reference M and "
                   "I = exp(-OD). Hemoglobin and scatter are intrinsic, so they are left untouched.")

#  live demo 
with tabs[7]:
    st.subheader("Live degradation demo (Feature 7)")
    ltask = st.selectbox("Task", list(sc.TASKS), key="lt")
    lc = cond_widget("live"); lseed = st.number_input("Field seed", 0, 9999, 2001, key="ls")

    @st.cache_resource(show_spinner="Training classifiers on baseline rig...")
    def models(task, wl_t, mm, w): return sc.train_models(task, wl_t, mm, w, 24)
    mdl = models(ltask, WL_T, mmode, weighted)

    def run_field(seed, cond):
        a = sc.acquire(sc.make_field(seed), wl, cond, seed); Pp = sc.run_pipeline(a, sc.get_M(mmode, cond, wl), weighted)
        out = {}
        for m in sc.METHODS:
            chs = sc.channels(Pp, wl, m); res = []
            for c in a["cells"]:
                if c["ctype"] not in sc.TASKS[ltask]["types"]: continue
                f = sc.cell_feats(chs, a["labelmap"] == c["id"])[None]
                res.append((c, int(mdl[m].predict(f)[0]), int(c["ctype"] == sc.TASKS[ltask]["types"][1])))
            out[m] = res
        return a, out

    a, out = run_field(int(lseed), lc)
    cols = st.columns(2)
    for col, m in zip(cols, ("rgb", "unmix")):
        fig, ax = plt.subplots(figsize=(4.2, 4.2)); ax.imshow(np.clip(a["stack"][sc.vis_idx(wl)].transpose(1, 2, 0), 0, 1)); ax.axis("off")
        for c, p, y in out[m]:
            ax.add_patch(plt.Circle((c["x"], c["y"]), c["r"] + 2, fill=False, ec="lime" if p == y else "red", lw=1.5))
        acc = np.mean([p == y for _, p, y in out[m]]) if out[m] else float("nan")
        ax.set_title(f"{sc.METHOD_NAMES[m]}: {acc:.0%} correct", fontsize=9); col.pyplot(fig); plt.close(fig)
    st.caption("Image shown is what an RGB camera would record (no flat-field). Green ring = correct, red = wrong.")
    if st.button("Auto-sweep: dim the LEDs step by step"):
        ph, xs, ys = st.empty(), [], {m: [] for m in sc.METHODS}
        for b in (1.4, 1.0, .8, .6, .45, .35):
            accs = {m: [] for m in sc.METHODS}
            for s in range(3010, 3013):
                _, o = run_field(s, lc._replace(brightness=b))
                for m in sc.METHODS: accs[m] += [p == y for _, p, y in o[m]]
            xs.append(b); [ys[m].append(np.mean(accs[m])) for m in sc.METHODS]
            fig, ax = plt.subplots(figsize=(6, 2.8))
            for m in sc.METHODS: ax.plot(xs, ys[m], "o-", label=sc.METHOD_NAMES[m])
            ax.set_xlabel("LED brightness"); ax.set_ylabel("accuracy"); ax.set_ylim(0, 1.05); ax.legend(fontsize=6); ph.pyplot(fig); plt.close(fig)
