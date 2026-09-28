from __future__ import annotations
import functools
from typing import NamedTuple
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.optimize import nnls
from skimage import filters, measure, morphology
from skimage.registration import phase_cross_correlation
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

BASE_WL = [450, 530, 630, 850]
EXT_WL = [450, 530, 590, 630, 700, 850, 940]
STAINS = ["eosin", "azure / methylene blue", "hemoglobin", "scatter"]
K = len(STAINS)
METHODS = ["rgb", "odrgb", "unmix"]
METHOD_NAMES = {"rgb": "Raw RGB (naive)", "odrgb": "Flat-fielded RGB (OD)", "unmix": "Unmixed stain concentrations"}


TASKS = {
    "malaria": dict(classes=["uninfected", "infected"], types=("rbc", "rbc_inf")),
    "wbc_diff": dict(classes=["lymphocyte", "neutrophil"], types=("lymph", "neut")),
}


#physics 
def _g(wl, mu, s, a):
    return a * np.exp(-0.5 * ((np.asarray(wl, float) - mu) / s) ** 2)


def stain_matrix(wl, batch="fresh"):
    """Absorption spectra M (rows = wavelengths, cols = STAINS). Synthetic, literature-like shapes."""
    wl = np.asarray(wl, float)
    e_mu, a_mu = (525, 645) if batch == "fresh" else (515, 630)
    eos = _g(wl, e_mu, 28, 1.0) + _g(wl, 490, 20, 0.3)
    azu = _g(wl, a_mu, 38, 1.0) + _g(wl, 595, 28, 0.55)
    hem = _g(wl, 415, 22, 1.6) + _g(wl, 542, 17, 0.9) + _g(wl, 577, 14, 0.85) + _g(wl, 450, 30, 0.3)
    sca = 0.175 * (550 / wl) ** 1.5
    return np.stack([eos, azu, hem, sca], 1)


def defect_spectrum(wl):
    return _g(wl, 700, 30, 0.9) + _g(wl, 590, 18, 0.5) + 0.15  


class Cond(NamedTuple):
    brightness: float = 1.0
    camera: str = "A"       # A = NoIR, B = IR-cut + noisier
    batch: str = "fresh"    # fresh / aged stain
    rig: str = "A"          # B = LED peak shifts + brightness spread
    noise: float = 1.0


RIG_OFF = {"A": {}, "B": {450: 7, 530: -9, 590: 8, 630: 6, 700: -8, 850: 12, 940: -14}}
RIG_B_LEDF = {450: .8, 530: 1.2, 590: .9, 630: 1.2, 700: .75, 850: 1.1, 940: .85}
CAM_X = [400, 450, 530, 600, 650, 700, 850, 950]
CAM_G = {"A": [.6, .85, 1, .95, .9, .8, .55, .25], "B": [.5, .75, 1, 1, .95, .5, .03, .01]}
CAM_SIG = {"A": .003, "B": .007}
BATCH_SCALE = {"fresh": np.array([1, 1, 1, 1.]), "aged": np.array([.7, .55, 1, 1.])}  # i made sure hemoglobin is not a stain here


def ref_index(wl):
    return int(np.argmin(np.abs(np.asarray(wl, float) - 530)))


def vis_idx(wl):
    return [int(np.argmin(np.abs(np.asarray(wl, float) - w))) for w in (630, 530, 450)]  # R,G,B


#simulator 
_RR = {"lymph": (9, 11), "neut": (11, 13), "eos": (11, 13), "blast": (14, 17), "hypo": (6.2, 7.8)}
CASES = {"normal": "Normal-like smear", "malaria": "Malaria-like (parasitized RBCs)", "sickle": "Sickle-cell-like",
         "thal": "Thalassemia-like (hypochromic + target cells)", "leuk": "Leukemia-like (many blasts)"}


def _case_plan(case, rng, n_rbc):
    wbc = {"normal": ["neut", "lymph", "neut"], "malaria": ["neut", "lymph"], "sickle": ["neut", "lymph", "neut"],
           "thal": ["neut", "lymph"], "leuk": ["blast"] * 8 + ["neut", "lymph"]}[case]
    mix = {"normal": {}, "malaria": {"rbc_inf": .3}, "sickle": {"sickle": .35, "target": .05},
           "thal": {"hypo": .55, "target": .15}, "leuk": {}}[case]
    rb = []
    for _ in range(n_rbc):
        u, acc, t = rng.random(), 0., "rbc"
        for k, p in mix.items():
            acc += p
            if u < acc:
                t = k; break
        rb.append(t)
    return wbc + rb


def make_field(seed, size=160, n_rbc=14, n_wbc=4, defects=False, case=None):
    """Synthetic smear. case=None keeps the legacy training-set generator; case in CASES builds a demo smear."""
    rng = np.random.default_rng(seed)
    if case:
        size, n_rbc = max(size, 240), 50
    H = W = size
    yy, xx = np.mgrid[:H, :W].astype(float)
    C = np.zeros((K, H, W)); C[3] += 0.04
    placed, cells, lab = [], [], np.zeros((H, W), int)
    gap = 2 if case is None else 1

    def place(r):
        for _ in range(300):
            x, y = rng.uniform(r + 4, W - r - 4), rng.uniform(r + 4, H - r - 4)
            if all((x - a) ** 2 + (y - b) ** 2 > (r + q + gap) ** 2 for a, b, q in placed):
                placed.append((x, y, r)); return x, y
        return None

    if case:
        plan = _case_plan(case, rng, n_rbc)
    else:
        plan = [["lymph", "neut"][i % 2] for i in range(n_wbc)] + [("rbc_inf" if rng.random() < .35 else "rbc") for _ in range(n_rbc)]
    for ct in plan:
        if case is None:
            r = {"lymph": rng.uniform(9, 11), "neut": rng.uniform(11, 13)}.get(ct, rng.uniform(8, 10))
        else:
            r = rng.uniform(*_RR.get(ct, (8, 10)))
        p = place(r * 1.3 if ct == "sickle" else r)
        if p is None:
            continue
        x, y = p
        d = np.hypot(xx - x, yy - y); m = d <= r
        if ct == "sickle":
            th = rng.uniform(0, np.pi); dx, dy = xx - x, yy - y
            ux = dx * np.cos(th) + dy * np.sin(th); uy = -dx * np.sin(th) + dy * np.cos(th)
            a_, b_ = 1.9 * r, 0.5 * r
            m = (ux / a_) ** 2 + ((uy - (0.35 / a_) * ux ** 2) / b_) ** 2 <= 1
        idx = len(cells) + 1; lab[m] = idx
        C[3][m] += 0.12
        if ct in ("rbc", "rbc_inf"):
            prof = 0.55 + 0.45 * (d / r) ** 2
            C[2][m] += 0.6 * prof[m]; C[0][m] += 0.32 * prof[m]
            if ct == "rbc_inf":
                a, o = rng.uniform(0, 2 * np.pi), rng.uniform(.15, .6) * r
                dd = np.hypot(xx - (x + o * np.cos(a)), yy - (y + o * np.sin(a)))
                C[1] += rng.uniform(.6, 1.0) * np.exp(-(dd / 2.3) ** 2) * m
                C[0][m] += 0.05
        elif ct == "sickle":
            C[2][m] += 0.62; C[0][m] += 0.32
        elif ct == "target":
            prof = 0.35 + 0.65 * (d / r) ** 2 + 0.7 * np.exp(-(d / (0.22 * r)) ** 2)
            C[2][m] += 0.6 * prof[m]; C[0][m] += 0.32 * prof[m]
        elif ct == "hypo":
            prof = 0.12 + 0.88 * (d / r) ** 4
            C[2][m] += 0.5 * prof[m]; C[0][m] += 0.25 * prof[m]
        elif ct == "lymph":
            n = d <= 0.75 * r; C[1][n] += 1.0; C[0][m & ~n] += 0.10
        elif ct == "neut":
            C[0][m] += 0.28 + 0.1 * rng.random((H, W))[m]
            for j in range(3):
                a = 2 * np.pi * j / 3 + rng.uniform(0, 1)
                C[1][np.hypot(xx - (x + .38 * r * np.cos(a)), yy - (y + .38 * r * np.sin(a))) <= .33 * r] += 0.85
        elif ct == "eos":
            C[0][m] += 0.85
            for j in range(2):
                a = np.pi * j + rng.uniform(0, 1)
                C[1][np.hypot(xx - (x + .35 * r * np.cos(a)), yy - (y + .35 * r * np.sin(a))) <= .3 * r] += 0.8
        elif ct == "blast":
            n = d <= 0.85 * r; C[1][n] += 0.9; C[1][m & ~n] += 0.25
        cells.append(dict(id=idx, x=x, y=y, r=r, ctype=ct))
    C = ndi.gaussian_filter(C, (0, .8, .8))
    D = None; dmask = np.zeros((H, W), bool)
    if defects:
        D = np.zeros((H, W))
        for _ in range(4):
            r = rng.uniform(3, 6); cx, cy = rng.uniform(10, W - 10), rng.uniform(10, H - 10)
            D += rng.uniform(.8, 1.2) * np.exp(-(np.hypot(xx - cx, yy - cy) / r) ** 2)
        dmask = D > 0.3
    return dict(C=C, labelmap=lab, cells=cells, D=D, dmask=dmask)


def render_rgb(field, scale=1.0, seed=0, srgb=True, batch="fresh"):
    """Brightfield-style RGB (uint8) of a synthetic field; optionally upsampled and sRGB-encoded ."""
    rng = np.random.default_rng(seed)
    C = field["C"] * BATCH_SCALE[batch][:, None, None]
    if scale != 1:
        C = ndi.zoom(C, (1, scale, scale), order=1)
    T = np.exp(-np.einsum("nk,khw->nhw", stain_matrix([630, 530, 450], batch), C))
    img = np.clip(T.transpose(1, 2, 0) * vignette(*C.shape[1:])[..., None] * 0.97 + rng.normal(0, .006, T.transpose(1, 2, 0).shape), 0, 1)
    if srgb:
        img = np.where(img <= .0031308, 12.92 * img, 1.055 * np.power(img, 1 / 2.4) - .055)
    return (img * 255).astype(np.uint8)


def illum(wl_nom, cond):
    wl_nom = np.asarray(wl_nom, float)
    off = np.array([RIG_OFF[cond.rig].get(int(w), 0) for w in wl_nom])
    ledf = np.array([RIG_B_LEDF.get(int(w), 1.) if cond.rig == "B" else 1. for w in wl_nom])
    gain = np.interp(wl_nom + off, CAM_X, CAM_G[cond.camera])
    return wl_nom + off, 0.55 * cond.brightness * ledf * gain


def vignette(H, W):
    y, x = np.mgrid[:H, :W]
    return 1 - 0.15 * (((x - W / 2) ** 2 + (y - H / 2) ** 2) / ((W / 2) ** 2 + (H / 2) ** 2))


def acquire(field, wl_nom, cond, seed=0, drift=True):
    """Simulate the sequential N-LED capture + a grey-card reference capture."""
    rng = np.random.default_rng(seed * 7919 + 13)
    wl_nom = np.asarray(wl_nom, float); N = len(wl_nom)
    C = field["C"]; H, W = C.shape[1:]
    wl_act, I0s = illum(wl_nom, cond)
    Mt = stain_matrix(wl_act, cond.batch)
    Cs = C * BATCH_SCALE[cond.batch][:, None, None]
    OD = np.einsum("nk,khw->nhw", Mt, Cs)
    if field.get("D") is not None:
        OD = OD + defect_spectrum(wl_act)[:, None, None] * field["D"][None]
    T = np.exp(-OD)
    ri = ref_index(wl_nom); true_shift = np.zeros((N, 2))
    if drift:
        for i in range(N):
            if i != ri:
                true_shift[i] = rng.normal(0, 1.2, 2)
                T[i] = ndi.shift(T[i], true_shift[i], order=1, mode="nearest")
    I0 = I0s[:, None, None] * vignette(H, W)[None]
    sig = CAM_SIG[cond.camera] * cond.noise

    def sense(x, avg=1):
        s = (sig + 0.004 * np.sqrt(np.clip(x, 0, None))) / np.sqrt(avg)
        return np.round(np.clip(x + rng.normal(0, 1, x.shape) * s, 0, 1) * 1023) / 1023

    return dict(stack=sense(I0 * T), ref=sense(I0 * 0.9, 8), wl=wl_nom, cond=tuple(cond), sim=True,
                field=field, true_C=Cs, labelmap=field["labelmap"], cells=field["cells"],
                dmask=field["dmask"], true_shift=true_shift)


# pipeline 
def _prep(im):
    g = ndi.gaussian_filter(np.log(np.clip(im, 1e-3, None)), 1.5)
    g = g - ndi.gaussian_filter(g, 12)               # remove vignetting / slow illumination gradient
    return (g - g.mean()) / (g.std() + 1e-9)


def register(stack, ri):
    base = _prep(stack[ri]); out = np.empty_like(stack); shifts = np.zeros((len(stack), 2))
    for i, f in enumerate(stack):
        if i == ri:
            out[i] = f; continue
        s, _, _ = phase_cross_correlation(base, _prep(f), upsample_factor=20, normalization=None)
        if np.abs(s).max() > 6:                       # implausible -> low-contrast frame. I am keeping it unshifted
            s = np.zeros(2)
        shifts[i] = s; out[i] = ndi.shift(f, s, order=1, mode="nearest")
    return out, shifts


def to_od(stack, ref, rho=0.9):
    I0 = np.clip(ref / rho, 1e-3, None)
    return -np.log(np.clip(stack / I0, 1e-3, None))


def unmix(od, M, w=None, method="lstsq"):
    N, H, W = od.shape
    w = np.ones(N) if w is None else np.asarray(w, float)
    A = np.sqrt(w)[:, None] * M; Y = np.sqrt(w)[:, None] * od.reshape(N, -1)
    if method == "nnls":
        out = np.empty((M.shape[1], Y.shape[1]))
        for j in range(Y.shape[1]):
            out[:, j] = nnls(A, Y[:, j])[0]
    else:
        out = np.clip(np.linalg.pinv(A) @ Y, 0, None)
    return out.reshape(M.shape[1], H, W)


def run_pipeline(acq, M, weighted=True, method="lstsq"):
    wl = acq["wl"]
    reg, shifts = register(acq["stack"], ref_index(wl))
    od = to_od(reg, acq["ref"])
    m = acq["ref"].mean((1, 2)); w = (m / m.max()) ** 2 if weighted else np.ones(len(wl))  # inverse-variance
    Cc = unmix(od, M, w, method)
    rec = np.einsum("nk,khw->nhw", M, Cc)
    res = np.sqrt((w[:, None, None] * (od - rec) ** 2).sum(0) / w.sum())
    return dict(reg=reg, od=od, C=Cc, res=res, shifts=shifts, w=w)


def segment(od, wl):
    i = vis_idx(wl)
    s = ndi.gaussian_filter(od[i[1]] + od[i[0]], 1.0)
    m = s > filters.threshold_otsu(s)
    m = ndi.binary_fill_holes(ndi.binary_opening(m, morphology.disk(2)))
    lab = measure.label(m)
    cells = []
    for r in measure.regionprops(lab):
        if r.area < 40:
            lab[lab == r.label] = 0; continue
        cells.append(dict(id=r.label, x=r.centroid[1], y=r.centroid[0], r=np.sqrt(r.area / np.pi), ctype="unknown"))
    return lab, cells


# calliberation (feature 4)
PATCHES = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [.5, .5, 0, .3],
                    [.4, 0, .6, .3], [0, .6, .5, .3], [.5, .4, .4, .5]], float).T  # (K, P)


@functools.lru_cache(maxsize=64)
def calibrate_M(cond_t, wl_t, seed=5):
    """Image known-stain patches under all LEDs and fit M by nonnegative least squares, row per wavelength."""
    cond = Cond(*cond_t); wl = np.array(wl_t, float); P = PATCHES.shape[1]; t = 16
    C = np.zeros((K, t, t * P))
    for p in range(P):
        C[:, :, p * t:(p + 1) * t] = PATCHES[:, p][:, None, None]
    a = acquire(dict(C=C, D=None, labelmap=None, cells=[], dmask=None), wl, cond, seed, drift=False)
    od = to_od(a["stack"], a["ref"])
    Y = np.stack([od[:, :, p * t:(p + 1) * t].mean((1, 2)) for p in range(P)], 1)  # (N,P)
    return np.stack([nnls(PATCHES.T, Y[i])[0] for i in range(len(wl))])


def get_M(mmode, cond, wl):
    return calibrate_M(tuple(cond), tuple(int(w) for w in wl)) if mmode == "calibrated" else stain_matrix(wl, "fresh")


def true_M_effective(cond, wl):
    wl_act, _ = illum(wl, cond)
    return stain_matrix(wl_act, cond.batch) * BATCH_SCALE[cond.batch][None]


@functools.lru_cache(maxsize=64)
def clean_residual(cond_t, wl_t, mmode, weighted):
    """Residual distribution from a defect-free 'clean slide' imaged on the same rig -> anomaly threshold."""
    cond = Cond(*cond_t); wl = np.array(wl_t, float)
    a = acquire(make_field(4242), wl, cond, 4242)
    return run_pipeline(a, get_M(mmode, cond, wl), weighted)["res"].ravel()


# features / datasets
def cell_feats(ch, m):
    v = ch[:, m]; mean = v.mean(1)
    return np.concatenate([mean, np.percentile(v, 90, axis=1), v.max(1), mean / (np.abs(mean).sum() + 1e-6), [m.sum()]])


def crop(ch, x, y, S=28):
    p = S // 2
    a = np.pad(ch, ((0, 0), (p, p), (p, p)), mode="edge")
    cx, cy = int(round(x)) + p, int(round(y)) + p
    return a[:, cy - p:cy + p, cx - p:cx + p]


def channels(P, wl, method):
    i = vis_idx(wl)
    return {"rgb": P["reg"][i], "odrgb": P["od"][i], "unmix": P["C"]}[method]


@functools.lru_cache(maxsize=128)
def build(seeds, cond_t, wl_t, mmode, weighted):
    cond = Cond(*cond_t); wl = np.array(wl_t, float); M = get_M(mmode, cond, wl)
    F = {m: [] for m in METHODS}; X = {m: [] for m in METHODS}; ct, res = [], []
    for s in seeds:
        a = acquire(make_field(s), wl, cond, s); P = run_pipeline(a, M, weighted)
        res.append(float(P["res"].mean()))
        chs = {m: channels(P, wl, m) for m in METHODS}
        for c in a["cells"]:
            msk = a["labelmap"] == c["id"]; ct.append(c["ctype"])
            for m in METHODS:
                F[m].append(cell_feats(chs[m], msk)); X[m].append(crop(chs[m], c["x"], c["y"]).astype(np.float32))
    return dict(ct=np.array(ct), F={m: np.array(v) for m, v in F.items()}, X={m: np.array(v) for m, v in X.items()}, res=np.array(res))


def subset(d, task, meth, kind="F"):
    t = TASKS[task]["types"]; sel = np.isin(d["ct"], t)
    return d[kind][meth][sel], (d["ct"][sel] == t[1]).astype(int)


def fit_rf(X, y):
    return RandomForestClassifier(300, random_state=0, n_jobs=-1, class_weight="balanced").fit(X, y)


def scores(y, p):
    return accuracy_score(y, p), f1_score(y, p, average="macro")


def train_models(task, wl_t, mmode, weighted, n_train):
    d = build(tuple(range(n_train)), tuple(Cond()), wl_t, mmode, weighted)
    return {m: fit_rf(*subset(d, task, m)) for m in METHODS}


COND_SET = {
    "Baseline": Cond(),
    "Dim -40%": Cond(brightness=.6),
    "Bright +40%": Cond(brightness=1.4),
    "Camera B (IR-cut, noisier)": Cond(camera="B"),
    "Aged stain batch": Cond(batch="aged"),
    "Rig B (cross-device)": Cond(camera="B", rig="B", brightness=.85),
}


def run_experiment(task, wl_t, mmode, weighted, conds, n_train=24, n_test=8):
    models = train_models(task, wl_t, mmode, weighted, n_train)
    rows, resid = [], {}
    for name, c in conds.items():
        d = build(tuple(range(1000, 1000 + n_test)), tuple(c), wl_t, mmode, weighted); resid[name] = d["res"]
        for m in METHODS:
            X, y = subset(d, task, m); acc, f1 = scores(y, models[m].predict(X))
            rows.append(dict(condition=name, method=METHOD_NAMES[m], accuracy=acc, macro_f1=f1))
    return pd.DataFrame(rows), resid


# (Feature 9, optional torch)
def torch_ok():
    try:
        import torch  # noqa
        return True
    except Exception:
        return False


def make_net(cin, kind, ncls=2):
    import torch.nn as nn
    if kind == "scratch":
        return nn.Sequential(nn.Conv2d(cin, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                             nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                             nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
                             nn.Flatten(), nn.Linear(64, ncls)), True
    import torchvision
    try:
        net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1); pre = True
    except Exception:
        net = torchvision.models.resnet18(weights=None); pre = False  
    w = net.conv1.weight.data                                        
    conv = nn.Conv2d(cin, 64, 7, 2, 3, bias=False)
    conv.weight.data = w.mean(1, keepdim=True).repeat(1, cin, 1, 1) * 3 / cin
    net.conv1 = conv; net.fc = nn.Linear(512, ncls)

    class Up(nn.Module):
        def __init__(s, n): super().__init__(); s.n = n
        def forward(s, x): return s.n(nn.functional.interpolate(x, size=64, mode="bilinear", align_corners=False))
    return Up(net), pre


def train_cnn(X, y, kind, epochs=20, seed=0):
    import torch, torch.nn.functional as F
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    mu = X.mean((0, 2, 3), keepdims=True); sd = X.std((0, 2, 3), keepdims=True) + 1e-6   
    Xt = torch.tensor((X - mu) / sd, dtype=torch.float32); yt = torch.tensor(y)
    net, pre = make_net(X.shape[1], kind)
    opt = torch.optim.Adam(net.parameters(), 1e-3 if kind == "scratch" else 3e-4)
    cw = torch.tensor(len(y) / (2 * np.bincount(y, minlength=2) + 1e-9), dtype=torch.float32)
    for _ in range(epochs):
        net.train(); perm = torch.randperm(len(yt))
        for i in range(0, len(perm) - 1, 32):
            b = perm[i:i + 32]
            if len(b) < 2:
                continue
            xb = Xt[b]
            if rng.random() < .5: xb = xb.flip(3)
            if rng.random() < .5: xb = xb.flip(2)
            xb = torch.rot90(xb, int(rng.integers(4)), (2, 3))
            loss = F.cross_entropy(net(xb), yt[b], weight=cw); opt.zero_grad(); loss.backward(); opt.step()
    net.eval()

    def predict(Xn):
        with torch.no_grad():
            return net(torch.tensor((Xn - mu) / sd, dtype=torch.float32)).argmax(1).numpy()
    return predict, pre


def run_dl(task, wl_t, mmode, weighted, conds, n_train, n_test, epochs, use_torch=True, progress=None):
    tr = build(tuple(range(n_train)), tuple(Cond()), wl_t, mmode, weighted)
    tests = {n: build(tuple(range(1000, 1000 + n_test)), tuple(c), wl_t, mmode, weighted) for n, c in conds.items()}
    rows, notes = [], []
    kinds = ["Random forest"] + (["Scratch CNN", "Adapted ResNet18"] if use_torch else [])
    jobs = [(k, inp) for inp in ("unmix", "rgb") for k in kinds]
    for j, (k, inp) in enumerate(jobs):
        if progress: progress(j / len(jobs), f"{k} on {METHOD_NAMES[inp]}")
        if k == "Random forest":
            Xtr, y = subset(tr, task, inp, "F"); mdl = fit_rf(Xtr, y); pred = mdl.predict; kx = "F"
        else:
            Xtr, y = subset(tr, task, inp, "X")
            pred, pre = train_cnn(Xtr, y, "scratch" if k.startswith("Scratch") else "adapted", epochs)
            if k.startswith("Adapted") and not pre and "pretrain" not in " ".join(notes):
                notes.append("ImageNet weights unavailable offline - adapted ResNet18 ran with random init (pretraining benefit not measured).")
            kx = "X"
        for n, d in tests.items():
            Xte, yte = subset(d, task, inp, kx); acc, f1 = scores(yte, pred(Xte))
            rows.append(dict(model=k, input=METHOD_NAMES[inp], condition=n, accuracy=acc, macro_f1=f1))
    return pd.DataFrame(rows), notes


# re-staining (Feature 8)
def _p95(v):
    m = v > 0.05 * max(v.max(), 1e-6)
    return np.percentile(v[m], 95) if m.sum() > 20 else max(v.max(), 1e-6)


@functools.lru_cache(maxsize=16)
def reference_levels(wl_t, mmode, weighted):
    wl = np.array(wl_t, float); a = acquire(make_field(777), wl, Cond(), 777)
    P = run_pipeline(a, get_M(mmode, Cond(), wl), weighted)
    return np.array([_p95(P["C"][0]), _p95(P["C"][1])])


def restain(P, ref_levels):
    """Normalise stain concentrations (eosin, azure) to reference 95th percentiles, reproject via reference M."""
    out = P["C"].copy()
    for k in (0, 1):
        out[k] = P["C"][k] * ref_levels[k] / max(_p95(P["C"][k]), 1e-6)
    Mvis = stain_matrix([630, 530, 450], "fresh")
    rgb = np.exp(-np.einsum("nk,khw->nhw", Mvis, out))       # I_syn = I0 * exp(-OD_syn), I0 = 1
    return np.clip(rgb.transpose(1, 2, 0), 0, 1)


def transmittance_rgb(P, wl):
    return np.clip(np.exp(-P["od"][vis_idx(wl)]).transpose(1, 2, 0), 0, 1)
