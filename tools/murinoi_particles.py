"""Murinoi — loop cinematográfico de partículas 3D (renderizador próprio, CPU/numpy).

Uso: python3 murinoi_particles.py W H FPS SECONDS OUT [frames_csv_for_stills]
"""
import sys, os, subprocess, multiprocessing as mp
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

W, H, FPS, SECS = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
OUT = sys.argv[5]
STILLS = [int(x) for x in sys.argv[6].split(",")] if len(sys.argv) > 6 else None
NF = int(round(FPS * SECS))
SCALE = W / 3840.0
rng = np.random.default_rng(7)

MINT = np.array([0.49, 0.96, 0.77], np.float32)
CYAN = np.array([0.35, 0.75, 1.00], np.float32)
SILVER = np.array([0.86, 0.88, 0.92], np.float32)

def rot(yaw, pitch, roll=0.0):
    cy_, sy_ = np.cos(yaw), np.sin(yaw); cp, sp = np.cos(pitch), np.sin(pitch); cr, sr = np.cos(roll), np.sin(roll)
    Ry = np.array([[cy_, 0, sy_], [0, 1, 0], [-sy_, 0, cy_]])
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return (Ry @ Rx @ Rz).astype(np.float32)

# ---------------------------------------------------------------- logo geometry
FONT = "/usr/share/fonts/opentype/inter/InterDisplay-ExtraBold.otf"
font = ImageFont.truetype(FONT, 640)
txt = "murinoi."
bb = font.getbbox(txt)
pad = 40
mw, mh = bb[2] - bb[0] + 2 * pad, bb[3] - bb[1] + 2 * pad
def mask_of(s):
    im = Image.new("L", (mw, mh), 0)
    ImageDraw.Draw(im).text((pad - bb[0], pad - bb[1]), s, font=font, fill=255)
    return np.asarray(im, np.float32) / 255.0
m_all = mask_of(txt)
m_word = mask_of("murinoi")
m_dot = np.clip(m_all - m_word, 0, 1)

LOGO_W = 4.4
px = LOGO_W / mw
DEPTH = 0.42
cx, cy = mw / 2, mh / 2

def to_world(xp, yp):
    return (xp - cx) * px, -(yp - cy) * px

N_FRONT, N_BACK, N_SIDE = 300_000, 60_000, 170_000
inside = np.argwhere(m_all > 0.5)
def sample_inside(n):
    k = inside[rng.integers(0, len(inside), n)]
    yp = k[:, 0] + rng.random(n); xp = k[:, 1] + rng.random(n)
    return xp, yp

xf, yf = sample_inside(N_FRONT)
xb, yb = sample_inside(N_BACK)
# side walls: edge pixels with outward normals from blurred mask gradient
mb = cv2.GaussianBlur(m_all, (0, 0), 2.0)
gy, gx = np.gradient(mb)
edge = np.argwhere((m_all > 0.5) & (cv2.erode((m_all > 0.5).astype(np.uint8), np.ones((3, 3))) == 0))
k = edge[rng.integers(0, len(edge), N_SIDE)]
ys_ = k[:, 0] + rng.random(N_SIDE); xs_ = k[:, 1] + rng.random(N_SIDE)
nx_ = -gx[k[:, 0], k[:, 1]]; ny_ = gy[k[:, 0], k[:, 1]]  # world y is flipped
nl = np.sqrt(nx_ ** 2 + ny_ ** 2) + 1e-6
nx_, ny_ = nx_ / nl, ny_ / nl
zs_ = (rng.random(N_SIDE) - 0.5) * DEPTH
# rounded bevel: tilt normals toward ±z near the faces
t = np.clip((np.abs(zs_) / (DEPTH / 2) - 0.55) / 0.45, 0, 1)
bev = t * t * (3 - 2 * t) * 0.85
nz_ = np.sign(zs_) * bev
sc = np.sqrt(1 - bev ** 2)
nx_, ny_ = nx_ * sc, ny_ * sc

wxf, wyf = to_world(xf, yf); wxb, wyb = to_world(xb, yb); wxs, wys = to_world(xs_, ys_)
HOME = np.concatenate([
    np.stack([wxf, wyf, np.full(N_FRONT, DEPTH / 2)], 1),
    np.stack([wxb, wyb, np.full(N_BACK, -DEPTH / 2)], 1),
    np.stack([wxs, wys, zs_], 1)]).astype(np.float32)
NORM = np.concatenate([
    np.tile([0, 0, 1.0], (N_FRONT, 1)) + rng.normal(0, 0.06, (N_FRONT, 3)),
    np.tile([0, 0, -1.0], (N_BACK, 1)),
    np.stack([nx_, ny_, nz_], 1)]).astype(np.float32)
NORM /= np.linalg.norm(NORM, axis=1, keepdims=True)
allx = np.concatenate([xf, xb, xs_]).astype(int).clip(0, mw - 1)
ally = np.concatenate([yf, yb, ys_]).astype(int).clip(0, mh - 1)
IS_DOT = (m_dot[ally, allx] > 0.5).astype(np.float32)
NL = len(HOME)

# ---------------------------------------------------------------- per-particle choreography
r = rng.random
sx = (HOME[:, 0] - HOME[:, 0].min()) / np.ptp(HOME[:, 0])
U_OUT = 0.17 + 0.15 * np.clip(sx + rng.normal(0, 0.05, NL), 0, 1)          # dissolve sweeps left → right
rad = np.sqrt(HOME[:, 0] ** 2 + HOME[:, 1] ** 2); rad /= rad.max()
U_IN = 0.58 + 0.20 * np.clip(0.55 * r(NL) + 0.45 * (1 - rad), 0, 1)        # reassemble, rim → centre
DUR_OUT = 0.10 + 0.05 * r(NL)
DUR_IN = 0.11 + 0.04 * r(NL)

# nebula target: flattened swirling galaxy around the logo, some particles fly at the lens
arm = rng.integers(0, 3, NL)
sarm = r(NL) ** 1.35
ang = arm * (2 * np.pi / 3) + sarm * 3.4 * np.pi + rng.normal(0, 0.05, NL)
radn = 0.35 + 5.0 * sarm
spread = 0.025 + 0.11 * sarm
NEB = np.stack([radn * np.cos(ang), rng.normal(0, 1, NL) * spread * 0.6, radn * np.sin(ang)], 1)
NEB[:, [0, 2]] += rng.normal(0, 1, (NL, 2)) * spread[:, None]
halo = r(NL) < 0.12  # loose halo stars
NEB[halo] = rng.normal(0, 1, (halo.sum(), 3)) * np.array([4.5, 1.6, 4.5])
lens = r(NL) < 0.02
NEB[lens] = np.stack([(r(lens.sum()) - 0.5) * 3, (r(lens.sum()) - 0.5) * 1.6, 3.0 + 3.0 * r(lens.sum())], 1)
NEB = NEB.astype(np.float32)
NEB_R = np.sqrt(NEB[:, 0] ** 2 + NEB[:, 2] ** 2)
TILT = rot(0.0, 0.62, 0.18)  # galaxy disc tilted toward the lens
SWIRL = (1.3 / (0.8 + 0.5 * NEB_R)).astype(np.float32) * (0.8 + 0.4 * r(NL)).astype(np.float32)
ARC = rng.normal(0, 1, (NL, 3)).astype(np.float32) * 0.9
ARC[:, 2] = np.abs(ARC[:, 2]) * 1.4
PHASE = (r((NL, 3)) * 2 * np.pi).astype(np.float32)
TW_F = (3 + 9 * r(NL)).astype(np.float32); TW_P = (r(NL) * 2 * np.pi).astype(np.float32)
GLOW = np.where((r(NL) < 0.6)[:, None], MINT, CYAN).astype(np.float32)
GLOW = np.where((r(NL) < 0.12)[:, None], np.array([1.0, 0.95, 0.88], np.float32), GLOW)
BASE = np.where(IS_DOT[:, None] > 0, MINT * 1.15, SILVER).astype(np.float32)
SIZE = (0.6 + 0.8 * r(NL)).astype(np.float32)
STAR = (0.15 + 3.0 * r(NL) ** 6).astype(np.float32)

# ambient dust motes (periodic drift → seamless loop)
ND = 5000
DUST = np.stack([(r(ND) - 0.5) * 18, (r(ND) - 0.5) * 9, -7 + 12 * r(ND)], 1).astype(np.float32)
DUST_A = (rng.normal(0, 0.25, (ND, 3))).astype(np.float32)
DUST_P = (r((ND, 3)) * 2 * np.pi).astype(np.float32)
DUST_C = np.where((r(ND) < 0.5)[:, None], MINT, np.array([0.9, 0.85, 0.75], np.float32)) * (0.15 + 0.5 * r(ND) ** 3)[:, None]
DUST_C = DUST_C.astype(np.float32)

# ---------------------------------------------------------------- frame-independent buffers
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
dxn, dyn = (xx - W / 2) / (W / 2), (yy - H / 2) / (H / 2)
rr = np.sqrt(dxn ** 2 * 1.0 + dyn ** 2 * 0.6)
BG = (np.array([0.006, 0.007, 0.011], np.float32)
      + np.exp(-rr ** 2 * 2.2)[..., None] * np.array([0.006, 0.016, 0.019], np.float32))
VIG = (1 - 0.55 * np.clip(rr - 0.35, 0, None) ** 1.5)[..., None].astype(np.float32)
del yy, xx, dxn, dyn, rr

def ss(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t)

def ease(x):  # smooth in-out (quintic-ish)
    x = np.clip(x, 0, 1); return x * x * x * (x * (x * 6 - 15) + 10)

def splat(bw, bh, x, y, col):
    """Bilinear additive splat of N points into a bh×bw×3 buffer."""
    ok = (x >= 0) & (y >= 0) & (x < bw - 1) & (y < bh - 1)
    x, y, col = x[ok], y[ok], col[ok]
    ix = x.astype(np.int32); iy = y.astype(np.int32)
    fx = (x - ix)[:, None]; fy = (y - iy)[:, None]
    i0 = iy * bw + ix
    idx = np.concatenate([i0, i0 + 1, i0 + bw, i0 + bw + 1])
    wcol = np.concatenate([col * (1 - fx) * (1 - fy), col * fx * (1 - fy), col * (1 - fx) * fy, col * fx * fy])
    out = np.empty((bh * bw, 3), np.float32)
    for c in range(3):
        out[:, c] = np.bincount(idx, weights=wcol[:, c], minlength=bh * bw)
    return out.reshape(bh, bw, 3)

EXPO = float(os.environ.get("EXPO", 0.22))
FLY = float(os.environ.get("FLY", 16.0))
STREAK = float(os.environ.get('STREAK', 0.6))
COC_EDGES = [0, 2.5, 5, 10, 20, 40, 80, 1e9]

def render(fi):
    u = (fi / NF) % 1.0
    tau = 2 * np.pi * u
    # ---------------- logo transform
    R = rot(0.38 * np.sin(tau) + 0.05 * np.sin(3 * tau), 0.07 * np.sin(2 * tau + 0.6), 0.02 * np.sin(tau + 1.0))
    # ---------------- per-particle dissolve amount d ∈ [0,1]
    d = ease((u - U_OUT) / DUR_OUT) * (1 - ease((u - U_IN) / DUR_IN))
    home = HOME @ R.T
    nrm = NORM @ R.T
    # nebula: swirl around Y, breathing turbulence
    th = SWIRL * 2 * np.pi * 0.55 * (u - U_OUT)
    c, s = np.cos(th), np.sin(th)
    neb = np.empty_like(NEB)
    neb[:, 0] = NEB[:, 0] * c + NEB[:, 2] * s
    neb[:, 2] = -NEB[:, 0] * s + NEB[:, 2] * c
    neb[:, 1] = NEB[:, 1] + 0.18 * np.sin(2 * tau * 2 + PHASE[:, 1] + NEB[:, 0])
    neb[:, 0] += 0.15 * np.sin(tau * 3 + PHASE[:, 0])
    neb[:, 2] += 0.15 * np.sin(tau * 3 + PHASE[:, 2])
    neb = np.where(lens[:, None], neb, neb @ TILT.T)
    dd = d[:, None]
    pos = home + (neb - home) * dd + ARC * np.sin(np.pi * dd) * 0.8
    # micro-shimmer even when assembled
    pos += 0.0025 * np.sin(tau * 40 + PHASE).astype(np.float32)

    # ---------------- camera (periodic dolly + orbit)
    push = np.sin(np.pi * u) ** 2
    cam = np.array([0.55 * np.sin(tau), 0.18 + 0.25 * np.sin(tau + 0.8), 7.4 - 2.4 * push], np.float32)
    tgt = np.array([0.0, 0.0, 0.0], np.float32)
    f = tgt - cam; f /= np.linalg.norm(f)
    rgt = np.cross(f, [0, 1, 0]); rgt /= np.linalg.norm(rgt)
    up = np.cross(rgt, f)
    fov = np.radians(34 + 6 * push)
    focal = (H / 2) / np.tan(fov / 2)
    zf = float(np.linalg.norm(cam - tgt))

    # ---------------- shading (logo particles)
    V = cam - pos; V /= np.linalg.norm(V, axis=1, keepdims=True)
    la = 0.9 * np.sin(tau * 2 + 0.3)
    L1 = np.array([np.sin(la) * 0.8, 0.55, np.cos(la) * 0.8], np.float32); L1 /= np.linalg.norm(L1)
    L2 = np.array([0.7, 0.15, -0.7], np.float32); L2 /= np.linalg.norm(L2)
    ndl = np.clip(nrm @ L1, 0, None)
    Hh = V + L1; Hh /= np.linalg.norm(Hh, axis=1, keepdims=True)
    spec = np.clip(np.sum(nrm * Hh, 1), 0, None) ** 48
    ndv = np.sum(nrm * V, 1)
    fres = (1 - np.clip(ndv, 0, 1)) ** 3
    rim = np.clip(nrm @ L2, 0, None)
    occl = 0.18 + 0.82 * ss(-0.15, 0.35, ndv)
    shaded = (BASE * (0.05 + 1.0 * ndl)[:, None]
              + (spec * 4.0)[:, None] * np.array([1.0, 0.98, 0.95], np.float32)
              + (fres * 0.9 + rim * 0.7)[:, None] * MINT) * occl[:, None]
    # lock-in pulse and dot glow
    pulse = float(ss(0.86, 0.90, u) * (1 - ss(0.90, 0.99, u)))
    shaded = shaded * (1 + 0.9 * pulse) + (IS_DOT * (0.35 + 0.8 * pulse))[:, None] * MINT
    tw = 0.65 + 0.35 * np.sin(TW_F * tau * 6 + TW_P)
    flying = GLOW * (FLY * tw * STAR)[:, None]
    col = shaded * 1.3 * (1 - dd) + flying * dd
    energy = (0.62 + 0.25 * np.sin(np.pi * d)) * SIZE  # flight sparkle
    col = col * energy[:, None]

    # ---------------- dust
    dpos = DUST + DUST_A * np.sin(tau + DUST_P)
    allp = np.concatenate([pos, dpos])
    allc = np.concatenate([col, DUST_C * (0.8 + 0.2 * np.sin(tau * 4 + DUST_P[:, 0]))[:, None]])

    # ---------------- projection
    rel = allp - cam
    zc = rel @ f; xc = rel @ rgt; yc = rel @ up
    vis = zc > 0.35
    zc, xc, yc, allc = zc[vis], xc[vis], yc[vis], allc[vis]
    sx_ = W / 2 + focal * xc / zc
    sy_ = H / 2 - focal * yc / zc
    coc = 95.0 * SCALE * np.abs(1 / zf - 1 / zc) * focal / (H / 2) * 1.0
    coc_bins = np.digitize(coc / SCALE, COC_EDGES) - 1
    # per-pixel energy normalised to 4K density so previews match the master
    allc = allc * (SCALE ** 2) * EXPO

    img = BG.copy()
    for b in range(len(COC_EDGES) - 1):
        sel = coc_bins == b
        if not sel.any():
            continue
        lo, hi = COC_EDGES[b], min(COC_EDGES[b + 1], 160)
        sigma = 0.5 * (lo + hi) * 0.5 * SCALE if b > 0 else 0.55 * SCALE
        ds = 1 if sigma < 4 else (2 if sigma < 12 else 4)
        bw, bh = W // ds, H // ds
        buf = splat(bw, bh, sx_[sel] / ds, sy_[sel] / ds, allc[sel] * (1.0 if b < 3 else 1.6) / ds ** 2 * ds ** 2)
        sg = max(sigma / ds, 0.5)
        buf = cv2.GaussianBlur(buf, (0, 0), sg)
        if ds > 1:
            buf = cv2.resize(buf, (W, H), interpolation=cv2.INTER_LINEAR) / (ds * ds)
        img += buf

    # ---------------- bloom + anamorphic streak (quarter res)
    q = cv2.resize(img, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    br = np.clip(q - 0.9, 0, None)
    bloom = (cv2.GaussianBlur(br, (0, 0), 3 * SCALE * 4 / 4 + 1) * 0.5
             + cv2.GaussianBlur(br, (0, 0), 12 * SCALE + 2) * 0.45
             + cv2.GaussianBlur(br, (0, 0), 40 * SCALE + 4) * 0.35)
    lum = br.mean(axis=2, keepdims=True)
    kw = int(W // 4 * 0.35) | 1
    streak = cv2.blur(cv2.blur(lum, (kw, 1)), (kw // 2 | 1, 1))
    streak = cv2.GaussianBlur(streak, (0, 0), 0.8)[..., None] if streak.ndim == 2 else streak
    streak = streak.reshape(H // 4, W // 4, 1) * np.array([0.30, 0.75, 1.0], np.float32) * STREAK
    glow = cv2.resize(bloom + streak, (W, H), interpolation=cv2.INTER_LINEAR)
    img = img + glow

    # ---------------- grade: ACES, vignette, chromatic aberration, grain
    x = img * 1.05
    x = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)
    x = np.clip(x, 0, 1) * VIG
    x = x ** (1 / 2.2)
    ca = 0.0009
    for ch, k_ in ((0, 1 + ca), (2, 1 - ca)):
        M = np.array([[k_, 0, W / 2 * (1 - k_)], [0, k_, H / 2 * (1 - k_)]], np.float32)
        x[..., ch] = cv2.warpAffine(np.ascontiguousarray(x[..., ch]), M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    g = np.random.default_rng(fi).standard_normal((H // 2, W // 2), dtype=np.float32)
    g = cv2.resize(g, (W, H), interpolation=cv2.INTER_LINEAR)[..., None]
    x = x * 255 + g * 2.2
    return np.clip(x[..., ::-1][..., ::-1] + 0.5, 0, 255).astype(np.uint8)

if __name__ == "__main__":
    cv2.setNumThreads(1)
    print(f"particles={NL} frames={NF} {W}x{H}@{FPS}", flush=True)
    if STILLS is not None:
        for fi in STILLS:
            Image.fromarray(render(fi)).save(f"{OUT}_{fi:04d}.png")
            print("still", fi, flush=True)
        sys.exit()
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                           "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-profile:v", "high",
                           "-pix_fmt", "yuv420p", "-x264-params", "aq-mode=3", "-tune", "grain",
                           "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
                           "-movflags", "+faststart", OUT], stdin=subprocess.PIPE)
    with mp.get_context("fork").Pool(int(os.environ.get("WORKERS", 4))) as pool:
        for i, fr in enumerate(pool.imap(render, range(NF), chunksize=1)):
            ff.stdin.write(fr.tobytes())
            if i % 60 == 0:
                print("frame", i, flush=True)
    ff.stdin.close(); ff.wait()
    print("done", OUT)
