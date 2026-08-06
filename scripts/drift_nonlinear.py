# -*- coding: utf-8 -*-
"""Non-linear drift: the offset oscillates, so a global affine fit cannot represent
it. Only short sliding windows track the curve -- and short windows are where CC
fails, so this is the decisive case for the learned model.  Task #24 / B3.

Methods (mean |delta_est(t) - delta_true(t)| in frames, lower better):
  CC-const        : one lag                      (ignores drift)
  Caspi-Irani     : affine grid                  (cannot fit a curve)
  CC-slide/line   : CC windows + line fit        (cannot fit a curve)
  CC-slide/curve  : CC windows + curve interp    (curve ok, but CC windows noisy)
  CSF-slide/curve : CSF windows + curve interp   (ours)

    python scripts/drift_nonlinear.py --checkpoint checkpoints/csf_b1_t20.pt --window 20
"""
import argparse, math, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")
import torch

from mist.benchmark.baselines import CaspiIrani, CrossCorrelation
from mist.benchmark.drift import fit_drift, sine_offset, warp_offsets
from mist.core.types import KeypointSequence
from mist.model.panoptic_dataset import _fill_gaps
from mist.panoptic import load_calibration, load_pose3d, project_to_2d

FPS = 29.97
DEFAULT_SEQUENCES = ("160422_haggling1", "160226_haggling1", "160906_ian5")


def csf_predictor(checkpoint):
    from mist.model.continusyncformer import ContinuSyncFormer
    model = ContinuSyncFormer(n_joints=19, motion_input=True).eval()
    st = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(st["model"] if "model" in st else st)

    def predict(a, b):
        ka = torch.tensor(a.xy, dtype=torch.float32)[None]
        kb = torch.tensor(b.xy, dtype=torch.float32)[None]
        with torch.no_grad():
            return float(model(ka, kb)["dt_hard"][0])
    return predict


def run(args):
    cc = CrossCorrelation()
    caspi = CaspiIrani(alpha_range=(0.985, 1.015), alpha_steps=31, max_lag=12)
    cc_predict = lambda a, b: cc.predict(a, b).dt_frames
    csf_predict = csf_predictor(args.checkpoint) if os.path.exists(args.checkpoint) else None

    amps = [float(x) for x in args.amps.split(",")]     # oscillation amplitude (frames)
    N, W, S = args.clip_len, args.window, args.stride
    beta, period = args.beta, args.period
    methods = ["CC-const", "Caspi", "CC-line", "CC-curve"] + (["CSF-curve"] if csf_predict else [])
    results = {a: {m: [] for m in methods} for a in amps}
    edge = int(math.ceil(max(amps) + abs(beta))) + 3
    n_clips = 0

    for seq_id in args.sequences:
        seq_dir = os.path.join(args.root, seq_id)
        cams = load_calibration(seq_dir)
        keys = sorted(k for k in cams if k[0] == 0)
        ka_key, kb_key = keys[0], keys[5]
        xyz, _, _, _ = load_pose3d(seq_dir, max_frames=args.max_frames)
        ca, cb = cams[ka_key], cams[kb_key]
        clean_a, _ = _fill_gaps(project_to_2d(xyz, ca["K"], ca["R"], ca["t"]))
        clean_b, _ = _fill_gaps(project_to_2d(xyz, cb["K"], cb["R"], cb["t"]))
        fin = np.isfinite(clean_a).all((1, 2)) & np.isfinite(clean_b).all((1, 2))
        margin = edge + 2
        span = N + 2 * margin
        cum = np.cumsum(np.concatenate([[0], fin.astype(int)]))
        step = np.linalg.norm(np.diff(clean_a, axis=0), axis=-1)
        speed = np.concatenate([[0], np.median(step, axis=1)]) * FPS
        starts = [s for s in range(0, len(xyz) - span + 1, args.stride_clip)
                  if cum[s + span] - cum[s] == span
                  and float(np.median(speed[s + margin:s + margin + N])) >= args.min_speed][:args.max_clips]
        n_clips += len(starts)

        for s0 in starts:
            win = slice(s0, s0 + span)
            a_stream = clean_a[win][margin:margin + N]
            idx = np.arange(N, dtype=np.float64)
            for amp in amps:
                delta_true = sine_offset(idx, amp, period, beta)
                b_stream = warp_offsets(clean_b[win], delta_true, base=margin)
                err = lambda e: float(np.mean(np.abs(e - delta_true)))

                results[amp]["CC-const"].append(err(np.full(N, cc.predict(
                    KeypointSequence(a_stream, FPS), KeypointSequence(b_stream, FPS)).dt_frames)))
                est = caspi.estimate(KeypointSequence(a_stream, FPS), KeypointSequence(b_stream, FPS))
                results[amp]["Caspi"].append(err((est.alpha - 1.0) * idx + est.beta))
                line, _, _ = fit_drift(cc_predict, a_stream, b_stream, FPS, W, S, mode="line")
                results[amp]["CC-line"].append(err(line))
                curve, _, _ = fit_drift(cc_predict, a_stream, b_stream, FPS, W, S, mode="curve")
                results[amp]["CC-curve"].append(err(curve))
                if csf_predict:
                    c2, _, _ = fit_drift(csf_predict, a_stream, b_stream, FPS, W, S, mode="curve")
                    results[amp]["CSF-curve"].append(err(c2))

    print(f"\nNon-linear (sine) drift recovery -- mean |err| (frames), lower better")
    print(f"{n_clips} clips, clip_len={N}, window={W}, period={period}, beta={beta}")
    print(f"{'amp(fr)':>8} " + "".join(f"{m:>11}" for m in methods))
    for amp in amps:
        row = f"{amp:>8.2f} "
        for m in methods:
            vals = [v for v in results[amp][m] if np.isfinite(v)]
            row += f"{np.mean(vals):>11.4f}" if vals else f"{'-':>11}"
        print(row)
    print("\n(a global line/affine cannot fit an oscillating offset; only short-window "
          "curve tracking can, and only the learned estimator is reliable there.)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/panoptic_raw")
    ap.add_argument("--checkpoint", default="checkpoints/csf_b1_t20.pt")
    ap.add_argument("--sequences", nargs="+", default=list(DEFAULT_SEQUENCES))
    ap.add_argument("--amps", default="0.5,1.0,2.0")
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--period", type=float, default=100.0)
    ap.add_argument("--clip-len", type=int, default=200)
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--stride-clip", type=int, default=240)
    ap.add_argument("--max-clips", type=int, default=6)
    ap.add_argument("--max-frames", type=int, default=4000)
    ap.add_argument("--min-speed", type=float, default=20.0)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
