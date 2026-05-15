"""
vision/trajectory_estimator.py
────────────────────────────────
Predicts the future trajectory of tracked objects.

Two complementary approaches:
  1. Kinematic predictor   — Kalman smoother + constant-velocity extrapolation.
                             Zero extra params.  Works for rigid objects.
  2. Learned predictor     — LSTM / Transformer over track history →
                             predicted N future positions.
                             More accurate for dynamic, interaction-heavy scenes.

The output is a list of (x, y, z) predictions + uncertainty estimates,
which feed directly into the action-generator to plan interception/placement.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    _TORCH = True
except ImportError:
    _TORCH = False

from vision.object_tracker import KalmanTrack


# ─────────────────────────────────────────────────────────
# Data containers
# ─────────────────────────────────────────────────────────

@dataclass
class TrajectoryPrediction:
    """Predicted future positions for one object track."""
    track_id:     int
    colour:       str
    shape:        str
    positions:    List[Tuple[float, float, float]]   # future (x,y,z)
    uncertainties: List[float]                        # 1-σ uncertainty (m)
    horizon_s:    float                               # total predicted time
    dt:           float                               # time step between predictions

    @property
    def final_position(self) -> Tuple[float, float, float]:
        return self.positions[-1] if self.positions else (0, 0, 0)

    @property
    def num_steps(self) -> int:
        return len(self.positions)

    @property
    def timestamps(self) -> List[float]:
        return [i * self.dt for i in range(1, self.num_steps + 1)]

    def position_at(self, t: float) -> Tuple[float, float, float]:
        """Interpolated position at time t (seconds from now)."""
        if not self.positions:
            return (0, 0, 0)
        idx   = t / (self.dt + 1e-9)
        lo    = int(np.floor(idx))
        hi    = min(lo + 1, len(self.positions) - 1)
        frac  = idx - lo
        lo    = max(0, min(lo, len(self.positions) - 1))
        p0, p1 = self.positions[lo], self.positions[hi]
        return tuple(float(p0[i] + frac * (p1[i] - p0[i])) for i in range(3))

    def to_numpy(self) -> np.ndarray:
        return np.array(self.positions, dtype=np.float32)


# ─────────────────────────────────────────────────────────
# 1. Kinematic predictor (Kalman extrapolation)
# ─────────────────────────────────────────────────────────

class KinematicPredictor:
    """
    Constant-velocity Kalman extrapolation.
    Uses the track's existing Kalman state (pos + vel) to project forward.
    """

    def predict(
        self,
        track:     KalmanTrack,
        horizon_s: float = 1.0,
        dt:        float = 0.1,
    ) -> TrajectoryPrediction:
        """
        Extrapolate track forward for horizon_s seconds at dt intervals.
        """
        n_steps = max(1, int(horizon_s / dt))
        x  = track.x[:6].copy()   # [px, py, pz, vx, vy, vz]
        F3 = np.array([            # 3-D kinematic step
            [1, 0, 0, dt, 0,  0 ],
            [0, 1, 0, 0,  dt, 0 ],
            [0, 0, 1, 0,  0,  dt],
            [0, 0, 0, 1,  0,  0 ],
            [0, 0, 0, 0,  1,  0 ],
            [0, 0, 0, 0,  0,  1 ],
        ], dtype=np.float64)

        P  = track.P[:6, :6].copy()
        Q  = np.eye(6) * 0.001     # small process noise for extrapolation

        positions     = []
        uncertainties = []

        for _ in range(n_steps):
            x = F3 @ x
            P = F3 @ P @ F3.T + Q
            positions.append((float(x[0]), float(x[1]), float(x[2])))
            uncertainties.append(float(np.sqrt(np.trace(P[:3, :3]) / 3)))

        return TrajectoryPrediction(
            track_id      = track.id,
            colour        = track.colour,
            shape         = track.shape,
            positions     = positions,
            uncertainties = uncertainties,
            horizon_s     = horizon_s,
            dt            = dt,
        )

    def predict_batch(
        self,
        tracks:    List[KalmanTrack],
        horizon_s: float = 1.0,
        dt:        float = 0.1,
    ) -> List[TrajectoryPrediction]:
        return [self.predict(t, horizon_s, dt) for t in tracks]


# ─────────────────────────────────────────────────────────
# 2. Learned predictor (LSTM)
# ─────────────────────────────────────────────────────────

if _TORCH:

    class LSTMTrajectoryModel(nn.Module):
        """
        LSTM-based trajectory predictor.

        Input:  (B, T_hist, 6)  — past [x,y,z,vx,vy,vz]
        Output: (B, T_fut, 6)   — future [x,y,z,vx,vy,vz] + uncertainty

        Architecture:
          Encoder LSTM → Decoder LSTM (seq-to-seq)
        """

        def __init__(
            self,
            input_dim:   int = 6,
            hidden_dim:  int = 128,
            num_layers:  int = 2,
            pred_steps:  int = 10,
            dropout:     float = 0.1,
        ):
            super().__init__()
            self.pred_steps = pred_steps
            self.hidden_dim = hidden_dim

            self.encoder = nn.LSTM(
                input_dim, hidden_dim, num_layers,
                batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
            )
            self.decoder = nn.LSTM(
                hidden_dim, hidden_dim, num_layers,
                batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
            )
            self.out_mean  = nn.Linear(hidden_dim, 3)    # (x, y, z)
            self.out_sigma = nn.Linear(hidden_dim, 3)    # log-var per axis

            # Learnable start token for decoder
            self.start_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        def forward(
            self,
            history: "torch.Tensor",            # (B, T, 6)
        ) -> Tuple["torch.Tensor", "torch.Tensor"]:
            """
            Returns
            ───────
            means  : (B, pred_steps, 3) predicted positions
            sigmas : (B, pred_steps, 3) uncertainty (positive)
            """
            B = history.size(0)
            _, (h, c) = self.encoder(history)

            dec_in = self.start_token.expand(B, -1, -1)   # (B, 1, H)
            means, sigmas = [], []

            for _ in range(self.pred_steps):
                out, (h, c) = self.decoder(dec_in, (h, c))
                m  = self.out_mean(out)               # (B, 1, 3)
                s  = torch.exp(self.out_sigma(out))   # (B, 1, 3)
                means.append(m)
                sigmas.append(s)
                dec_in = out   # autoregressive

            means  = torch.cat(means,  dim=1)   # (B, T_fut, 3)
            sigmas = torch.cat(sigmas, dim=1)
            return means, sigmas

        def compute_loss(
            self,
            pred_means:  "torch.Tensor",   # (B, T, 3)
            pred_sigmas: "torch.Tensor",   # (B, T, 3)
            targets:     "torch.Tensor",   # (B, T, 3)
        ) -> "torch.Tensor":
            """Gaussian NLL loss."""
            var  = pred_sigmas ** 2 + 1e-6
            nll  = 0.5 * (torch.log(var) + (targets - pred_means) ** 2 / var)
            return nll.mean()


class LearnedPredictor:
    """
    Wraps LSTMTrajectoryModel for inference with a history buffer.

    Parameters
    ──────────
    hist_len   : number of past timesteps fed to the LSTM
    pred_steps : number of future steps predicted
    dt         : time between steps (seconds)
    """

    def __init__(
        self,
        hist_len:   int   = 10,
        pred_steps: int   = 10,
        dt:         float = 0.1,
        hidden_dim: int   = 128,
        checkpoint: Optional[str] = None,
    ):
        self.hist_len   = hist_len
        self.pred_steps = pred_steps
        self.dt         = dt
        self._model     = None

        if _TORCH:
            self._model = LSTMTrajectoryModel(
                input_dim=6, hidden_dim=hidden_dim,
                num_layers=2, pred_steps=pred_steps,
            ).eval()
            if checkpoint:
                try:
                    state = torch.load(checkpoint, map_location="cpu")
                    self._model.load_state_dict(state)
                except Exception as e:
                    warnings.warn(f"Could not load checkpoint: {e}")

    def predict(
        self,
        track:  KalmanTrack,
        fallback: Optional[KinematicPredictor] = None,
    ) -> TrajectoryPrediction:
        """
        Predict future trajectory for one track.
        Falls back to kinematic predictor if history is too short or
        model is unavailable.
        """
        history = self._build_history(track)

        if self._model is None or len(track.history) < 3:
            fb = fallback or KinematicPredictor()
            return fb.predict(track, horizon_s=self.pred_steps * self.dt,
                              dt=self.dt)

        with torch.no_grad():
            h_t  = torch.tensor(history, dtype=torch.float32).unsqueeze(0)
            means, sigmas = self._model(h_t)
            means  = means.squeeze(0).cpu().numpy()    # (T, 3)
            sigmas = sigmas.squeeze(0).cpu().numpy()   # (T, 3)

        positions     = [tuple(means[i].tolist()) for i in range(len(means))]
        uncertainties = [float(sigmas[i].mean()) for i in range(len(sigmas))]

        return TrajectoryPrediction(
            track_id      = track.id,
            colour        = track.colour,
            shape         = track.shape,
            positions     = positions,
            uncertainties = uncertainties,
            horizon_s     = self.pred_steps * self.dt,
            dt            = self.dt,
        )

    def predict_batch(
        self,
        tracks: List[KalmanTrack],
    ) -> List[TrajectoryPrediction]:
        return [self.predict(t) for t in tracks]

    def _build_history(self, track: KalmanTrack) -> np.ndarray:
        """Build (hist_len, 6) history array from track.history."""
        hist = np.array(track.history[-self.hist_len:], dtype=np.float64)
        # Pad with first entry if too short
        while len(hist) < self.hist_len:
            hist = np.vstack([hist[:1], hist])
        hist = hist[-self.hist_len:]

        # Compute velocities
        vel  = np.diff(hist, axis=0, prepend=hist[:1]) / (self.dt + 1e-9)
        return np.hstack([hist, vel]).astype(np.float32)  # (T, 6)


# ─────────────────────────────────────────────────────────
# Unified estimator (auto-selects best backend)
# ─────────────────────────────────────────────────────────

class TrajectoryEstimator:
    """
    High-level trajectory estimator that chooses between kinematic
    and learned prediction based on track quality.
    """

    def __init__(
        self,
        use_learned:  bool  = True,
        hist_len:     int   = 10,
        pred_steps:   int   = 10,
        dt:           float = 0.1,
        checkpoint:   Optional[str] = None,
    ):
        self._kinematic = KinematicPredictor()
        self._learned   = (
            LearnedPredictor(hist_len, pred_steps, dt, checkpoint=checkpoint)
            if use_learned else None
        )
        self.pred_steps = pred_steps
        self.dt         = dt

    def estimate(
        self,
        tracks: List[KalmanTrack],
    ) -> List[TrajectoryPrediction]:
        """
        Estimate trajectories for all active tracks.
        Uses learned model for tracks with enough history; kinematic fallback otherwise.
        """
        predictions = []
        for track in tracks:
            if self._learned and len(track.history) >= 3:
                pred = self._learned.predict(track, fallback=self._kinematic)
            else:
                pred = self._kinematic.predict(
                    track,
                    horizon_s=self.pred_steps * self.dt,
                    dt=self.dt,
                )
            predictions.append(pred)
        return predictions

    def estimate_single(self, track: KalmanTrack) -> TrajectoryPrediction:
        return self.estimate([track])[0]


if __name__ == "__main__":
    from vision.object_tracker import KalmanTrack
    from vision.object_detector import DetectedObject

    det = DetectedObject(0, "blue", "block", centre_3d=(0.1, 0.0, 0.65))
    track = KalmanTrack(det)

    # Simulate 6 frames of movement
    for i in range(6):
        d = DetectedObject(0, "blue", "block",
                           centre_3d=(0.1 + i*0.02, 0.0, 0.65))
        track.update(d)

    est  = TrajectoryEstimator(use_learned=True, pred_steps=5, dt=0.1)
    pred = est.estimate_single(track)
    print(f"Track {pred.track_id} ({pred.colour} {pred.shape})")
    print(f"  History length: {len(track.history)}")
    print(f"  Predicted {pred.num_steps} steps over {pred.horizon_s:.1f}s:")
    for i, (pos, unc) in enumerate(zip(pred.positions, pred.uncertainties)):
        print(f"    t={pred.timestamps[i]:.1f}s  pos={tuple(round(v,3) for v in pos)}  ±{unc:.3f}m")
