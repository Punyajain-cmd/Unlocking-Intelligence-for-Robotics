"""
vision/object_tracker.py
─────────────────────────
Multi-object tracker that maintains consistent identities across video frames.

Algorithm: SORT (Simple Online and Realtime Tracking) with extensions:
  • Kalman filter  per track (state: [x, y, z, vx, vy, vz, w, h])
  • Hungarian algorithm for detection → track assignment
  • Track lifecycle: tentative → confirmed → lost → deleted
  • Re-ID by colour+shape similarity (no deep features needed)
  • 3D position tracking when depth is available

Usage
─────
  tracker = ObjectTracker()
  for frame in video:
      detections = detector.detect(frame.rgb)
      tracks     = tracker.update(detections)
      for t in tracks:
          print(t.id, t.position_3d, t.velocity_3d)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
    _SCIPY = True
except ImportError:
    _SCIPY = False
    warnings.warn("scipy not installed – tracker falls back to greedy matching.")

from vision.object_detector import DetectedObject


# ─────────────────────────────────────────────────────────
# Kalman filter state: [x, y, z, vx, vy, vz, w, h]
#   x, y, z   : 3-D centroid (m or px)
#   vx,vy,vz  : velocity
#   w, h      : bounding-box width/height (pixels)
# ─────────────────────────────────────────────────────────

_STATE_DIM = 8     # state vector length
_OBS_DIM   = 6     # observation: [x, y, z, 0, 0, 0]  (no vel observed directly)


class KalmanTrack:
    """
    One tracked object — maintains a Kalman filter over its state.
    """

    _id_counter = 0

    # Process noise covariance (tunable)
    _Q = np.diag([0.01, 0.01, 0.01, 0.1, 0.1, 0.1, 0.01, 0.01]).astype(np.float64)
    # Measurement noise covariance
    _R = np.diag([0.05, 0.05, 0.05, 0.0, 0.0, 0.0]).astype(np.float64) + 1e-6 * np.eye(6)

    def __init__(self, det: DetectedObject):
        KalmanTrack._id_counter += 1
        self.id     = KalmanTrack._id_counter
        self.colour = det.colour
        self.shape  = det.shape

        # State: [x, y, z, vx, vy, vz, w, h]
        x, y, z = det.centre_3d
        w, h    = det.bbox_2d[2], det.bbox_2d[3]
        self.x  = np.array([x, y, z, 0, 0, 0, w, h], dtype=np.float64)

        # Covariance
        self.P  = np.eye(_STATE_DIM, dtype=np.float64) * 1.0

        # State transition: constant velocity model
        self.F  = np.eye(_STATE_DIM, dtype=np.float64)
        self.F[0, 3] = 1.0   # x  += vx
        self.F[1, 4] = 1.0   # y  += vy
        self.F[2, 5] = 1.0   # z  += vz

        # Observation matrix: only observe [x, y, z, 0, 0, 0]
        self.H  = np.zeros((_OBS_DIM, _STATE_DIM), dtype=np.float64)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0

        # Track lifecycle counters
        self.hits           = 1
        self.misses         = 0
        self.consecutive_hits = 1
        self.is_confirmed   = False
        self.age            = 0

        # History (positions over time for trajectory)
        self.history: List[np.ndarray] = [np.array([x, y, z])]

    # ── Kalman predict ────────────────────────────────────────

    def predict(self, dt: float = 1.0):
        """Advance one timestep (no new measurement)."""
        # Apply dt-scaled velocity
        F = self.F.copy()
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self._Q
        self.age   += 1
        self.misses += 1

    # ── Kalman update ─────────────────────────────────────────

    def update(self, det: DetectedObject):
        """Incorporate a new detection."""
        x, y, z = det.centre_3d
        z_obs   = np.array([x, y, z, 0, 0, 0], dtype=np.float64)

        S   = self.H @ self.P @ self.H.T + self._R
        K   = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z_obs - self.H @ self.x)
        self.P = (np.eye(_STATE_DIM) - K @ self.H) @ self.P

        self.hits             += 1
        self.consecutive_hits += 1
        self.misses            = 0
        self.colour = det.colour
        self.shape  = det.shape
        self.history.append(np.array([self.x[0], self.x[1], self.x[2]]))

        if self.consecutive_hits >= 2:
            self.is_confirmed = True

    # ── properties ───────────────────────────────────────────

    @property
    def position_3d(self) -> Tuple[float, float, float]:
        return (float(self.x[0]), float(self.x[1]), float(self.x[2]))

    @property
    def velocity_3d(self) -> Tuple[float, float, float]:
        return (float(self.x[3]), float(self.x[4]), float(self.x[5]))

    @property
    def is_lost(self) -> bool:
        return self.misses > 0 and not self.is_confirmed

    def to_detected_object(self) -> DetectedObject:
        return DetectedObject(
            id        = self.id,
            colour    = self.colour,
            shape     = self.shape,
            centre_3d = self.position_3d,
            confidence = min(1.0, self.hits / 5.0),
        )

    def __str__(self) -> str:
        return (f"Track#{self.id}[{self.colour} {self.shape}] "
                f"pos={self.position_3d}  vel={self.velocity_3d} "
                f"confirmed={self.is_confirmed}")


# ─────────────────────────────────────────────────────────
# Cost matrix helpers
# ─────────────────────────────────────────────────────────

def _position_cost(track: KalmanTrack, det: DetectedObject) -> float:
    tx, ty, tz = track.position_3d
    dx, dy, dz = det.centre_3d
    return float(np.sqrt((tx-dx)**2 + (ty-dy)**2 + (tz-dz)**2))


def _appearance_penalty(track: KalmanTrack, det: DetectedObject) -> float:
    """Add a large penalty if colour or shape mismatch."""
    penalty = 0.0
    if track.colour != det.colour:
        penalty += 1.0
    if track.shape != "unknown" and det.shape != "unknown":
        if track.shape != det.shape:
            penalty += 0.5
    return penalty


def _cost_matrix(
    tracks: List[KalmanTrack],
    dets:   List[DetectedObject],
    dist_thresh: float = 0.5,
) -> np.ndarray:
    C = np.zeros((len(tracks), len(dets)), dtype=np.float64)
    for i, t in enumerate(tracks):
        for j, d in enumerate(dets):
            c = _position_cost(t, d) + _appearance_penalty(t, d)
            C[i, j] = c
    return C


def _hungarian(C: np.ndarray) -> List[Tuple[int, int]]:
    """Return list of (track_idx, det_idx) matched pairs."""
    if _SCIPY:
        r, c = linear_sum_assignment(C)
        return list(zip(r.tolist(), c.tolist()))
    # Greedy fallback
    pairs = []
    used_r, used_c = set(), set()
    flat = sorted(np.ndindex(*C.shape), key=lambda ij: C[ij])
    for i, j in flat:
        if i not in used_r and j not in used_c:
            pairs.append((i, j))
            used_r.add(i)
            used_c.add(j)
    return pairs


# ─────────────────────────────────────────────────────────
# Multi-Object Tracker
# ─────────────────────────────────────────────────────────

class ObjectTracker:
    """
    SORT-based multi-object tracker with 3-D Kalman state.

    Parameters
    ──────────
    max_misses    : delete track after this many unmatched frames
    min_hits      : confirm track after this many consecutive matches
    dist_thresh   : max 3-D distance (m) for a valid match
    iou_thresh    : 2-D IoU threshold for secondary match (px-based)
    """

    def __init__(
        self,
        max_misses:  int   = 5,
        min_hits:    int   = 2,
        dist_thresh: float = 0.30,
        dt:          float = 1.0 / 10.0,
    ):
        self.max_misses  = max_misses
        self.min_hits    = min_hits
        self.dist_thresh = dist_thresh
        self.dt          = dt
        self._tracks:    List[KalmanTrack] = []
        self._frame_idx: int = 0

    # ── public API ───────────────────────────────────────────

    def update(
        self,
        detections: List[DetectedObject],
        dt: Optional[float] = None,
    ) -> List[KalmanTrack]:
        """
        Process one frame of detections.

        Parameters
        ──────────
        detections : list of DetectedObject from the current frame
        dt         : elapsed time since last frame (seconds)

        Returns
        ───────
        Confirmed tracks (id-stable DetectedObject-like objects).
        """
        _dt = dt if dt is not None else self.dt
        self._frame_idx += 1

        # 1. Predict all tracks forward
        for t in self._tracks:
            t.predict(dt=_dt)

        # 2. Match detections to predicted tracks
        matched, unmatched_tracks, unmatched_dets = self._match(detections)

        # 3. Update matched tracks
        for ti, di in matched:
            self._tracks[ti].update(detections[di])

        # 4. Create new tracks for unmatched detections
        for di in unmatched_dets:
            self._tracks.append(KalmanTrack(detections[di]))

        # 5. Increment miss counter for unmatched tracks
        for ti in unmatched_tracks:
            pass   # already incremented in predict()

        # 6. Delete dead tracks
        self._tracks = [
            t for t in self._tracks if t.misses <= self.max_misses
        ]

        return [t for t in self._tracks if t.is_confirmed]

    def reset(self):
        self._tracks = []
        self._frame_idx = 0
        KalmanTrack._id_counter = 0

    @property
    def all_tracks(self) -> List[KalmanTrack]:
        return list(self._tracks)

    @property
    def confirmed_tracks(self) -> List[KalmanTrack]:
        return [t for t in self._tracks if t.is_confirmed]

    def get_track_by_id(self, track_id: int) -> Optional[KalmanTrack]:
        for t in self._tracks:
            if t.id == track_id:
                return t
        return None

    # ── matching ─────────────────────────────────────────────

    def _match(
        self,
        dets: List[DetectedObject],
    ) -> Tuple[List[Tuple[int,int]], List[int], List[int]]:
        """
        Returns:
          matched        : [(track_idx, det_idx), …]
          unmatched_trks : [track_idx, …]
          unmatched_dets : [det_idx,  …]
        """
        if not self._tracks or not dets:
            return [], list(range(len(self._tracks))), list(range(len(dets)))

        C       = _cost_matrix(self._tracks, dets, self.dist_thresh)
        pairs   = _hungarian(C)

        matched, unmatched_t, unmatched_d = [], [], []
        matched_t, matched_d = set(), set()

        for ti, di in pairs:
            if C[ti, di] < self.dist_thresh:
                matched.append((ti, di))
                matched_t.add(ti)
                matched_d.add(di)

        unmatched_t = [i for i in range(len(self._tracks)) if i not in matched_t]
        unmatched_d = [i for i in range(len(dets))         if i not in matched_d]
        return matched, unmatched_t, unmatched_d


# ─────────────────────────────────────────────────────────
# Track state aggregator
# ─────────────────────────────────────────────────────────

@dataclass
class TrackState:
    """Snapshot of all active tracks at one timestep."""
    frame_idx:  int
    timestamp_s: float
    tracks:     List[KalmanTrack] = field(default_factory=list)

    def get_object(self, colour: str, shape: Optional[str] = None) -> Optional[KalmanTrack]:
        candidates = [t for t in self.tracks if t.colour == colour]
        if shape:
            candidates = [t for t in candidates if t.shape == shape] or candidates
        if not candidates:
            return None
        return max(candidates, key=lambda t: t.hits)

    def to_detected_objects(self) -> List[DetectedObject]:
        return [t.to_detected_object() for t in self.tracks]


if __name__ == "__main__":
    from vision.object_detector import DetectedObject

    tracker = ObjectTracker()

    # Simulate 5 frames with two moving objects
    objs_t0 = [
        DetectedObject(0, "blue",  "block",  centre_3d=(0.0, 0.0, 0.65)),
        DetectedObject(1, "green", "sphere", centre_3d=(0.1, 0.0, 0.65)),
    ]
    for frame_i in range(5):
        dets = [
            DetectedObject(0, "blue",  "block",  centre_3d=(0.0 + frame_i*0.01, 0.0, 0.65)),
            DetectedObject(1, "green", "sphere", centre_3d=(0.1 - frame_i*0.01, 0.0, 0.65)),
        ]
        confirmed = tracker.update(dets)
        print(f"Frame {frame_i}: {len(confirmed)} confirmed tracks")
        for t in confirmed:
            print(f"  {t}")
