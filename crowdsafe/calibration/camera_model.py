"""Full camera model (focal length, lens distortion, pose) fitted from desk tops (T2.3).

Why not a plain homography: the classroom CCTV lenses are wide-angle with strong barrel
distortion, and people's heads are ~1 m above the floor. A pinhole camera with radial distortion
maps pixels to *any* horizontal plane (floor, desk tops, head height), so heads can be placed on
the floor plan correctly.

Fitting cue: every desk top is a rectangle of known length (measured by the user) lying on one
horizontal plane, and all desks are parallel. Each desk gets its own unknown floor position, so the
fit does not assume the row pitch or the column gaps — those measurements are kept back as an
independent check of the calibration.

World frame: X along the desk long side, Y towards the back of the room, Z up; Z = 0 is the
desk-top plane during fitting, then shifted so that Z = 0 is the floor (`with_floor_at`).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import least_squares
from shapely.geometry import Polygon


@dataclass
class CameraModel:
    """Pinhole camera with radial distortion (k1, k2), world frame in metres, Z up."""

    K: np.ndarray                 # 3×3 intrinsics
    dist: np.ndarray              # (5,) OpenCV distortion: k1, k2, p1, p2, k3
    rvec: np.ndarray              # (3,) world -> camera rotation (Rodrigues)
    tvec: np.ndarray              # (3,) world -> camera translation
    image_size: tuple[int, int]   # (width, height)
    note: str = ""

    @property
    def R(self) -> np.ndarray:
        return cv2.Rodrigues(np.asarray(self.rvec, float))[0]

    @property
    def center(self) -> np.ndarray:
        """Camera centre in world coordinates (m); center[2] is the height above Z = 0."""
        return -self.R.T @ np.asarray(self.tvec, float)

    def project(self, xyz: np.ndarray) -> np.ndarray:
        """(N, 3) world metres -> (N, 2) image pixels (distorted, as in the video)."""
        pts = np.asarray(xyz, np.float64).reshape(-1, 1, 3)
        if len(pts) == 0:
            return np.zeros((0, 2))
        uv, _ = cv2.projectPoints(pts, np.asarray(self.rvec, float), np.asarray(self.tvec, float), self.K, self.dist)
        return uv.reshape(-1, 2)

    def rays(self, px: np.ndarray) -> np.ndarray:
        """(N, 2) image pixels -> (N, 3) unit ray directions in world coordinates.

        Pixels beyond the range where the lens model is invertible (extreme image corners of a strongly
        distorted lens) give NaN.
        """
        p = np.asarray(px, np.float64).reshape(-1, 1, 2)
        n = cv2.undistortPoints(p, self.K, self.dist).reshape(-1, 2)
        d = (self.R.T @ np.c_[n, np.ones(len(n))].T).T
        d = d / np.linalg.norm(d, axis=1, keepdims=True)
        rv = self.max_valid_radius()
        if np.isfinite(rv):
            rd_max = rv * (1 + self.dist[0] * rv ** 2 + self.dist[1] * rv ** 4)
            q = (p.reshape(-1, 2) - self.K[:2, 2]) / self.K[0, 0]
            d[np.hypot(q[:, 0], q[:, 1]) > 0.98 * rd_max] = np.nan
        return d

    def to_plane(self, px: np.ndarray, z: float = 0.0) -> np.ndarray:
        """(N, 2) image pixels -> (N, 2) world XY where the viewing ray meets the plane Z = z.

        Rays that never reach the plane (pointing above the horizon of that plane) give NaN.
        """
        d = self.rays(px)
        c = self.center
        with np.errstate(divide="ignore", invalid="ignore"):
            s = (z - c[2]) / d[:, 2]
        xy = c[:2] + s[:, None] * d[:, :2]
        xy[~(s > 0)] = np.nan
        return xy

    def area_m2(self, polygon_xy: list[tuple[float, float]]) -> float:
        """Area of a polygon given in world XY metres (zones are defined on the floor plan)."""
        return float(Polygon(polygon_xy).area)

    def zone_mask(self, polygon_xy: list[tuple[float, float]], z: float, shape_hw: tuple[int, int] | None = None,
                  step_m: float = 0.05) -> np.ndarray:
        """Image mask (uint8 0/1) of the pixels whose rays meet plane Z = z inside the floor polygon.

        Built by sampling the polygon edges densely and projecting them, so distortion bends the
        edges correctly. `shape_hw` defaults to the full image.
        """
        h, w = shape_hw or (self.image_size[1], self.image_size[0])
        poly = np.asarray(polygon_xy, float)
        pts = []
        for a, b in zip(poly, np.roll(poly, -1, axis=0)):
            n = max(2, int(np.ceil(np.linalg.norm(b - a) / step_m)))
            pts.append(a + np.linspace(0, 1, n, endpoint=False)[:, None] * (b - a))
        edge = np.vstack(pts)
        uv = self.project(np.c_[edge, np.full(len(edge), z)])
        sx, sy = w / self.image_size[0], h / self.image_size[1]
        m = np.zeros((h, w), np.uint8)
        cv2.fillPoly(m, [np.round(uv * [sx, sy]).astype(np.int32)], 1)
        return m

    def max_valid_radius(self) -> float:
        """Largest undistorted normalised radius where the radial model is still monotonic (inf if always)."""
        k1, k2 = float(self.dist[0]), float(self.dist[1])
        # d/dr [r (1 + k1 r^2 + k2 r^4)] = 1 + 3 k1 x + 5 k2 x^2, x = r^2
        roots = np.roots([5 * k2, 3 * k1, 1.0]) if abs(k2) > 1e-12 else (np.array([-1 / (3 * k1)]) if k1 < 0 else np.array([]))
        pos = [r.real for r in np.atleast_1d(roots) if abs(r.imag) < 1e-12 and r.real > 0]
        return float(np.sqrt(min(pos))) if pos else float("inf")

    def in_view(self, xyz: np.ndarray, margin_px: float = 0.0) -> np.ndarray:
        """Boolean mask: point is in front of the camera, inside the valid lens range and inside the image."""
        P = (self.R @ np.asarray(xyz, float).reshape(-1, 3).T).T + np.ravel(self.tvec)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.hypot(P[:, 0], P[:, 1]) / P[:, 2]
        ok = (P[:, 2] > 0.05) & (r < 0.98 * self.max_valid_radius())
        uv = np.full((len(P), 2), np.nan)
        if ok.any():
            uv[ok] = self.project(np.asarray(xyz, float).reshape(-1, 3)[ok])
        W, H = self.image_size
        return ok & (uv[:, 0] >= -margin_px) & (uv[:, 0] < W + margin_px) & (uv[:, 1] >= -margin_px) & (uv[:, 1] < H + margin_px)

    def with_floor_at(self, z_floor: float) -> "CameraModel":
        """Same camera, world shifted so that the plane Z = z_floor becomes Z = 0."""
        t = np.asarray(self.tvec, float) + self.R @ np.array([0.0, 0.0, z_floor])
        return CameraModel(self.K.copy(), self.dist.copy(), np.asarray(self.rvec, float).copy(), t,
                           self.image_size, self.note)

    def to_dict(self) -> dict:
        return {"K": np.round(self.K, 4).tolist(), "dist": np.round(self.dist, 6).tolist(),
                "rvec": np.round(np.ravel(self.rvec), 6).tolist(), "tvec": np.round(np.ravel(self.tvec), 6).tolist(),
                "image_size": list(self.image_size), "note": self.note}

    @classmethod
    def from_dict(cls, d: dict) -> "CameraModel":
        return cls(np.asarray(d["K"], float), np.asarray(d["dist"], float), np.asarray(d["rvec"], float),
                   np.asarray(d["tvec"], float), tuple(d["image_size"]), d.get("note", ""))


@dataclass
class RectFit:
    """Result of `fit_from_rectangles`."""

    camera: CameraModel                     # world Z = 0 is the rectangle (desk-top) plane
    depth_m: float                          # fitted rectangle depth (shared by all rectangles)
    positions: dict[str, np.ndarray]        # name -> (x, y) of the rectangle's first corner
    residuals_px: dict[str, np.ndarray]     # name -> (4,) reprojection error per corner (NaN = not observed)
    rms_px: float
    params: dict = field(default_factory=dict)


def rect_corners(x: float, y: float, length: float, depth: float) -> np.ndarray:
    """Corners in the order front-left, front-right, back-right, back-left (Z = 0)."""
    return np.array([[x, y, 0], [x + length, y, 0], [x + length, y + depth, 0], [x, y + depth, 0]], float)


def _init_pose(q: np.ndarray, length: float, depth: float, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pose from one rectangle's homography (planar pose decomposition)."""
    Hm, _ = cv2.findHomography(rect_corners(0, 0, length, depth)[:, :2], q)
    B = np.linalg.inv(K) @ Hm
    lam = 1 / np.linalg.norm(B[:, 0])
    if B[2, 2] * lam < 0:
        lam = -lam
    r1, r2, t = B[:, 0] * lam, B[:, 1] * lam, B[:, 2] * lam
    U, _, Vt = np.linalg.svd(np.column_stack([r1, r2, np.cross(r1, r2)]))
    return cv2.Rodrigues(U @ Vt)[0].ravel(), t


def fit_from_rectangles(obs: dict[str, np.ndarray], length_m: float, image_size: tuple[int, int],
                        ref: str | None = None, free_k2: bool = True, free_pp: bool = False,
                        f_init: float | None = None, depth_init: float = 0.4,
                        depth_bounds: tuple[float, float] = (0.15, 1.0),
                        lattice: dict[str, tuple[int, int]] | None = None,
                        row_pitch_m: float | None = None, pitch_weight: float = 100.0,
                        lines: list[np.ndarray] | None = None, line_weight: float = 1.0) -> RectFit:
    """Fit focal length, distortion, pose and the floor position of every rectangle.

    obs: name -> (4, 2) image corners [near-left, near-right, far-right, far-left] (near = closer to
    the camera); NaN rows are missing corners (occluded). All rectangles are parallel, on one plane,
    with the known length `length_m` along X and an unknown shared depth along Y. The first corner
    of `ref` is the origin.

    Free mode (default): every rectangle has its own unknown (x, y).
    Pitch prior (`lattice` = name -> (column, row) and `row_pitch_m`): positions stay free (desks are
    movable), but for every pair of rectangles in one column the y difference is pulled softly
    towards (row difference) × `row_pitch_m`, weight `pitch_weight` px per metre. This pins the depth
    scale, which a single view determines only to a few per cent (focal length and camera tilt trade
    off). Column gaps are not used, so they remain an independent check.

    Plumb lines (`lines`): image samples along edges that are straight in the room (wall corners,
    door frames, ceiling ducts). After undistortion each must be a straight line; their residuals
    (px) pin the lens distortion, which desks alone may leave loose near the image border.
    """
    names = list(obs)
    ref = ref or names[0]
    names = [ref] + [n for n in names if n != ref]
    if lattice is not None and row_pitch_m is None:
        raise ValueError("the pitch prior needs row_pitch_m")
    pairs = [] if lattice is None else [
        (i, j, (lattice[b][1] - lattice[a][1]) * row_pitch_m)
        for i, a in enumerate(names) for j, b in enumerate(names)
        if j > i and lattice[a][0] == lattice[b][0] and lattice[a][1] != lattice[b][1]]
    W, H = image_size
    Y = np.vstack([np.asarray(obs[n], float).reshape(4, 2) for n in names])
    seen = ~np.isnan(Y).any(1)
    f0 = f_init or 0.7 * W
    K0 = np.array([[f0, 0, W / 2], [0, f0, H / 2], [0, 0, 1]])
    if np.isnan(Y[:4]).any():
        raise ValueError(f"reference rectangle {ref!r} needs all four corners")
    rv0, t0 = _init_pose(Y[:4], length_m, depth_init, K0)
    Hinv = np.linalg.inv(cv2.findHomography(rect_corners(0, 0, length_m, depth_init)[:, :2], Y[:4])[0])
    pos0 = []
    for i in range(1, len(names)):
        q, ok = Y[4 * i:4 * i + 4], seen[4 * i:4 * i + 4]
        w = cv2.perspectiveTransform(q[ok].reshape(-1, 1, 2), Hinv).reshape(-1, 2)
        pos0.append((w - rect_corners(0, 0, length_m, depth_init)[ok, :2]).mean(0))
    # parameters: f, k1, k2, cx, cy, rvec(3), tvec(3), depth, positions(2 per non-ref rectangle)
    p0 = np.r_[f0, 0.0, 0.0, W / 2, H / 2, rv0, t0, depth_init, np.ravel(pos0)]
    lo = np.full(p0.shape, -np.inf)
    hi = np.full(p0.shape, np.inf)
    lo[0], hi[0] = 0.2 * W, 3.0 * W
    lo[1], hi[1] = -1.0, 1.0
    lo[2], hi[2] = (0.0, 1.0) if free_k2 else (-1e-12, 1e-12)        # k2 >= 0 keeps barrel distortion monotonic
    dpp = 0.08 * W if free_pp else 1e-9
    lo[3], hi[3], lo[4], hi[4] = W / 2 - dpp, W / 2 + dpp, H / 2 - dpp, H / 2 + dpp
    lo[11], hi[11] = depth_bounds

    def unpack(p):
        K = np.array([[p[0], 0, p[3]], [0, p[0], p[4]], [0, 0, 1]])
        pos = np.vstack([[0.0, 0.0], p[12:].reshape(-1, 2)])
        return K, np.array([p[1], p[2], 0, 0, 0]), p[5:8], p[8:11], p[11], pos

    def project(p):
        K, dist, rv, tv, d, pos = unpack(p)
        X = np.vstack([rect_corners(x, y, length_m, d) for x, y in pos])
        return cv2.projectPoints(X, rv, tv, K, dist)[0].reshape(-1, 2)

    lines = [np.asarray(ln, float).reshape(-1, 2) for ln in (lines or [])]

    def straightness(p):
        K, dist = unpack(p)[:2]
        out = []
        for ln in lines:
            n = cv2.undistortPoints(ln.reshape(-1, 1, 2), K, dist).reshape(-1, 2)
            m = n.mean(0)
            normal = np.linalg.svd(n - m)[2][1]          # total least squares line normal
            out.append(K[0, 0] * (n - m) @ normal)
        return np.concatenate(out) if out else np.zeros(0)

    def resid(p, use_pairs):
        r = np.r_[(project(p) - Y)[seen].ravel(), line_weight * straightness(p)]
        if not use_pairs:
            return r
        pos = unpack(p)[5]
        return np.r_[r, [pitch_weight * (pos[j, 1] - pos[i, 1] - dy) for i, j, dy in pairs]]

    sol = least_squares(resid, p0, bounds=(lo, hi), loss="soft_l1", f_scale=3.0, x_scale="jac", args=(False,))
    free_pos = unpack(sol.x)[5]
    if pairs:     # second stage: start from the free solution, add the pitch prior
        sol = least_squares(resid, sol.x, bounds=(lo, hi), loss="soft_l1", f_scale=3.0, x_scale="jac", args=(True,))
    K, dist, rv, tv, d, pos = unpack(sol.x)
    return _result(sol.x, names, project(sol.x), Y, seen, K, dist, rv, tv, d, pos, (W, H),
                   {"free_k2": free_k2, "free_pp": free_pp, "pitch_prior": bool(pairs), "row_pitch_m": row_pitch_m,
                    "n_pitch_pairs": len(pairs), "n_lines": len(lines),
                    "line_rms_px": float(np.sqrt(np.mean(straightness(sol.x) ** 2))) if lines else None,
                    "free_fit_positions": {n: np.round(free_pos[i], 3).tolist() for i, n in enumerate(names)}})


def _result(x, names, proj, Y, seen, K, dist, rv, tv, d, pos, size, extra) -> RectFit:
    err = np.linalg.norm(proj - Y, axis=1)
    err[~seen] = np.nan
    cam = CameraModel(K, dist, np.asarray(rv, float), np.asarray(tv, float), size,
                      note="fitted from desk-top rectangles")
    return RectFit(cam, float(d), {n: np.asarray(pos[i]) for i, n in enumerate(names)},
                   {n: err[4 * i:4 * i + 4] for i, n in enumerate(names)},
                   float(np.sqrt(np.nanmean(err ** 2))),
                   {"f_px": float(x[0]), "k1": float(x[1]), "k2": float(x[2]), "cx": float(x[3]),
                    "cy": float(x[4]), **extra})


@dataclass
class EdgeFit:
    """Result of `fit_from_edges`."""

    camera: CameraModel
    rms_px: dict[str, float]               # per constraint group
    params: dict = field(default_factory=dict)


def _look_rotation(yaw: float, pitch: float, roll: float = 0.0) -> np.ndarray:
    """World->camera rotation for a camera looking along yaw (from +Y towards +X) and tilted down by pitch (rad)."""
    F = np.array([np.cos(pitch) * np.sin(yaw), np.cos(pitch) * np.cos(yaw), -np.sin(pitch)])
    right = np.cross(F, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(F, right)
    R = np.vstack([right, down, F])
    c, s = np.cos(roll), np.sin(roll)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ R


def fit_from_edges(edges: list[dict], image_size: tuple[int, int], verticals: list[np.ndarray] | None = None,
                   lines: list[np.ndarray] | None = None, cam_xy_init: tuple[float, float] = (0.0, 0.0),
                   cam_z_init: float = 2.0, cam_z_prior: float | None = None, prior_weight: float = 200.0,
                   fixed: tuple[str, ...] = ("cy",), free_k2: bool = True) -> EdgeFit:
    """Fit a camera from image edges that lie on known lines of a horizontal plane.

    For scenes without repeated rectangles (e.g. long lab tables): each entry of `edges` is
    {"pts": (N, 2) image samples, "axis": "X" or "Y", "value": known coordinate (m), "z": plane height}
    — e.g. a table's long edge lies on X = 2.74 m at table height. `verticals`: samples along edges that
    are vertical in the room (window frames, panel edges). `lines`: other straight edges (straightness
    only). `fixed` lists camera-centre coordinates that are not observable ("cx", "cy", "cz") and are
    held at their initial value. `cam_z_prior` softly ties the camera height (m above Z = 0).
    Residuals are in pixels; the camera centre is solved directly (tvec = -R·C).
    """
    W, H = image_size
    verticals = [np.asarray(v, float).reshape(-1, 2) for v in (verticals or [])]
    lines = [np.asarray(v, float).reshape(-1, 2) for v in (lines or [])]
    edges = [dict(e, pts=np.asarray(e["pts"], float).reshape(-1, 2)) for e in edges]
    free_c = [k not in fixed for k in ("cx", "cy", "cz")]
    c_init = np.array([cam_xy_init[0], cam_xy_init[1], cam_z_init], float)

    def unpack(p):
        f, k1, k2 = p[0], p[1], p[2]
        K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])
        R = cv2.Rodrigues(p[3:6])[0]
        C = c_init.copy()
        C[free_c] = p[6:6 + sum(free_c)]
        return K, np.array([k1, k2, 0, 0, 0]), R, C

    def world_rays(K, dist, R, pts):
        n = cv2.undistortPoints(pts.reshape(-1, 1, 2), K, dist).reshape(-1, 2)
        d = (R.T @ np.c_[n, np.ones(len(n))].T).T
        return d / np.linalg.norm(d, axis=1, keepdims=True)

    def groups(p):
        K, dist, R, C = unpack(p)
        f = K[0, 0]
        out = {"edges": [], "verticals": [], "lines": []}
        for e in edges:
            d = world_rays(K, dist, R, e["pts"])
            s = (e["z"] - C[2]) / np.where(np.abs(d[:, 2]) < 1e-6, -1e-6, d[:, 2])
            s = np.where(s > 0, s, 1e3)                          # behind the camera: large penalty
            q = C + s[:, None] * d
            k = 0 if e["axis"] == "X" else 1
            out["edges"].append(f * (q[:, k] - e["value"]) / s)
        for v in verticals:
            d = world_rays(K, dist, R, v)
            M = d[:, :2].T @ d[:, :2]
            nrm = np.linalg.eigh(M)[1][:, 0]                    # horizontal normal of the best vertical plane
            out["verticals"].append(f * d[:, :2] @ nrm)
        for ln in lines:
            n = cv2.undistortPoints(ln.reshape(-1, 1, 2), K, dist).reshape(-1, 2)
            m = n.mean(0)
            out["lines"].append(f * (n - m) @ np.linalg.svd(n - m)[2][1])
        return {k: np.concatenate(v) if v else np.zeros(0) for k, v in out.items()}, C

    def resid(p):
        g, C = groups(p)
        r = np.r_[g["edges"], g["verticals"], g["lines"]]
        if cam_z_prior is not None:
            r = np.r_[r, prior_weight * (C[2] - cam_z_prior)]
        return r

    lo_f, hi_f = 0.2 * W, 3.0 * W
    best = None
    for yaw in np.radians(np.arange(-180, 180, 15)):          # coarse search for the viewing direction
        for pitch in np.radians([10, 20, 30, 40, 50]):
            rv = cv2.Rodrigues(_look_rotation(yaw, pitch))[0].ravel()
            p = np.r_[0.6 * W, 0.0, 0.0, rv, c_init[free_c]]
            cost = np.sum(np.minimum(resid(p) ** 2, 1e4))
            if best is None or cost < best[0]:
                best = (cost, p)
    lo = np.r_[lo_f, -1.0, 0.0 if free_k2 else -1e-12, [-np.inf] * 3, [-np.inf] * sum(free_c)]
    hi = np.r_[hi_f, 1.0, 1.0 if free_k2 else 1e-12, [np.inf] * 3, [np.inf] * sum(free_c)]
    p0 = np.clip(best[1], lo + 1e-12, hi - 1e-12)
    sol = least_squares(resid, p0, bounds=(lo, hi), loss="soft_l1", f_scale=3.0, x_scale="jac")
    K, dist, R, C = unpack(sol.x)
    rv = cv2.Rodrigues(R)[0].ravel()
    cam = CameraModel(K, dist, rv, -R @ C, (W, H), note="fitted from known table edges and vertical lines")
    g, _ = groups(sol.x)
    return EdgeFit(cam, {k: float(np.sqrt(np.mean(v ** 2))) if len(v) else float("nan") for k, v in g.items()},
                   {"f_px": float(K[0, 0]), "k1": float(dist[0]), "k2": float(dist[1]), "camera_center": C.round(3).tolist()})
