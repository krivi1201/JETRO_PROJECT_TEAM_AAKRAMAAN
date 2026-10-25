"""Vectorised quaternion helpers. Arrays are (..., 4) in (w, x, y, z) order."""

import numpy as np


def normalize(q):
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def conj(q):
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def mul(a, b):
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=-1)


def make_continuous(q):
    """Flip signs so consecutive samples stay on the same hemisphere (q and -q are equal)."""
    q = q.copy()
    flips = np.sum(q[1:] * q[:-1], axis=-1) < 0
    sign = np.r_[1.0, np.where(np.cumsum(flips) % 2 == 1, -1.0, 1.0)]
    return q * sign[:, None]


def mean(q):
    """Average of nearby orientations (sign-aligned to the first, then normalised)."""
    q = q * np.where(np.sum(q * q[:1], axis=-1) < 0, -1.0, 1.0)[:, None]
    return normalize(q.mean(axis=0))


def angle_between(a, b):
    """Rotation angle (degrees) taking orientation a to b.

    Invariant to how the sensor is mounted on the segment, which is unknown here.
    """
    d = np.abs(np.sum(a * b, axis=-1))
    return np.degrees(2 * np.arccos(np.clip(d, 0.0, 1.0)))


def angular_velocity(q, fs):
    """Body-frame angular velocity vectors (rad/s), same length as q."""
    dq = mul(conj(q[:-1]), q[1:])
    dq *= np.where(dq[:, :1] < 0, -1.0, 1.0)
    vec = dq[:, 1:]
    s = np.linalg.norm(vec, axis=-1, keepdims=True)
    ang = 2 * np.arctan2(s, dq[:, :1])
    omega = np.where(s > 1e-12, vec / np.maximum(s, 1e-12) * ang, 0.0) * fs
    return np.vstack([omega, omega[-1:]])
