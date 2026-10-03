"""Column-vector transforms; camera axes x right, y down, z forward.

Pixel centers are (column + .5, row + .5), as in COLMAP.
"""
import numpy as np
import cv2


def homogeneous(rotation, translation=None):
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    if translation is not None:
        result[:3, 3] = translation
    return result


def rays(size):
    y, x = np.mgrid[:size, :size]
    out = np.stack(((x + .5 - size / 2) / (size / 2),
                    (y + .5 - size / 2) / (size / 2), np.ones_like(x)), -1)
    return out / np.linalg.norm(out, axis=-1, keepdims=True)


def erp_coordinates(directions, width, height):
    lon = np.arctan2(directions[..., 0], directions[..., 2])
    lat_down = np.arctan2(directions[..., 1],
                        np.linalg.norm(directions[..., [0, 2]], axis=-1))
    u = ((lon / (2 * np.pi) + .5) * width - .5) % width
    v = np.clip((lat_down / np.pi + .5) * height - .5, 0, height - 1)
    return u.astype(np.float32), v.astype(np.float32)


def project(image, R_face_from_pano, size, mask=False):
    source = np.asarray(image)
    directions = np.einsum('...i,ij->...j', rays(size), R_face_from_pano, optimize=False)
    u, v = erp_coordinates(directions, source.shape[1], source.shape[0])
    # Horizontal wrapping is necessary at the ERP seam; vertical coords are clamped.
    if mask:
        # Every source sample contributing to RGB must be valid. Nearest-neighbour
        # masks would leak excluded source pixels through bilinear interpolation.
        x0, y0 = np.floor(u).astype(int), np.floor(v).astype(int)
        x1, y1 = (x0 + 1) % source.shape[1], np.minimum(y0 + 1, source.shape[0] - 1)
        valid = ((source[y0, x0] > 0) & (source[y0, x1] > 0) &
                 (source[y1, x0] > 0) & (source[y1, x1] > 0))
        return valid.astype(np.uint8) * 255
    return cv2.remap(source, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def ownership(R_face_from_pano, all_rotations, size, index):
    directions = np.einsum('...i,ij->...j', rays(size), R_face_from_pano, optimize=False)
    axes = np.asarray(all_rotations)[:, 2, :]
    return (np.argmax(np.einsum('...i,ji->...j', directions, axes, optimize=False), axis=-1) == index).astype(np.uint8) * 255


def cube_rotations():
    """Six proper rotations including zenith and nadir, with zero baseline."""
    forward = {'front': (0, 0, 1), 'right': (1, 0, 0), 'back': (0, 0, -1),
               'left': (-1, 0, 0), 'up': (0, -1, 0), 'down': (0, 1, 0)}
    result = {}
    for name, axis in forward.items():
        z = np.array(axis, dtype=float)
        down = np.array((0, 0, 1) if name == 'up' else
                        (0, 0, -1) if name == 'down' else (0, 1, 0))
        x = np.cross(down, z)
        result[name] = np.stack((x, np.cross(z, x), z))
    return result
