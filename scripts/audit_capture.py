#!/usr/bin/env python3
"""Read-only capture audit; emits metadata and local inspection contact sheets.

Requires Pillow and NumPy. No images are uploaded and no 3D is inferred.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def perspective(im, yaw_deg, pitch_deg=0, size=480, fov_deg=90):
    """ERP -> pinhole preview, bilinear sampling with longitude wrapping.

    Directions use x right, y up, z forward. This is for visual audit only;
    these previews do not define a calibrated COLMAP reconstruction.
    """
    a = (2 * (np.arange(size) + 0.5) / size - 1) * math.tan(math.radians(fov_deg / 2))
    x, ny = np.meshgrid(a, a)
    y, z = -ny, np.ones_like(x)
    pitch = math.radians(pitch_deg)
    y, z = y * math.cos(pitch) + z * math.sin(pitch), -y * math.sin(pitch) + z * math.cos(pitch)
    yaw = math.radians(yaw_deg)
    x, z = x * math.cos(yaw) + z * math.sin(yaw), -x * math.sin(yaw) + z * math.cos(yaw)
    lon = np.arctan2(x, z)
    lat = np.arctan2(y, np.sqrt(x * x + z * z))
    src = np.asarray(im.convert('RGB'))
    h, w = src.shape[:2]
    u = (lon / (2 * math.pi) + 0.5) * w - 0.5
    v = np.clip((0.5 - lat / math.pi) * h - 0.5, 0, h - 1)
    iu, iv = np.floor(u).astype(int), np.floor(v).astype(int)
    fu, fv = (u - iu)[..., None], (v - iv)[..., None]
    p00, p10 = src[iv, iu % w], src[iv, (iu + 1) % w]
    p01, p11 = src[np.minimum(iv + 1, h - 1), iu % w], src[np.minimum(iv + 1, h - 1), (iu + 1) % w]
    out = (p00 * (1 - fu) + p10 * fu) * (1 - fv) + (p01 * (1 - fu) + p11 * fu) * fv
    return Image.fromarray(np.clip(out, 0, 255).astype('uint8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('Input'))
    parser.add_argument('--output', type=Path, default=Path('Output/RD/audit'))
    args = parser.parse_args()
    paths = sorted(p for p in args.input.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.tif', '.tiff'})
    if not paths:
        raise SystemExit('No supported images found')
    if args.output.resolve() == args.input.resolve() or args.input.resolve() in args.output.resolve().parents:
        raise SystemExit('Output must be outside the source image directory')
    args.output.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default(size=18)
    rows = []
    thumb_w, thumb_h, label_h, columns = 672, 336, 42, 3
    sheet = Image.new('RGB', (columns * thumb_w, math.ceil(len(paths) / columns) * (thumb_h + label_h)), '#111b29')
    draw = ImageDraw.Draw(sheet)
    selected = sorted(set([0, len(paths) // 3, 2 * len(paths) // 3, len(paths) - 1]))
    previews = Image.new('RGB', (4 * 480, len(selected) * 514), '#111b29')
    preview_draw = ImageDraw.Draw(previews)
    preview_row = 0
    for index, path in enumerate(paths):
        with Image.open(path) as raw:
            raw.load()
            exif = raw.getexif()
            details = exif.get_ifd(34665)
            im = raw.convert('RGB')
            row = {
                'file': path.name,
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'bytes': path.stat().st_size,
                'width': im.width,
                'height': im.height,
                'aspect_ratio': im.width / im.height,
                'make': exif.get(271),
                'model': exif.get(272),
                'software': exif.get(305),
                'orientation': exif.get(274),
                'date_time_original': details.get(36867),
                'utc_offset': details.get(36881),
                'exposure_s': float(details[33434]) if 33434 in details else None,
                'f_number': float(details[33437]) if 33437 in details else None,
                'iso': details.get(34855),
                'white_balance': details.get(41987),
                'exposure_bias_ev': float(details[37380]) if 37380 in details else None,
                'gps_ifd_present': bool(exif.get_ifd(34853)),
                'erp_candidate': im.width == 2 * im.height,
            }
            small = np.asarray(im.resize((1024, 512)), dtype=np.float32)
            weights = np.cos((0.5 - (np.arange(512) + 0.5) / 512) * math.pi)[:, None]
            near_white = np.all(small >= 250, axis=-1)
            near_black = np.all(small <= 5, axis=-1)
            denominator = weights.sum() * 1024
            row['near_white_spherical_fraction'] = float((near_white * weights).sum() / denominator)
            row['near_black_spherical_fraction'] = float((near_black * weights).sum() / denominator)
            rows.append(row)
            x, y = (index % columns) * thumb_w, (index // columns) * (thumb_h + label_h)
            sheet.paste(im.resize((thumb_w, thumb_h)), (x, y))
            draw.text((x + 12, y + thumb_h + 9), f'{path.name} | ISO {row["iso"]} | {row["exposure_s"]} s', font=font, fill='white')
            if index in selected and row['erp_candidate']:
                for j, yaw in enumerate([0, 90, 180, 270]):
                    previews.paste(perspective(im, yaw), (j * 480, preview_row * 514))
                    preview_draw.text((j * 480 + 8, preview_row * 514 + 486), f'{path.stem} / yaw {yaw}', font=font, fill='white')
                preview_row += 1
    hashes = [r['sha256'] for r in rows]
    payload = {
        'input_directory': str(args.input),
        'count': len(rows),
        'total_bytes': sum(r['bytes'] for r in rows),
        'unique_file_hashes': len(set(hashes)),
        'measurement_notes': [
            'A 2:1 ratio is an ERP candidate; visual inspection is required.',
            'Near-white/black rates use 1024x512 RGB previews, cos(latitude) weighting; they do not prove exposure clipping.',
            'No camera positions, baseline distances, metric scale or reconstruction quality are measured here.',
            'EXIF dates are camera metadata, not independently verified timestamps.',
        ],
        'images': rows,
    }
    (args.output / 'capture_manifest.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    sheet.save(args.output / 'panoramas_contact.jpg', quality=90)
    previews.save(args.output / 'perspective_contact.jpg', quality=90)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
