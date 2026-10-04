"""One comparative sheet of the same validation faces across trainings/checkpoints (CPU).

    python -m theta_pipeline.gsplat_sheet --prep Output/runs/salon-gsplat-007 --name ctrl-duration-pruning \
        --columns ctrl-3k-s0:step_003000.pt ctrl-3k-s1:step_003000.pt \
                  ctrl-10k-s0:step_003000.pt ctrl-10k-s0:step_010000.pt \
                  ctrl-10k-s1:step_003000.pt ctrl-10k-s1:step_010000.pt

Assembles existing inspection v2 exports (validation/full); renders nothing and reads no
test data. Per face: renders, error maps and depth (floaters) in rows, one column per
training/checkpoint, plus a table of face and region PSNR (glass, contours, furniture,
mirror) and the low-alpha fraction. Refuses columns that did not inspect the same faces.
"""
import argparse
import sys
from pathlib import Path
from PIL import Image, ImageDraw
from .storage import now, read, write

ROWS = ('render', 'error', 'depth')
REGIONS = ('glass', 'contours', 'furniture', 'mirror')


def load_columns(prep, columns):
    loaded = []
    for column in columns:
        name, checkpoint = column.split(':')
        folder = Path(prep) / 'inspection' / f'{name}-{Path(checkpoint).stem}-v2'
        summary = read(folder / 'inspection.json')
        if summary.get('test_loaded'):
            raise RuntimeError(f'{folder.name} loaded the test set')
        faces = read(folder / 'validation-full' / 'inspection.json')['faces']
        loaded.append({'label': f'{name} @ {Path(checkpoint).stem.removeprefix("step_").lstrip("0")}',
                       'folder': folder / 'validation-full', 'faces': {f['camera']: f for f in faces}})
    names = [sorted(c['faces']) for c in loaded]
    if any(n != names[0] for n in names):
        raise RuntimeError('columns did not inspect the same validation faces')
    return loaded, names[0]


def thumbnail(path, size):
    with Image.open(path) as image:
        return image.convert('RGB').resize((size, size))


def face_block(face, columns, size):
    """Reference + one column per run; rows render / error / depth."""
    width = (len(columns) + 1) * size
    block = Image.new('RGB', (width, len(ROWS) * size + 22), '#1d252c')
    draw = ImageDraw.Draw(block)
    draw.text((4, 4), face, fill='#f0c040')
    first = columns[0]['faces'][face]['files']
    block.paste(thumbnail(columns[0]['folder'] / first['reference'], size), (0, 22))
    for k, column in enumerate(columns, start=1):
        files = column['faces'][face]['files']
        for r, row in enumerate(ROWS):
            block.paste(thumbnail(column['folder'] / files[row], size), (k * size, 22 + r * size))
    return block


def sheet(columns, faces, size=256):
    header = Image.new('RGB', ((len(columns) + 1) * size, 40), '#10161b')
    draw = ImageDraw.Draw(header)
    draw.text((4, 4), 'reference', fill='white')
    for k, column in enumerate(columns, start=1):
        draw.text((k * size + 4, 4), column['label'], fill='white')
        draw.text((k * size + 4, 22), 'rendu / erreur / profondeur', fill='#9aa4ad')
    blocks = [face_block(face, columns, size) for face in faces]
    out = Image.new('RGB', (header.width, header.height + sum(b.height for b in blocks)), '#1d252c')
    out.paste(header, (0, 0))
    y = header.height
    for block in blocks:
        out.paste(block, (0, y))
        y += block.height
    return out


def table(columns, faces):
    fmt = lambda v: '—' if v is None else f'{v:.2f}'
    lines = ['| Face | Colonne | PSNR | ' + ' | '.join(REGIONS) + ' | alpha < 0,5 |',
             '|---|---|---:|' + '---:|' * len(REGIONS) + '---:|']
    for face in faces:
        for column in columns:
            f = column['faces'][face]
            lines.append(f"| {face} | {column['label']} | {fmt(f['psnr'])} | "
                         + ' | '.join(fmt(f['regions'].get(r, {}).get('psnr')) for r in REGIONS)
                         + f" | {f['low_alpha_fraction_of_valid']:.3f} |")
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(description='Comparative sheet of validation faces')
    parser.add_argument('--prep', required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--columns', nargs='+', required=True, help='<training>:<checkpoint file>')
    parser.add_argument('--faces', nargs='*', help='Subset of validation faces (default: all)')
    parser.add_argument('--size', type=int, default=256)
    args = parser.parse_args(argv)
    try:
        columns, faces = load_columns(args.prep, args.columns)
        if args.faces:
            unknown = set(args.faces) - set(faces)
            if unknown:
                raise RuntimeError(f'unknown faces {sorted(unknown)}')
            faces = [f for f in faces if f in args.faces]
        target = Path(args.prep) / 'comparisons' / args.name
        target.mkdir(parents=True, exist_ok=True)
        sheet(columns, faces, args.size).save(target / 'validation_sheet.jpg', quality=90)
        for face in faces:
            face_block(face, columns, args.size * 2).save(
                target / f"face__{face.replace('/', '__').removesuffix('.png')}.jpg", quality=92)
        lines = [f'# Planche comparative — {args.name} (validation uniquement)', '',
                 'Colonnes : ' + ', '.join(c['label'] for c in columns) + '.',
                 'Lignes par face : rendu, carte d’erreur (pixels exclus en bleu), profondeur attendue '
                 '(les flottants y apparaissent comme des taches proches).', '',
                 '![Planche](validation_sheet.jpg)', '', *table(columns, faces), '',
                 'Résultats limités au panorama de validation ; aucun seuil de qualité.', '']
        (target / 'validation_sheet.md').write_text('\n'.join(lines))
        write(target / 'validation_sheet.json', {'columns': args.columns, 'faces': faces, 'created_at': now()})
    except Exception as error:
        print(f'ERROR: {error}', file=sys.stderr)
        return 1
    print(target / 'validation_sheet.md')
    return 0


if __name__ == '__main__':
    sys.exit(main())
