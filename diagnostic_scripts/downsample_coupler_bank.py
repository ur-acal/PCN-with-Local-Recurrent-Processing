#!/usr/bin/env python3
"""Select native rows and export a shared-Vin coupler bank as paired X/Y CSVs.

Numeric CSV fields are copied verbatim, never reformatted or interpolated.
Both endpoints and the existing zero-voltage row are retained. A manifest
records row indices and source/output hashes; source files are read-only.
"""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import tempfile


def select_rows(voltages, points=150):
    if points < 3 or points > len(voltages):
        raise ValueError("Point count must be between 3 and the original row count")
    if not all(math.isfinite(x) for x in voltages) or any(
            a >= b for a, b in zip(voltages, voltages[1:])):
        raise ValueError("Source voltage grid must be finite and strictly increasing")
    zero = min(range(len(voltages)), key=lambda i: abs(voltages[i]))
    if abs(voltages[zero]) > 1e-12:
        raise ValueError("Source grid must contain an existing zero-voltage sample")
    selected = [round(i * (len(voltages)-1) / (points-1)) for i in range(points)]
    if zero not in selected:
        nearest = min(range(1, points-1), key=lambda i: abs(voltages[selected[i]]))
        selected[nearest] = zero
    selected.sort()
    assert len(set(selected)) == points
    assert selected[0] == 0 and selected[-1] == len(voltages)-1 and zero in selected
    return selected


def export_bank(source, output, points=150):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    files = sorted(source.glob('*.csv'))
    if not files:
        raise ValueError(f"No source CSVs in {source}")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = dict(source=str(source), points_per_curve=points, curves_per_file=100,
        selection='rounded uniform row indices, replacing the nearest interior selection with the existing zero row',
        numeric_fields='verbatim source strings; no interpolation, scaling, averaging or rounding',
        original_grid=None, selected_row_indices_zero_based=None, files=[])
    with tempfile.TemporaryDirectory(prefix='.coupler_downsample_', dir=output.parent) as temporary:
        staged = Path(temporary)/'bank'
        staged.mkdir()
        for path in files:
            raw = path.read_bytes()
            reader = csv.reader(io.StringIO(raw.decode('utf-8')))
            header = next(reader)
            if header[0].strip().lower() != 'vin' or len(header) != 101:
                raise ValueError(f"Expected Vin plus 100 conductance columns: {path}")
            ids = [int(re.fullmatch(r'Gm(?:_full)?_([0-9]+)', name).group(1))
                   for name in header[1:]]
            if set(ids) != set(range(1,101)):
                raise ValueError(f"Expected curve IDs 1..100: {path}")
            rows = [row for row in reader if row]
            if any(len(row) != 101 for row in rows):
                raise ValueError(f"Invalid data row width: {path}")
            grid = [float(row[0]) for row in rows]
            indices = select_rows(grid, points)
            if manifest['original_grid'] is None:
                manifest['original_grid'] = grid
                manifest['selected_row_indices_zero_based'] = indices
            elif grid != manifest['original_grid'] or indices != manifest['selected_row_indices_zero_based']:
                raise ValueError(f"All corner files must share the same source grid: {path}")
            destination = staged/path.name
            with destination.open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow([f'Gm_full (mcparamset={i}) {axis}' for i in ids for axis in ('X','Y')])
                for index in indices:
                    row = rows[index]
                    if not all(math.isfinite(float(value)) for value in row):
                        raise ValueError(f"Non-finite selected sample: {path}")
                    writer.writerow([value for y in row[1:] for value in (row[0],y)])
            with destination.open(newline='') as stream:
                exported = list(csv.reader(stream))[1:]
            assert all(out == [v for y in rows[i][1:] for v in (rows[i][0],y)]
                       for out,i in zip(exported,indices))
            manifest['files'].append(dict(name=path.name, source_rows=len(rows),
                source_sha256=hashlib.sha256(raw).hexdigest(),
                output_sha256=hashlib.sha256(destination.read_bytes()).hexdigest()))
        (staged/'downsampling_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        staged.rename(output)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--points', type=int, default=150)
    args = parser.parse_args()
    manifest = export_bank(args.source, args.output, args.points)
    print(f"Exported {len(manifest['files'])} files, 100 curves/file, {args.points} native samples/curve to {args.output}")


if __name__ == '__main__':
    main()
