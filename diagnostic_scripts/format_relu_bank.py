#!/usr/bin/env python3
"""Export shared-Vin ReLU CSVs as 0906-compatible paired MC curves.

All numeric samples are copied verbatim. Header supply values are explicit
user-approved midpoint references, not recovered simulator metadata. Runtime
loaders, not this exporter, subtract half the referenced supply.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import tempfile


def format_bank(source, output, supplies=(1.176, 1.2, 1.224)):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    files = sorted(source.glob('*.csv'))
    if len(files) != 45:
        raise ValueError('Expected exactly 45 corner CSV files')
    manifest = dict(source=str(source), supplies_by_voltage_category=list(supplies),
        supply_metadata_provenance='User-approved 0906 midpoint references, consistent with observed 1022 negative-input baselines; not original simulation metadata',
        samples='Copied verbatim, no interpolation, scaling, offset subtraction or resampling', files=[])
    seen = set()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.relu_format_', dir=output.parent) as temporary:
        staged = Path(temporary)/'bank'
        staged.mkdir()
        for path in files:
            match = re.fullmatch(r'(tt|ff|ss|fs|sf)_(-20|25|85)_([0-2])',path.stem)
            if match is None:
                raise ValueError(f'Unrecognized PVT filename: {path.name}')
            seen.add(path.stem)
            vdd, temperature = supplies[int(match[3])], int(match[2])
            raw = path.read_bytes()
            reader = csv.reader(io.StringIO(raw.decode('utf-8')))
            header = next(reader)
            if header != ['Vin', *[f'VOUT_mc_{i:03d}' for i in range(1,101)]]:
                raise ValueError(f'Expected Vin plus VOUT_mc_001..100: {path}')
            rows = [row for row in reader if row]
            if len(rows) != 201 or any(len(row)!=101 for row in rows):
                raise ValueError(f'Expected 201 points for each of 100 curves: {path}')
            destination = staged/path.name
            with destination.open('w',newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow([
                    f'/Vout (VDD={vdd:g},temperature={temperature},mcparamset={i}) {axis}'
                    for i in range(1,101) for axis in ('X','Y')])
                writer.writerows([v for y in row[1:] for v in (row[0],y)] for row in rows)
            with destination.open(newline='') as stream:
                exported = list(csv.reader(stream))[1:]
            assert len(exported) == len(rows)
            assert all(out==[v for y in row[1:] for v in (row[0],y)] for out,row in zip(exported,rows))
            manifest['files'].append(dict(name=path.name,points_per_curve=201,curves=100,
                source_sha256=hashlib.sha256(raw).hexdigest(),
                output_sha256=hashlib.sha256(destination.read_bytes()).hexdigest()))
        assert len(seen) == 45
        (staged/'format_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        staged.rename(output)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=format_bank(args.source,args.output)
    print(f"Formatted {len(result['files'])} files, 100 curves/file, 201 points/curve: {args.output}")


if __name__=='__main__':
    main()
