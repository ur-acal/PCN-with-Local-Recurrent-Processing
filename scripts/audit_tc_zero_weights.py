"""Read-only 5-bit PCN zero-coordinate audit; no forward run or checkpoint writes."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from ode_pc import SymQuantizeWeight


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', default='results/tc_zero_weight_audit')
    p.add_argument('--weight-quant-factor-bits', type=int, default=None,
                   help='Match the audited run; omit for unquantized max-abs scale.')
    args = p.parse_args()
    path, out = Path(args.checkpoint), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    state = ckpt['net']
    flat_path = Path(str(path).replace('_full_param_best', '_best'))
    assert flat_path != path and flat_path.exists()
    flat = torch.load(flat_path, map_location='cpu', weights_only=False)['net']
    arch = ckpt['init_args']['model_args']
    masks, arrays, rows = {}, {}, []
    with gzip.open(out/'zero_positions.csv.gz', 'wt') as stream:
        writer = csv.writer(stream)
        writer.writerow(['layer_zero_based','branch','storage_outer','storage_inner','kernel_y','kernel_x',
                         'runtime_conv_out','runtime_conv_in','runtime_kernel_y','runtime_kernel_x'])
        for layer,(inp,output) in enumerate(zip(arch['inp_channels'],arch['out_channels'])):
            for branch in ('FFconv','FBconv'):
                key=f'PcConvs.{layer}.{branch}'
                weight=state[key+'.parametrizations.weight.original']
                assert int(state[key+'.parametrizations.weight.0.w_bits']) == 5,key
                quantizer=SymQuantizeWeight(w_bits=5,weight_quant_factor_bits=args.weight_quant_factor_bits)
                quantizer.compute_s(weight)
                with torch.no_grad(): mask=(quantizer(weight)==0).numpy()
                # Current QAT recomputes its max-abs scale. Verify every position
                # against the separately saved baked-in quantized checkpoint.
                assert np.array_equal(mask,(flat[key+'.weight']==0).numpy()),key
                assert mask.shape == (output,inp,3,3)
                masks[(layer,branch)]=mask
                coords=np.argwhere(mask).astype(np.int16)
                arrays[f'L{layer}_{branch}_zero_mask']=mask
                arrays[f'L{layer}_{branch}_zero_coordinates']=coords
                for a,b,y,x in coords.tolist():
                    # Existing replace_transpose_conv swaps channel axes and
                    # flips both kernel axes for FB. No spatial unrolling here.
                    runtime=(a,b,y,x) if branch=='FFconv' else (b,a,2-y,2-x)
                    writer.writerow([layer,branch,a,b,y,x,*runtime])
                rows.append(dict(layer=layer,branch=branch,channels=f'{inp}->{output}',
                    total=int(mask.size),zeros=int(mask.sum()),fraction=float(mask.mean()),
                    quantization_scale=float(quantizer.s_w),
                    original_fp_exact_zeros=int((weight==0).sum()),
                    all_zero_3x3_kernels=int(mask.all(axis=(2,3)).sum())))
    with (out/'per_tensor.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    overlap=[]
    for width in (24,48,96):
        layers=[i for i,c in enumerate(arch['out_channels']) if c==width]
        regular=[i for i in layers if arch['inp_channels'][i]==width]
        common=min(arch['inp_channels'][i] for i in layers)
        for case,selected,channels in [('without_expansion',regular,width),
                                       ('without_expansion_common_crop',regular,common),
                                       ('with_expansion',layers,common)]:
            for branch in ('FFconv','FBconv','FF_and_FB'):
                branches=('FFconv','FBconv') if branch=='FF_and_FB' else (branch,)
                stack=np.stack([masks[(i,b)][:,:channels] for i in selected for b in branches])
                count=stack.sum(axis=0);shared=count==len(stack)
                key=f'C{width}_{case}_{branch}'
                arrays[key+'_zero_multiplicity']=count.astype(np.uint8)
                arrays[key+'_common_zero_coordinates']=np.argwhere(shared).astype(np.int16)
                overlap.append(dict(pattern=width,case=case,branch=branch,layers=selected,
                    shape=list(shared.shape),compared_positions=int(shared.size),
                    common_zero_count=int(shared.sum()),common_zero_fraction=float(shared.mean()),
                    all_zero_3x3_kernels=int(shared.all(axis=(2,3)).sum()),
                    all_zero_outer_channels=np.flatnonzero(shared.all(axis=(1,2,3))).tolist(),
                    all_zero_inner_channels=np.flatnonzero(shared.all(axis=(0,2,3))).tolist(),
                    multiplicity_histogram=np.bincount(count.ravel(),minlength=len(stack)+1).tolist(),
                    common_zero_coordinates=np.argwhere(shared).tolist()))
    np.savez_compressed(out/'zero_masks_and_overlap.npz',**arrays)
    result=dict(checkpoint=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                quantized_checkpoint=str(flat_path.resolve()),checkpoint_acc=ckpt['acc'],epoch=ckpt['epoch'],
                weight_quant_factor_bits=args.weight_quant_factor_bits,
                zero_definition='Exactly zero after existing 5-bit SymQuantizeWeight, scale recomputed with specified weight_quant_factor_bits; all masks match flattened checkpoint.',
                coordinate_definition='Checkpoint storage [outer,inner,ky,kx]; FF=[out,in], FB transpose=[in,out]. Runtime FB swaps channels and flips kernel axes.',
                comparison='Index-aligned, no learned-channel permutation; expansion included on common coordinates only; absent channels are NOT zeros.',
                layers=rows,overlap=overlap)
    (out/'audit.json').write_text(json.dumps(result,indent=2))
    lines=['# PCN zero-weight audit','',f'Checkpoint: `{path.resolve()}`','',
           f'Weight-quantization-factor bits: `{args.weight_quant_factor_bits}`. Saved accuracy: {100*ckpt["acc"]:.2f}%; epoch {ckpt["epoch"]}.','',
           'Exactly-zero **5-bit quantized** weights, before hardware mismatch/noise. Every mask matches the saved flattened checkpoint.',
           'Layer indices are zero-based. Fractions exclude the final digital classifier. No convolution unrolling/padding zeros.', '',
           '| Layer | Channel pattern | FF zeros / total (%) | FB zeros / total (%) | Combined zero % |',
           '|---|---|---:|---:|---:|']
    for i in range(len(arch['inp_channels'])):
        ff,fb=[r for r in rows if r['layer']==i]
        lines.append(f"| {i} | {ff['channels']} | {ff['zeros']}/{ff['total']} ({100*ff['fraction']:.2f}%) | {fb['zeros']}/{fb['total']} ({100*fb['fraction']:.2f}%) | {100*(ff['zeros']+fb['zeros'])/(ff['total']+fb['total']):.2f}% |")
    lines+=['','## Positions zero in every layer of a pattern','',
            'FF and FB are compared separately in checkpoint storage coordinates. Joint means the same stored coordinate is zero in BOTH branches in ALL selected layers.',
            'With expansion: restrict regular layers to inner-channel indices 0:4, 0:24, or 0:48, respectively. Missing channels are excluded, never counted as zero.',
            'This is index alignment, not a claim that channel indices represent identical learned features across layers. A whole zero kernel means all nine spatial coefficients are zero.', '',
            '| Pattern | Case | Compared layers | Coordinate domain | FF common zeros | FB common zeros | Joint |',
            '|---|---|---|---|---:|---:|---:|']
    for width in (24,48,96):
        for case in ('without_expansion','without_expansion_common_crop','with_expansion'):
            items=[r for r in overlap if r['pattern']==width and r['case']==case]
            vals=[f"{r['common_zero_count']}/{r['compared_positions']} ({100*r['common_zero_fraction']:.4f}%)" for r in items]
            lines.append(f"| {width} | {case} | {items[0]['layers']} | {items[0]['shape']} | {' | '.join(vals)} |")
    structured=any(r['all_zero_3x3_kernels'] or r['all_zero_outer_channels'] or r['all_zero_inner_channels'] for r in overlap)
    lines+=['','## Main finding','',
            f"Overall: {sum(r['zeros'] for r in rows)}/{sum(r['total'] for r in rows)} quantized coefficients are zero ({100*sum(r['zeros'] for r in rows)/sum(r['total'] for r in rows):.4f}%).",
            f"Original floating-point exact zeros: {sum(r['original_fp_exact_zeros'] for r in rows)}.",
            ('Structured common-zero regions exist; see audit.json.' if structured else
             'No complete 3x3 kernel, outer-channel row, or inner-channel column is zero across every selected layer, in any pattern/case/branch. Common zeros are individual coefficient positions rather than removable whole kernels/channels.'),
            '', '## Complete coordinate exports','',
            '- `zero_positions.csv.gz`: every zero coefficient per layer/branch, with checkpoint and runtime conv2d coordinates.',
            '- `zero_masks_and_overlap.npz`: full boolean masks, zero-coordinate arrays, and per-coordinate number of layers with zero weights.',
            '- `audit.json`: all common-zero coordinates, zero-kernel/channel counts and multiplicity histograms.',
            '- `per_tensor.csv`: counts/fractions and original floating-point exact-zero counts.', '',
            'Reproduce: `python scripts/audit_tc_zero_weights.py --checkpoint <full_param_best_ckpt.pth>' +
            (f' --weight-quant-factor-bits {args.weight_quant_factor_bits}' if args.weight_quant_factor_bits is not None else '') +
            f' --output {out}`.',
            'No checkpoint, model implementation, or training process was modified.']
    (out/'summary.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__=='__main__':main()
