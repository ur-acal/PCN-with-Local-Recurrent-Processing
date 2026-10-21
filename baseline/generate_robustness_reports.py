#!/usr/bin/env python3
"""Regenerate paper tables and PDFs from recorded results; never run inference.

Extra controls: --controls-source DIRECTORY_OR_TAR (repeatable).
Extra PCN paired evaluations: --pcn-bn-root DIRECTORY (repeatable).
Original-WRN paired unfused evaluations: --baseline-paired-root DIRECTORY.
Arbitrary future curves: --extra-data CSV --extra-registry JSON. CSV columns:
dataset,architecture,mismatch_type,level,variant,accuracy,std (optional).
Registry is {ID: {name,family,base,bn,bias,pooling,recipe,policy}}.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import re
import statistics
import sys
import tarfile
import textwrap
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from baseline.wrn_control_specs import SIZES, DATASETS, parse_size
EVIDENCE = ROOT.parent / 'papers/hardware-native-neural-ode/shared/evidence/generic-mismatch'
PAIRS = [(d, 'WRN_' + s) for d in ('cifar10', 'cifar100')
         for s in ('16_2', '16_4', '28_2', '28_4')]
TYPES = ('max_additive', 'multiplicative', 'rms_additive')
ENDPOINTS = dict(max_additive=.1, multiplicative=.4, rms_additive=1.25)
SEVERE_LEVELS = dict(max_additive=(.08, .09, .1),
                     multiplicative=(.3, .35, .4),
                     rms_additive=(.75, 1., 1.25))
RANKING_MODES = ('dataset-severe-mean', 'global-endpoint', 'per-point')
GRIDS = dict(max_additive={n / 100 for n in range(11)},
             multiplicative={n / 100 for n in range(0, 41, 5)},
             rms_additive={.25, .5, .75, 1., 1.25})
COMMON = '300 epochs; BS128; SGD momentum .9; cosine; 5 warmup epochs; TIMM CIFAR augmentation'
PCN_SECONDARY_LINESTYLES = [
    '--', ':', '-.', (0, (5, 1)), (0, (3, 1, 1, 1)),
    (0, (1, 1)), (0, (5, 2, 1, 2)), (0, (3, 2)),
]


def number(x):
    try:
        x = float(x)
        return x if math.isfinite(x) else None
    except (ValueError, TypeError):
        return None


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '|' + '|'.join(['---'] * len(headers)) + '|'] +
                     ['| ' + ' | '.join(map(str, r)) + ' |' for r in rows]) + '\n\n'


def kind(r):
    t = r['mismatch_type']
    if t in TYPES:
        return t
    if t == 'additive_max':
        return 'max_additive'
    if t in ('additive_rms', 'rms'):
        return 'rms_additive'
    if t == 'additive':
        return {'rms': 'rms_additive', 'max_abs': 'max_additive'}.get(
            r.get('additive_scale_mode', 'max_abs'), 'unsupported')
    return 'unsupported'


def top3_plot_folder(pcn_ids):
    if pcn_ids == ['P0', 'P3']:
        return 'pcn_vs_top3_baseline_wrns'
    parts = [re.sub(r'[^a-z0-9-]+', '-', id_.lower()).strip('-') for id_ in pcn_ids]
    if any(not part for part in parts):
        raise ValueError('PCN IDs must contain a filename-safe letter or number')
    return '_'.join(parts) + '_vs_top3_baseline_wrns'


def pcn_linestyles(pcn_ids):
    if not pcn_ids:
        return {}
    solid_id = 'P0' if 'P0' in pcn_ids else pcn_ids[0]
    styles = {solid_id: '-'}
    secondary_index = 0
    for id_ in pcn_ids:
        if id_ != solid_id:
            styles[id_] = PCN_SECONDARY_LINESTYLES[
                secondary_index % len(PCN_SECONDARY_LINESTYLES)]
            secondary_index += 1
    return styles


class Results:
    def __init__(self, exclude_avg=False, pairs=None,
                 ranking_mode='dataset-severe-mean'):
        self.meta, self.values, self.sources = {}, {}, set()
        self.recipes = {}
        self.exclude_avg = exclude_avg
        self.complete_wrns = None
        self.pairs = PAIRS if pairs is None else pairs
        if ranking_mode not in RANKING_MODES:
            raise ValueError(f'Unknown WRN ranking mode: {ranking_mode}')
        self.ranking_mode = ranking_mode

    def require_complete(self, types):
        """All selected models must have every requested grid, not just endpoints."""
        self.complete_wrns = {
            i for i, m in self.meta.items() if m['family'] == 'wrn' and
            all(GRIDS[t] <= {l for l, _ in self.curve(i, d, a, t)}
                for d, a in self.pairs for t in types)}

    def label(self, i):
        pcn = {
            'P0': 'PCN_noX (PCNetNoBatchNorm / ODEXInitFFFB)',
            'P1': 'PCN_noX_1stconv (PCNetWith1stConv / ODEXInitFFFB)',
            'P2': 'PCN_X_ConvInit (PCNetNoBatchNorm / ODEBlockPC)',
            'P3': 'PCN_X_XInit (PCNetNoBatchNorm / ODEBlockXInit)',
            'PB-F': 'PCNetBoundaryBN / ODEXInitFFFB / frozen',
            'PB-R': 'PCNetBoundaryBN / ODEXInitFFFB / recal',
        }
        if i in pcn:
            return pcn[i]
        m = self.meta[i]
        if 'label' in m:
            return m['label']
        if m['base'] in ('W0', 'W1', 'W2'):
            wd = '5e-4' if m['base'] == 'W0' else '1e-3'
            drop = '0p25' if m['base'] == 'W2' else '0'
            base = f'std_BN_wd{wd}_lr1e-1_fd{drop}'
        elif i == 'H0':
            return 'std_noBN-bias_tuned'
        elif re.fullmatch(r'R\d+', m['base']):
            row = int(m['base'][1:])
            structure = ('avgSC' if row in (4, 5, 6) else 'avgMain' if row in (7, 8, 9, 17)
                         else 'maxSC' if row in (11, 12, 13) else 'maxMain' if row in (14, 15, 16, 18) else 'std')
            bn = 'BN' if m['bn'] else 'noBN-bias' if m['bias'] else 'noBN-nobias'
            low = row in (10, 17, 18)
            base = f'{structure}_{bn}_wd1e-3_lr1e-{2 if low else 1}_fd0p25'
            if low:
                base += '_ptinit'
        else:
            base = m['name']
        suffix = {'folded-frozen': 'fold-frozen', 'folded-recal': 'fold-recal',
                  'none': '', 'frozen': 'frozen', 'recal': 'recal'}[m['policy']]
        return base + ('_' + suffix if suffix else '')

    def register(self, id_, name, family='wrn', base=None, bn=True, bias=False,
                 pooling='none', recipe='', policy='recal'):
        self.meta[id_] = dict(name=name, family=family, base=base or id_, bn=bn,
                              bias=bias, pooling=pooling, recipe=recipe, policy=policy)

    def put(self, id_, d, a, t, level, value, std, source):
        v = number(value)
        if v is None:
            return
        if not 0 <= v <= 100:
            raise ValueError(f'Invalid accuracy {source}: {v}')
        key = (d, a, t, round(float(level), 8), id_)
        old = self.values.get(key)
        # Raw precision may replace rounded report values, never different runs.
        if old and abs(old[0] - v) > .011:
            raise ValueError(f'Conflicting results for {key}: {old[0]} vs {v}, {source}')
        self.values[key] = (v, number(std), str(source))
        self.sources.add(str(source))

    def eligible(self, id_):
        m = self.meta[id_]
        if m['family'] == 'wrn' and self.complete_wrns is not None and id_ not in self.complete_wrns:
            return False
        return not (self.exclude_avg and m['family'] == 'wrn' and m['pooling'] == 'avg')

    def curve(self, id_, d, a, t):
        return sorted((k[3], v) for k, v in self.values.items()
                      if k[:3] == (d, a, t) and k[4] == id_)

    def ids(self, d, a, t, family=None):
        return sorted({k[4] for k in self.values if k[:3] == (d, a, t)
                       and self.eligible(k[4]) and
                       (family is None or self.meta[k[4]]['family'] == family)})

    def top(self, t, policy='recal', dataset=None, architecture=None, level=None):
        if self.ranking_mode == 'dataset-severe-mean':
            if dataset is None:
                datasets = {d for d, _ in self.pairs}
                if len(datasets) != 1:
                    raise ValueError('dataset-severe-mean ranking requires a dataset')
                dataset = next(iter(datasets))
            points = [(d, a, l) for d, a in self.pairs if d == dataset
                      for l in SEVERE_LEVELS[t]]
        elif self.ranking_mode == 'global-endpoint':
            points = [(d, a, ENDPOINTS[t]) for d, a in self.pairs]
        else:
            if dataset is None or architecture is None or level is None:
                raise ValueError('per-point ranking requires dataset, architecture and level')
            points = [(dataset, architecture, round(float(level), 8))]
        scores = []
        for id_, m in self.meta.items():
            if m['family'] != 'wrn' or not self.eligible(id_):
                continue
            if policy is not None and m['policy'] != policy:
                continue
            values = [self.values.get((d, a, t, l, id_)) for d, a, l in points]
            if all(values):
                scores.append((statistics.mean(x[0] for x in values), id_))
        return sorted(scores, key=lambda x: (-x[0], x[1]))[:3]

    def selected_top(self, d, a, t):
        if self.ranking_mode != 'per-point':
            return self.top(t, dataset=d)
        selected = []
        selected_ids = set()
        levels = sorted({l for i in self.ids(d, a, t, 'wrn')
                         for l, _ in self.curve(i, d, a, t)})
        for level in levels:
            for score, id_ in self.top(t, dataset=d, architecture=a, level=level):
                if id_ not in selected_ids:
                    selected.append((score, id_))
                    selected_ids.add(id_)
        return selected

    def top3_baseline_ids(self, d, a, t, pcn_ids=('P0', 'P3')):
        top = self.selected_top(d, a, t)
        if len(top) < 3 or 'W0-R' not in self.meta or not self.eligible('W0-R'):
            raise ValueError('Top-three view requires three complete recalibrated WRNs and the W0-R baseline')
        return list(dict.fromkeys([*pcn_ids, *[i for _, i in top], 'W0-R']))

    def paired_ids(self, d, a, t):
        ids = []
        for base in ['W0'] + [self.meta[id_]['base'] for _, id_ in self.selected_top(d, a, t)]:
            candidates = [i for i, m in self.meta.items() if m['base'] == base]
            for policy in ('frozen', 'recal', 'none'):
                selected = [i for i in candidates if self.meta[i]['policy'] == policy]
                # The historical baseline has only folded frozen measurements.
                if not selected and policy == 'frozen':
                    selected = [i for i in candidates if self.meta[i]['policy'] == 'folded-frozen']
                for i in sorted(selected):
                    if i not in ids and self.eligible(i):
                        ids.append(i)
        return ids

    def matched(self, d, a, t):
        found = []
        for rows in ((4, 5, 6), (7, 8, 9), (11, 12, 13), (14, 15, 16)):
            recipes = [self.recipes.get((f'R{r}', d, a)) for r in rows]
            if not all(recipes) or any(r != recipes[0] for r in recipes):
                continue
            ids = [f'R{rows[0]}-F', f'R{rows[0]}-R', f'R{rows[1]}-N', f'R{rows[2]}-N']
            curves = [self.curve(i, d, a, t) for i in ids]
            levels = [{x for x, _ in c} for c in curves]
            if all(curves) and all(self.eligible(i) for i in ids) and all(s == levels[0] for s in levels) and ENDPOINTS[t] in levels[0]:
                found.extend(ids)
        return found


def load_wide(data, path):
    rows = list(csv.DictReader(path.open()))
    pcn = ['PCNetNoBatchNorm', 'PCNetWith1stConv (new SLURM)',
           'PCNetNoBatchNorm ODEBlockPC t1.75', 'PCNetNoBatchNorm ODEBlockXInit t1.75']
    mapping = {}
    for n, name in enumerate(pcn):
        id_ = f'P{n}'
        data.register(id_, name, 'pcn', bn=False, pooling='max', policy='none',
                      recipe='LR .01; WD .001; final dropout .25; PyTorch-default init; ' + COMMON)
        mapping[name] = id_
    for base, prefix, recipe in [('W0', '', 'LR .1; WD .0005; final dropout 0'),
                                  ('W1', 'WD1e-3 ', 'LR .1; WD .001; final dropout 0')]:
        for suffix, policy in [('folded frozen-BN', 'folded-frozen'),
                               ('folded recal-BN', 'folded-recal'), ('unfused recal-BN', 'recal')]:
            name = prefix + 'WRN ' + suffix
            id_ = base + '-' + {'folded-frozen': 'FF', 'folded-recal': 'FR', 'recal': 'R'}[policy]
            data.register(id_, name, base=base, recipe=recipe + '; WRN init; ' + COMMON, policy=policy)
            mapping[name] = id_
    name = 'WD1e-3 final-drop0.25 WRN unfused recal-BN'
    data.register('W2-R', name, base='W2', recipe='LR .1; WD .001; final dropout .25; WRN init; ' + COMMON)
    mapping[name] = 'W2-R'
    names = ['BN-free WRN', 'AvgPool BN', 'Stride-2 BN', 'AvgPool BN-free', 'Stride-2 BN-free',
             'BN-free/no-conv-bias, WD5e-4/drop0', 'BN-free/no-conv-bias, WD1e-3/final-drop0.25']
    for n, name in enumerate(names):
        bn = name in ('AvgPool BN', 'Stride-2 BN')
        recipe = ('WD .001; final dropout .25; ' if 'Pool' in name or 'Stride' in name else '')
        recipe += 'Historical resolved recipe: see paper_handoff_max_mul.md and source manifest; not assumed matched'
        data.register(f'H{n}', name, bn=bn, bias=not bn and 'no-conv-bias' not in name,
                      pooling='avg' if n in (1, 2, 3, 4) else 'none', recipe=recipe,
                      policy='recal' if bn else 'none')
        mapping[name] = f'H{n}'
    for r in rows:
        for name, id_ in mapping.items():
            data.put(id_, r['dataset'], r['architecture'], kind(r), r['level'], r.get(name), None, path)


def aggregate(data, rows, base, source):
    for r in rows:
        for suffix, col, stdcol in (
                ('F', 'paired_wrn_frozen_bn_mean_accuracy', 'paired_wrn_frozen_bn_std_accuracy'),
                ('R', 'wrn_recalibrated_bn_mean_accuracy', 'wrn_recalibrated_bn_std_accuracy'),
                ('N', 'wrn_nobn_mean_accuracy', 'wrn_nobn_std_accuracy')):
            if r.get(col):
                id_ = base + '-' + suffix
                if id_ not in data.meta:
                    meta = dict(data.meta[base + '-R'])
                    meta['policy'] = 'frozen'
                    meta['name'] = meta['name'].replace('unfused recal-BN', 'unfused frozen-BN')
                    data.meta[id_] = meta
                data.put(id_, r['dataset'], r['architecture'], kind(r), r['mismatch_level'],
                         r[col], r.get(stdcol), source)


def load_controls(data, path):
    from baseline.wrn_control_specs import ROWS, model_options, training_override
    for row, (bn, bias, shortcut, main) in ROWS.items():
        opts = model_options(row)
        pool = opts.get('pool_type', 'avg') if shortcut else 'none'
        structure = (f'{pool} main + padding shortcut' if main else
                     f'stride-2 main + {pool}/padding shortcuts' if shortcut else
                     'stride-2 main + learned shortcuts')
        recipe = training_override(row) + f'; init={opts["init_mode"]}; ' + COMMON
        for suffix, policy in ([('F', 'frozen'), ('R', 'recal')] if bn else [('N', 'none')]):
            data.register(f'R{row}-{suffix}', f'Row {row}: {structure}', base=f'R{row}',
                          bn=bn, bias=bias, pooling=pool, recipe=recipe, policy=policy)
    def consume(name, handle):
        match = re.search(r'(?:^|/)row(\d+)/', name)
        if match and name.endswith('/baseline_config.json'):
            parts = name[match.start():].lstrip('/').split('/')
            cfg = json.load(handle)['cfg']
            signature = {k: v for k, v in cfg.items() if k != 'model_name'}
            data.recipes['R' + match[1], parts[1], 'WRN_' + parts[2]] = signature
        elif match and name.endswith('/full_aggregate.csv'):
            aggregate(data, csv.DictReader(handle), 'R' + match[1], str(path) + ':' + name)
    if path.is_dir():
        paths = list(path.glob('row*/**/evaluation/*/full_aggregate.csv')) + list(path.glob('row*/**/baseline_config.json'))
        for p in sorted(paths):
            with p.open() as f:
                consume(str(p.relative_to(path)), f)
    elif path.exists():
        with tarfile.open(path) as tar:
            for member in tar.getmembers():
                if member.isfile() and (member.name.endswith('/baseline_config.json') or
                                        '/evaluation/' in member.name and member.name.endswith('/full_aggregate.csv')):
                    with io.TextIOWrapper(tar.extractfile(member)) as f:
                        consume(member.name, f)


def load_baseline_paired(data, root):
    """Load the original WRN's paired unfused frozen/recalibrated results."""
    if not root.exists():
        return
    for path in sorted(root.glob('*/*/WRN_*/full_aggregate.csv')):
        condition, dataset, architecture = path.relative_to(root).parts[:3]
        with path.open() as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            raise ValueError(f'Empty baseline paired results: {path}')
        for row in rows:
            if (row['dataset'], row['architecture'], kind(row)) != (dataset, architecture, condition):
                raise ValueError(f'Wrong baseline paired identity: {path}')
        aggregate(data, rows, 'W0', path)


def load(args):
    data = Results(args.exclude_avgpool_wrn, [(d, 'WRN_' + s) for d in args.datasets for s in args.sizes])
    load_wide(data, args.wide_csv)
    # Import RMS from the earlier complete report without overriding max/mul.
    rms = ROOT / 'logs/combined_pcn_wrn_mismatch_finaldrop025_complete/full_accuracy_comparison.csv'
    if rms.exists():
        mapping = {m['name']: i for i, m in data.meta.items()}
        for r in csv.DictReader(rms.open()):
            if kind(r) == 'rms_additive':
                for name, id_ in mapping.items():
                    data.put(id_, r['dataset'], r['architecture'], 'rms_additive', r['level'], r.get(name), None, rms)
    for path in args.controls_source:
        load_controls(data, path)
    for root in args.baseline_paired_root:
        load_baseline_paired(data, root)
    for p in sorted((ROOT / 'logs/wrn_wd1e3_finaldrop025_mismatch_full').glob('*/unfolded/*/*/full_aggregate.csv')):
        with p.open() as f:
            aggregate(data, csv.DictReader(f), 'W2', p)
    for root in args.pcn_bn_root:
        for p in sorted(root.glob('*/*/*/full_aggregate.csv')):
            d, tag, t = p.relative_to(root).parts[:3]
            match = re.search(r'WRN_([1-9][0-9]*)_([1-9][0-9]*)(?![0-9])', tag)
            if not match or t not in TYPES:
                continue
            parse_size(match[1] + '_' + match[2])
            for r in csv.DictReader(p.open()):
                for mode, label in [('frozen', 'F'), ('recalibrated', 'R')]:
                    id_ = 'PB-' + label
                    data.register(id_, 'PCNetBoundaryBN no-X', 'pcn', base='PB', bn=True,
                                  pooling='max', policy='frozen' if label == 'F' else 'recal',
                                  recipe='LR .01; WD .001; final dropout .25; PyTorch-default init; ' + COMMON)
                    data.put(id_, d, match[0], t, r['level'], r[mode + '_mean_accuracy'],
                             r[mode + '_std_accuracy'], p)
    if args.extra_registry:
        data.meta.update(json.loads(args.extra_registry.read_text()))
    for p in args.extra_data:
        for r in csv.DictReader(p.open()):
            if r['variant'] not in data.meta:
                raise ValueError('Register variant first: ' + r['variant'])
            data.put(r['variant'], r['dataset'], r['architecture'], kind(r), r['level'],
                     r['accuracy'], r.get('std'), p)
    data.values = {k: v for k, v in data.values.items() if k[:2] in data.pairs}
    return data


def intro(data, args, title):
    if args.wrn_ranking == 'dataset-severe-mean':
        ranking = ('For each dataset and mismatch type, complete recalibrated-BN WRNs are ranked by mean '
                   'accuracy across all selected architectures and all three severe levels. '
                   'The top three remain fixed across architectures and levels.')
    elif args.wrn_ranking == 'global-endpoint':
        ranking = ('Legacy ranking: complete recalibrated-BN WRNs are ranked globally by mean endpoint '
                   f'accuracy across all {len(data.pairs)} selected dataset/size pairs.')
    else:
        ranking = ('Pointwise ranking: complete recalibrated-BN WRNs are ranked independently at every '
                   'dataset, architecture, mismatch type and level; selected-WRN views contain the union '
                   'of variants entering a pointwise top three.')
    return [f'# {title}\n',
            'Generated from recorded results, not new evaluations. Accuracy is percent; each point is a trial mean.\n',
            '[Variant definitions and PDF links](wrn_variants_and_plots.md). '
            '[Training/evaluation provenance](paper_handoff_max_mul.md).\n',
            'Generator: `' + str(Path(__file__).resolve()) + '`; wrapper: `' + str(ROOT / 'launch_scripts/generate_robustness_reports.sh') + '`.\n',
            ranking + ' The ranking is retrospective and favors WRN by selecting the strongest observed controls. '
            'Every reported WRN must have all selected models and complete requested evaluation grids; '
            'missing models or grid points exclude that evaluation variant globally. Completed test results '
            'provide evidence of an available trained model, not a fresh checkpoint-file audit. '
            'Best available is a per-level oracle restricted to these complete variants.\n',
            f'AvgPool WRN exclusion: **{args.exclude_avgpool_wrn}** (downsampling/shortcut pooling only, not GAP).\n',
            'BN parameters are excluded from mismatch in these comparisons; recalibration uses perturbed weights '
            'and training images per level/trial. Historical folded and unfused policies remain explicitly separate.\n']


def write_md(data, args, wrn=False):
    out = intro(data, args, 'WRN BN/bias and top-three comparison' if wrn else 'PCN versus top-three WRNs')
    if wrn:
        out += ['## Matched controls\n',
                'Only complete four-curve groups are included: BN-on/conv-bias-off frozen and recalibrated; '
                'BN-off/conv-bias-on; BN-off/conv-bias-off. Classifier bias remains enabled. '
                'Rows 4/5/6 use stride-2 main + AvgPool/padding shortcuts; rows 7/8/9 use AvgPool main + shortcuts. '
                'Rows 11/12/13 and 14/15/16 are the corresponding completed MaxPool controls. '
                'Matched recipes: LR .1, WD .001, final dropout .25, block dropout 0, WRN initialization; ' + COMMON + '.\n',
                'Results: `' + '`, `'.join(map(str, args.controls_source)) + '`. '
                'Plots: `plots/wrn_bn_bias/` and `plots/wrn_frozen_recal_bn/`.\n']
    else:
        out += ['Plots: `plots/pcn_vs_all_wrn/`; legend/recipe table: `wrn_variants_and_plots.md`.\n',
                'PCN uses recurrent FF/FB units, boundary MaxPool and a biased linear head; '
                'WRN uses two-convolution residual blocks, with variant-specific shortcuts, pooling and BN. '
                'Size pairs align widths/depth organization, not identical operations or parameter counts. '
                'PCN LR .01 and default initialization differ from most WRN controls; row 10 tests these jointly. '
                'Boundary-BN PCN and with-X PCNs remain separate curves, never replacements for the original PCN.\n']
    for t in args.mismatch_types:
        out.append(f'## {t}\n')
        if data.ranking_mode == 'dataset-severe-mean':
            for d in args.datasets:
                top = data.top(t, dataset=d)
                out += [f'### {d} fixed ranking\n', table(
                    ['Rank', 'Variant', 'Mean severe-level accuracy'],
                    [(n, data.label(i), f'{v:.2f}') for n, (v, i) in enumerate(top, 1)])]
        elif data.ranking_mode == 'global-endpoint':
            top = data.top(t)
            out.append(table(['Rank', 'Variant', 'Mean endpoint accuracy'],
                       [(n, data.label(i), f'{v:.2f}') for n, (v, i) in enumerate(top, 1)]))
        for d, a in data.pairs:
            selected = data.selected_top(d, a, t)
            ids = data.paired_ids(d, a, t) if wrn else data.ids(d, a, t, 'pcn') + [i for _, i in selected]
            out.append(f'### {d} / {a}\n')
            if wrn:
                matched = data.matched(d, a, t)
                if matched:
                    out.append('Matched BN/bias controls:\n')
                    out.append(curve_table(data, d, a, t, matched))
                baseline_note = ('W0-F is unfused frozen' if 'W0-F' in data.meta and data.eligible('W0-F')
                                 else 'W0-FF is folded frozen because W0-F is unavailable')
                out.append(f'Baseline plus selected top three (F=frozen, R=recalibrated; {baseline_note}):\n')
            out.append(curve_table(data, d, a, t, ids, oracle=not wrn))
    filename = 'wrn_bn_bias_and_top3_comparison.md' if wrn else 'pcn_vs_top3_wrn_max_mul.md'
    (args.output_dir / filename).write_text('\n'.join(out))
    if wrn:
        write_bnfree_md(data, args)


def write_bnfree_md(data, args):
    """Write the BN-free comparison using the active ranking policy."""
    bnfree = sorted(i for i, meta in data.meta.items()
                    if meta['family'] == 'wrn' and not meta['bn'] and data.eligible(i))
    selection_rows = []
    if data.ranking_mode == 'per-point':
        reference_name = 'Best pointwise top-three recal-BN WRN'
        reference_note = ('The recalibrated-BN reference is the best member of the top three selected '
                          'independently at each dataset, architecture, mismatch type and level.')
        selection_header = ['Dataset', 'Architecture', 'Mismatch', 'Level', 'Rank 1', 'Rank 2', 'Rank 3']
        for mismatch_type in args.mismatch_types:
            for dataset, architecture in data.pairs:
                for level in GRIDS[mismatch_type]:
                    top = data.top(mismatch_type, dataset=dataset, architecture=architecture, level=level)
                    labels = [f'{data.label(variant)} ({score:.2f})' for score, variant in top]
                    selection_rows.append([dataset, architecture, mismatch_type, f'{level:g}',
                                           *labels, *['N/A'] * (3 - len(labels))])
    else:
        reference_name = 'Best fixed top-three recal-BN WRN'
        reference_note = ('The recalibrated-BN reference is the best member, at each reported point, '
                          'of the fixed top-three set selected ' +
                          ('separately for that dataset and mismatch type by severe-range mean accuracy.'
                           if data.ranking_mode == 'dataset-severe-mean' else
                           'globally for each mismatch type by mean endpoint accuracy across all selected pairs.'))
        selection_header = ['Dataset', 'Mismatch', 'Rank 1', 'Rank 2', 'Rank 3']
        for mismatch_type in args.mismatch_types:
            datasets = args.datasets if data.ranking_mode == 'dataset-severe-mean' else ['All selected']
            for dataset in datasets:
                top = data.top(mismatch_type, dataset=dataset)
                labels = [f'{data.label(variant)} ({score:.2f})' for score, variant in top]
                selection_rows.append([dataset, mismatch_type, *labels, *['N/A'] * (3 - len(labels))])
    lines = intro(data, args, 'PCN versus BN-free WRN controls')
    lines += [
        '## Scope and answer\n',
        'This report includes every BN-free WRN evaluation variant that is complete across all selected '
        'dataset/architecture pairs and requested mismatch grids. It does not substitute incomplete pilots. '
        + reference_note + '\n',
        'Removing BN does not by itself reproduce PCN robustness. Frozen/recalibrated BN and matched '
        'BN/bias controls are detailed in [the WRN comparison](wrn_bn_bias_and_top3_comparison.md); '
        'model IDs and recipes are in [the registry](wrn_variants_and_plots.md).\n',
        ('## Pointwise WRN selections\n' if data.ranking_mode == 'per-point' else '## Fixed WRN selections\n'),
        table(selection_header, selection_rows),
        '## Aggregate nonzero results\n',
    ]
    summary = []
    for mismatch_type in args.mismatch_types:
        for variant in ['P0', *bnfree]:
            model_values, reference_values = [], []
            for dataset, architecture in data.pairs:
                selected = [i for _, i in data.selected_top(dataset, architecture, mismatch_type)]
                for level, value in data.curve(variant, dataset, architecture, mismatch_type):
                    if level == 0:
                        continue
                    candidates = [data.values.get((dataset, architecture, mismatch_type, level, i))
                                  for i in selected]
                    if candidates and all(candidate is not None for candidate in candidates):
                        model_values.append(value[0])
                        reference_values.append(max(candidate[0] for candidate in candidates))
            if model_values:
                gaps = [model - reference for model, reference in zip(model_values, reference_values)]
                summary.append([mismatch_type, data.label(variant), len(model_values),
                                f'{statistics.mean(model_values):.2f}',
                                f'{statistics.mean(reference_values):.2f}',
                                f'{statistics.mean(gaps):+.2f}',
                                f'{sum(gap > 0 for gap in gaps)}/{len(gaps)}'])
    lines.append(table(['Mismatch', 'Model', 'Points', 'Model mean', reference_name + ' mean',
                        'Model - reference', 'Wins'], summary))
    lines += ['## Level-by-level results\n']
    for mismatch_type in args.mismatch_types:
        for dataset, architecture in data.pairs:
            selected = [i for _, i in data.selected_top(dataset, architecture, mismatch_type)]
            ids = ['P0', *bnfree]
            levels = sorted({level for variant in ids
                             for level, _ in data.curve(variant, dataset, architecture, mismatch_type)})
            rows = []
            for level in levels:
                candidates = [data.values.get((dataset, architecture, mismatch_type, level, i))
                              for i in selected]
                reference = max((candidate[0] for candidate in candidates if candidate is not None), default=None)
                row = [f'{level:g}', f'{reference:.2f}' if reference is not None else 'N/A']
                for variant in ids:
                    value = data.values.get((dataset, architecture, mismatch_type, level, variant))
                    row.append(f'{value[0]:.2f}' if value else 'N/A')
                rows.append(row)
            lines += [f'### {dataset} / {architecture} / {mismatch_type}\n',
                      table(['Level', reference_name] +
                            [data.label(i) for i in ids], rows)]
    (args.output_dir / 'pcn_vs_bnfree_wrn_max_mul.md').write_text('\n'.join(lines).rstrip() + '\n')


def curve_table(data, d, a, t, ids, oracle=False):
    wrns = data.ids(d, a, t, 'wrn')
    levels = sorted({x for i in ids + (wrns if oracle else []) for x, _ in data.curve(i, d, a, t)})
    rows = []
    for l in levels:
        row = [f'{l:g}']
        for i in ids:
            v = data.values.get((d, a, t, l, i))
            row.append(f'{v[0]:.2f}' if v else 'N/A')
        if oracle:
            candidates = [(data.values[d, a, t, l, i][0], i) for i in wrns if (d, a, t, l, i) in data.values]
            best = max(candidates) if candidates else None
            row.append(f'{best[0]:.2f} ({data.label(best[1])})' if best else 'N/A')
        rows.append(row)
    return table(['Level'] + [data.label(i) for i in ids] + (['Best available WRN (variant)'] if oracle else []), rows)


def plot(data, args, folder, ids, d, a, t):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ids = [i for i in ids if data.curve(i, d, a, t)]
    if not ids:
        return None
    top3_view = folder == 'pcn_vs_top3_baseline_wrns' or folder.endswith('_vs_top3_baseline_wrns')
    pcn_plot = folder == 'pcn_vs_all_wrn' or top3_view
    top = {i for _, i in data.selected_top(d, a, t)}
    all_ids = sorted(data.meta)
    pcn_styles = pcn_linestyles([i for i in ids if data.meta[i]['family'] == 'pcn'])
    labels = {i: textwrap.fill(data.label(i), width=62, break_long_words=True) for i in ids}
    nrows = math.ceil(len(ids) / 2)
    legend_height = .35 + sum(.07 + .15 * max(
        labels[ids[j]].count('\n') + 1,
        labels[ids[j+nrows]].count('\n') + 1 if j+nrows < len(ids) else 1)
        for j in range(nrows))
    height = 6 + legend_height
    fig, ax = plt.subplots(figsize=(14, height))
    for i in ids:
        m = data.meta[i]
        idx = all_ids.index(i)
        if m['family'] == 'pcn':
            palette = 'Blues'
        elif not pcn_plot:
            palette = 'Greens' if not m['bn'] else 'Reds' if 'frozen' in m['policy'] else 'Purples'
        elif args.color_mode == 'batchnorm':
            palette = 'Reds' if m['bn'] else 'Greens'
        elif args.color_mode == 'pooling':
            palette = 'Reds' if m['pooling'] != 'none' else 'Greens'
        else:
            palette = 'Reds'
        shade = .45 + .5 * idx / max(1, len(all_ids) - 1)
        # Base identity fixes markers across frozen/recalibrated branches.
        style = int(hashlib.sha256(m['base'].encode()).hexdigest()[:8], 16)
        marker = ['o', 's', '^', 'D', 'v', 'P', 'X', '<', '>'][style % 9]
        linestyle = ['-', '--', '-.', ':'][style % 4]
        if m['family'] == 'pcn':
            linestyle = pcn_styles[i]
        if pcn_plot and args.highlight_top3 and m['family'] == 'wrn':
            linestyle = '-' if i in top else '--' if style % 2 else ':'
        elif folder == 'wrn_bn_bias' and not m['bn']:
            linestyle = '--' if m['bias'] else ':'
        c = data.curve(i, d, a, t)
        x, y = [l for l, _ in c], [v[0] for _, v in c]
        color = plt.get_cmap(palette)(shade)
        ax.plot(x, y, label=labels[i], color=color, marker=marker, markersize=4,
                linewidth=1.5, linestyle=linestyle)
        if args.uncertainty:
            # Do not fabricate uncertainty for historical means-only data.
            z = [(l, v) for l, v in c if v[1] is not None]
            if z:
                ax.fill_between([l for l, _ in z], [max(0, v[0]-v[1]) for _, v in z],
                                [min(100, v[0]+v[1]) for _, v in z], color=color, alpha=.1)
    ax.set(xlabel='Mismatch level', ylabel='Test accuracy (%)', ylim=(0, 100),
           title=f'{d} / {a} / {t}')
    ax.grid(alpha=.2)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=8, frameon=False,
               columnspacing=2, handlelength=3)
    fig.tight_layout(rect=(0, legend_height / height, 1, 1))
    path = args.output_dir / 'plots' / folder / f'{d}_{a}_{t}.pdf'
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return str(path.relative_to(args.output_dir))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', required=True, choices=['pcn-md', 'pcn-plots', 'wrn-md', 'wrn-plots'])
    p.add_argument('--output-dir', type=Path, default=EVIDENCE)
    p.add_argument('--wide-csv', type=Path, default=EVIDENCE / 'all_pcn_wrn_max_mul.csv')
    p.add_argument('--sizes', default=','.join(SIZES), help='All modes: depth_width list; any positive integer width')
    p.add_argument('--datasets', default=','.join(DATASETS), help='All modes: cifar10,cifar100 or one dataset')
    p.add_argument('--controls-source', type=Path, action='append', default=[])
    p.add_argument('--baseline-paired-root', type=Path, action='append', default=[])
    p.add_argument('--pcn-bn-root', type=Path, action='append', default=[])
    p.add_argument('--extra-data', type=Path, action='append', default=[])
    p.add_argument('--extra-registry', type=Path)
    p.add_argument('--mismatch-types', type=lambda s: s.split(','), default=list(TYPES[:2]))
    p.add_argument('--wrn-ranking', choices=RANKING_MODES, default='dataset-severe-mean',
                   help='All modes: dataset severe-range fixed top three (default), legacy global endpoint, or pointwise')
    p.add_argument('--exclude-avgpool-wrn', action='store_true')
    p.add_argument('--color-mode', choices=['family', 'batchnorm', 'pooling'],
                   help='pcn-plots only: WRN color grouping; PCN curves stay blue')
    p.add_argument('--highlight-top3', action='store_true',
                   help='pcn-plots only: top-three WRNs solid, other WRNs dashed/dotted')
    p.add_argument('--top3-plus-baseline', action='store_true',
                   help='pcn-plots only: selected PCNs + top-three unfused recal-BN WRNs + W0-R; separate plot folder')
    p.add_argument('--pcn-models', help='pcn-plots only: comma-separated IDs, default P0,P3; or all')
    p.add_argument('--uncertainty', action='store_true',
                   help='pcn-plots/wrn-plots only: mean +/- STD where loaded; OFF by default')
    args = p.parse_args()
    args.sizes, args.datasets = args.sizes.split(','), args.datasets.split(',')
    try:
        for size in args.sizes:
            parse_size(size)
        if len(set(args.sizes)) != len(args.sizes) or len(set(args.datasets)) != len(args.datasets) or not set(args.datasets) <= set(DATASETS):
            raise ValueError('Invalid or duplicate dataset/size selection')
    except ValueError as error:
        p.error(str(error))
    if not set(args.mismatch_types) <= set(TYPES):
        p.error('Unknown mismatch type')
    if args.mode != 'pcn-plots' and (args.color_mode is not None or args.highlight_top3 or args.pcn_models is not None or args.top3_plus_baseline):
        p.error('--color-mode, --highlight-top3, --pcn-models and --top3-plus-baseline apply only to pcn-plots')
    if args.uncertainty and not args.mode.endswith('-plots'):
        p.error('--uncertainty applies only to plot modes')
    args.color_mode = args.color_mode or 'family'
    args.pcn_models = args.pcn_models or 'P0,P3'
    args.controls_source = list(dict.fromkeys([
        ROOT / 'logs/wrn_controls_analysis_minimal.tar.gz', ROOT / 'logs/wrn_pool_controls_slurm',
        *args.controls_source]))
    if not args.baseline_paired_root:
        args.baseline_paired_root = [ROOT / 'logs/wrn_baseline_unfused_missing_trials']
    if not args.pcn_bn_root:
        args.pcn_bn_root = [ROOT / 'logs/pcn_boundary_bn_mismatch_slurm']
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = load(args)
    data.ranking_mode = args.wrn_ranking
    data.require_complete(args.mismatch_types)
    selected_pcn_ids = []
    if args.mode == 'pcn-plots':
        selected_pcn_ids = (sorted(i for i, meta in data.meta.items()
                                   if meta['family'] == 'pcn' and any(k[4] == i for k in data.values))
                            if args.pcn_models == 'all' else args.pcn_models.split(','))
        if (not selected_pcn_ids or len(set(selected_pcn_ids)) != len(selected_pcn_ids)
                or any(i not in data.meta or data.meta[i]['family'] != 'pcn'
                       for i in selected_pcn_ids)):
            p.error('--pcn-models must name registered PCNs (P0,P1,P2,P3, or loaded PB-F/PB-R)')
    top3_folder = top3_plot_folder(selected_pcn_ids) if args.top3_plus_baseline else None
    with (args.output_dir / 'robustness_report_values.csv').open('w', newline='') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(['dataset', 'architecture', 'mismatch_type', 'level', 'variant', 'accuracy', 'std', 'source'])
        for k, v in sorted(data.values.items()):
            if data.eligible(k[4]):
                writer.writerow([*k, v[0], '' if v[1] is None else v[1], v[2]])
    plots = []
    if args.mode.endswith('-md'):
        write_md(data, args, wrn=args.mode == 'wrn-md')
    else:
        for t in args.mismatch_types:
            for d, a in data.pairs:
                pcn_ids = [i for i in data.ids(d, a, t) if data.meta[i]['family'] != 'pcn'
                           or args.pcn_models == 'all' or i in args.pcn_models.split(',')]
                groups = [('pcn_vs_all_wrn', pcn_ids)] if args.mode == 'pcn-plots' else [
                    ('wrn_bn_bias', data.matched(d, a, t)),
                    ('wrn_frozen_recal_bn', data.paired_ids(d, a, t))]
                if args.top3_plus_baseline:
                    groups = [(top3_folder, data.top3_baseline_ids(d, a, t, selected_pcn_ids))]
                for folder, ids in groups:
                    path = plot(data, args, folder, ids, d, a, t)
                    if path:
                        plots.append(path)
    manifest = args.output_dir / 'robustness_report_manifest.json'
    previous = json.loads(manifest.read_text()) if manifest.exists() else {}
    entry = ('pcn-plots-top3-baseline' if top3_folder == 'pcn_vs_top3_baseline_wrns'
             else f'pcn-plots-top3-baseline:{top3_folder}' if args.top3_plus_baseline
             else args.mode)
    previous[entry] = dict(plots=plots, options={k: str(v) for k, v in vars(args).items()},
                          sources=sorted(data.sources), wrn_top3={
                              f'{d}/{t}': data.top(t, dataset=d)
                              for t in args.mismatch_types for d in args.datasets
                              if args.wrn_ranking == 'dataset-severe-mean'},
                          global_endpoint_top3={
                              t: data.top(t) for t in args.mismatch_types
                              if args.wrn_ranking == 'global-endpoint'})
    manifest.write_text(json.dumps(previous, indent=2) + '\n')
    registry = ['# WRN/PCN variants and plots\n',
                'Reproduce with `bash launch_scripts/generate_robustness_reports.sh`. '
                'Use `--output-dir` for alternate filtering/styling campaigns. '
                'Raw source files are never modified. Optional bands are population standard deviations where available.\n',
                'BN/bias columns refer to convolution bias; classifier bias is retained. '
                'W0-FF/W1-FF are folded frozen; F suffix otherwise means unfused frozen. '
                'Historical recipe details: [handoff](paper_handoff_max_mul.md). '
                'Per-value source mapping: [CSV](robustness_report_values.csv).\n']
    active = sorted({k[4] for k in data.values if data.eligible(k[4])})
    registry.append('Plot legends/table headings use descriptive labels below; IDs remain stable source keys. '
                    'PCN plots default to PCNetNoBatchNorm / ODEXInitFFFB and PCNetNoBatchNorm / ODEBlockXInit; '
                    'Markdown retains every available PCN. fd means final dropout, ptinit means PyTorch-default '
                    'initialization (otherwise WRN initialization), SC means shortcut. '
                    f'Uncertainty is OFF by default. Only complete WRN variants across the {len(data.pairs)} selected pairs are included.\n')
    registry.append(table(['Source ID', 'Plot/table label', 'BN', 'Conv bias', 'Pool', 'Evaluation', 'Training recipe'],
                          [(i, data.label(i), data.meta[i]['bn'], data.meta[i]['bias'],
                            data.meta[i]['pooling'], data.meta[i]['policy'], data.meta[i]['recipe']) for i in active]))
    registry.append('## PDFs\n')
    for key, plot_entry in sorted(previous.items()):
        if not key.startswith('pcn-plots-top3-baseline'):
            continue
        folder = ('pcn_vs_top3_baseline_wrns' if ':' not in key else key.split(':', 1)[1])
        models = plot_entry.get('options', {}).get('pcn_models', 'P0,P3')
        ranking = plot_entry.get('options', {}).get('wrn_ranking', 'global-endpoint')
        registry.append(f'`plots/{folder}/`: {models} plus W0-R and selected complete '
                        f'unfused recalibrated-BN WRNs. Ranking mode: `{ranking}`. '
                        'Selection is recomputed from loaded data on every invocation.\n')
    for mode, entry in sorted(previous.items()):
        for path in entry.get('plots', []):
            registry.append(f'- [{path}]({path})\n')
    registry.append('## Source files\n')
    registry.extend(f'- `{s}`\n' for s in sorted(data.sources))
    (args.output_dir / 'wrn_variants_and_plots.md').write_text('\n'.join(registry))
    print(f'{args.mode}: {len(data.values)} values; {len(plots)} PDFs; output={args.output_dir}')


if __name__ == '__main__':
    main()
