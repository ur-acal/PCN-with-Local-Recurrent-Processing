"""Generate one readable Claim 1/2 report from saved diagnostics; no inference."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT.parent / 'papers/hardware-native-neural-ode/shared/evidence/generic-mismatch'
LEVELS = {'max_additive': (.08, .09, .10), 'multiplicative': (.30, .35, .40)}
SIZES = ('16_2', '16_4', '28_2', '28_4')
FAMILIES = {'pcn': 'PCN_noX', 'wrn': 'WRN_original', 'wd1e3': 'WRN_wd1e-3',
            'finaldrop': 'std_BN_wd1e-3_lr1e-1_fd0p25'}
FAMILY_LABELS = {**FAMILIES,
            'r4': 'avgSC_BN_wd1e-3_lr1e-1_fd0p25',
            'r7': 'avgMain_BN_wd1e-3_lr1e-1_fd0p25',
            'r10': 'std_BN_wd1e-3_lr1e-2_fd0p25_ptinit',
            'r14': 'maxMain_BN_wd1e-3_lr1e-1_fd0p25',
            'r17': 'avgMain_BN_wd1e-3_lr1e-2_fd0p25_ptinit',
            'r18': 'maxMain_BN_wd1e-3_lr1e-2_fd0p25_ptinit'}
LEGACY_FAMILIES = ('wrn', 'wd1e3', 'finaldrop')
SELECTED_TOP3 = {
    ('cifar10', 'max_additive'): ('r10', 'r17', 'finaldrop'),
    ('cifar100', 'max_additive'): ('r10', 'r7', 'r18'),
    ('cifar10', 'multiplicative'): ('finaldrop', 'r14', 'r7'),
    ('cifar100', 'multiplicative'): ('r7', 'r14', 'r4'),
}


def stats(values):
    if any(v is None for v in values):
        return None
    if not values or any(not math.isfinite(v) for v in values):
        raise ValueError('Empty or nonfinite metric')
    return statistics.mean(values), statistics.pstdev(values)


def fmt(value, accuracy=False):
    if value is None:
        return 'N/A'
    precision = 2 if accuracy else 4
    return f'{value[0]:.{precision}f} +/- {value[1]:.{precision}f}'


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '|' + '|'.join(['---'] * len(headers)) + '|'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows]) + '\n'


def stage_means(metrics, units, claim, policy, trials):
    if len(units) not in (7, 13) or len(set(units)) != len(units):
        raise ValueError('Expected stem plus three stages of two or four units')
    n = (len(units) - 1) // 3
    result = {}
    for stage in range(3):
        group = units[1 + stage * n:1 + (stage + 1) * n]
        means = []
        for trial in range(trials):
            errors = [metrics[claim, policy, trial, unit] for unit in group]
            means.append(statistics.mean(errors) if all(v is not None for v in errors) else None)
        result[f'stage{stage + 1}_c{claim}'] = stats(means)
    return result


def read_run(directory, condition, level, trials=10, logit_directory=None, pcn_c2_directory=None):
    state = json.loads((directory / 'state.json').read_text())
    if state.get('status') != 'complete' or state.get('smoke') is not False:
        raise ValueError(f'Incomplete or smoke run: {directory}')
    model = json.loads((directory / 'model.json').read_text())
    if logit_directory is not None:
        complete = json.loads((logit_directory / 'complete.json').read_text())
        provenance = json.loads((logit_directory / 'provenance.json').read_text())
        if (complete['status'] != 'complete' or complete['trials'] != trials or
                provenance['checkpoint_sha256'] != model['checkpoint_sha256'] or
                Path(provenance['original']).resolve() != directory.resolve() or
                provenance['condition'] != condition or provenance['level'] != level):
            raise ValueError(f'Supplemental provenance mismatch: {logit_directory}')
    units = model['units']
    policies = ('no_bn',) if model['spec']['family'] == 'pcn' else ('frozen', 'recalibrated')
    with (directory / 'per_trial.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    if pcn_c2_directory is not None:
        replacement = json.loads((pcn_c2_directory / 'model.json').read_text())
        state = json.loads((pcn_c2_directory / 'state.json').read_text())
        provenance = json.loads((pcn_c2_directory.parent / 'provenance.json').read_text())
        if (model['spec']['family'] != 'pcn' or replacement != model or
                state != dict(status='complete', smoke=False) or
                not provenance['pcn_full'] or not provenance['deterministic_algorithms'] or
                Path(provenance['original']).resolve() != directory.resolve()):
            raise ValueError('Invalid deterministic PCN C2 replacement')
        with (pcn_c2_directory / 'per_trial.csv').open() as stream:
            replacement_rows = list(csv.DictReader(stream))
        if any(int(r['claim']) != 2 for r in replacement_rows):
            raise ValueError('PCN replacement must contain C2 only')
        rows = [r for r in rows if int(r['claim']) == 1] + replacement_rows
    metrics = {}
    zero_rows = 0
    for row in rows:
        claim, trial = int(row['claim']), int(row['trial'])
        if row['condition'] != condition or float(row['level']) != level or int(row['samples']) != 10000:
            raise ValueError(f'Unexpected condition/level/sample count: {directory}')
        expected_target = row['layer'] if claim == 1 else 'network'
        if row['target'] != expected_target:
            raise ValueError(f'Unexpected perturbation scope: {directory}')
        key = (claim, row['bn_policy'], trial, row['layer'])
        if key in metrics:
            raise ValueError(f'Duplicate metric {key}: {directory}')
        raw = row['dataset_relative_l2']
        metrics[key] = float(raw) if raw else None
        if metrics[key] is not None and (not math.isfinite(metrics[key]) or metrics[key] < 0):
            raise ValueError(f'Invalid metric {key}: {directory}')
        zero_rows += int(row['zero_reference_images']) > 0
    expected = {(claim, policy, trial, unit) for claim in (1, 2) for policy in policies
                for trial in range(trials) for unit in (units if claim == 1 else units + ['final_feature', 'logits'])}
    if set(metrics) != expected:
        raise ValueError(f'Incomplete/unexpected trial coverage: {directory}')
    result = {}
    for policy in policies:
        values = {}
        for claim in (1, 2):
            means = []
            for trial in range(trials):
                errors = [metrics[claim, policy, trial, unit] for unit in units]
                means.append(statistics.mean(errors) if all(v is not None for v in errors) else None)
            values[f'mean{claim}'] = stats(means)
            values[f'first{claim}'] = stats([metrics[claim, policy, t, units[0]] for t in range(trials)])
            last = units[-1] if claim == 1 else 'final_feature'
            values[f'last{claim}'] = stats([metrics[claim, policy, t, last] for t in range(trials)])
            values.update(stage_means(metrics, units, claim, policy, trials))
        values['logits'] = stats([metrics[2, policy, t, 'logits'] for t in range(trials)])
        accuracies = []
        extras = {k: [] for k in ('centered', 'margin', 'correct_to_wrong', 'wrong_to_correct')}
        for trial in range(trials):
            summary = (pcn_c2_directory or directory) / f'claim2/network/{condition}/level_{level:g}/trial_{trial}/{policy}/summary.json'
            transition = json.loads(summary.read_text())
            acc = transition['noisy_accuracy']
            if transition['samples'] != 10000 or not 0 <= acc <= 100 or not math.isclose(
                    acc, 100 * transition['noisy_correct'] / transition['samples']):
                raise ValueError(f'Invalid classification result: {summary}')
            accuracies.append(acc)
            diagnostics = (json.loads((logit_directory / f'trial_{trial}/{policy}/summary.json').read_text())
                           if logit_directory is not None else transition)
            if logit_directory is not None:
                if not diagnostics['replay_validated'] or any(diagnostics[k] != transition[k] for k in
                        ('samples', 'clean_correct', 'noisy_correct', 'correct_to_wrong', 'wrong_to_correct')):
                    raise ValueError('Supplemental classification differs from original')
                if not math.isclose(diagnostics['original_logits']['dataset_relative_l2'],
                                    metrics[2, policy, trial, 'logits'], rel_tol=1e-5, abs_tol=1e-8):
                    raise ValueError('Supplemental original error differs')
            if 'centered_logits' in diagnostics:
                extras['centered'].append(diagnostics['centered_logits']['dataset_relative_l2'])
                extras['margin'].append(diagnostics['margin_degradation'])
            elif logit_directory is not None:
                raise ValueError('Missing supplemental logit metrics')
            for key in ('correct_to_wrong', 'wrong_to_correct'):
                extras[key].append(100 * transition[key] / transition['samples'])
        values['accuracy'] = stats(accuracies)
        values.update({k: stats(v) if len(v) == trials else None for k, v in extras.items()})
        result[policy] = values
    return model['spec'], result, zero_rows


def anomalies(pcn, wrn):
    """Overlapping descriptive comparisons of trial means, not significance tests."""
    def compare(key, sign):
        return pcn[key] is not None and wrn[key] is not None and sign(pcn[key][0], wrn[key][0])
    lower_feature = compare('last2', lambda a, b: a < b)
    lower_logits = compare('logits', lambda a, b: a < b)
    higher_logits = compare('logits', lambda a, b: a > b)
    lower_acc = compare('accuracy', lambda a, b: a < b)
    return [str(n) for n, yes in ((1, lower_feature and higher_logits),
                                  (2, lower_logits and lower_acc), (3, lower_feature and lower_acc)) if yes]


def summarization_tables(data, family_selector=None):
    pairs = [(d, s) for d in ('cifar10', 'cifar100') for s in SIZES]

    def wrns(dataset, condition):
        return (family_selector[dataset, condition] if family_selector is not None
                else LEGACY_FAMILIES)

    def comparisons(condition, levels):
        for dataset, size in pairs:
            for level in levels:
                prefix = dataset, size, condition, level
                yield (dataset, size, data[*prefix, 'pcn', 'no_bn'],
                       [data[*prefix, f, 'recalibrated'] for f in wrns(dataset, condition)])

    def lower(pcn, others, metric):
        values = [pcn[metric]] + [w[metric] for w in others]
        if any(v is None for v in values):
            return None
        return values[0][0] < min(v[0] for v in values[1:])

    def count(outcomes):
        wins, undefined = sum(v is True for v in outcomes), sum(v is None for v in outcomes)
        result = f'{wins}/{len(outcomes)}'
        if undefined:
            return result + f' ({undefined} undefined)'
        return f'**{result}**' if wins == len(outcomes) else result

    lines = ['## Summarization tables for Main comparisons', '',
             'Counts use unrounded dataset-relative error means and recalibrated-BN WRNs only. '
             'Each stage mean equally averages its units per trial, then averages over trials. '
             'Widths in the first column are for factors x2 / x4. Stage means exclude the stem.', '',
             'All-level denominators: 8 dataset/size pairs x 3 levels x 3 WRNs = 72 comparisons '
             '(24 distinct PCN cases). Endpoint denominators: 8 pairs, each compared against the '
             'lowest-error WRN, selected separately for that metric and pair. '
             'Ties are not wins; undefined metrics are explicitly counted.', '']
    names = ['Max-additive', 'Multiplicative']
    for claim in (1, 2):
        groups = [('Overall, including first layer', f'mean{claim}'),
                  ('First layer - 16 / 16', f'first{claim}'),
                  ('First stage - 32 / 64', f'stage1_c{claim}'),
                  ('Second stage - 64 / 128', f'stage2_c{claim}'),
                  ('Third stage - 128 / 256', f'stage3_c{claim}')]
        lines += [f'### PCN with lower C{claim} mean error than WRN', '',
                  'All three evaluated mismatch levels, compared separately against each WRN:', '']
        for endpoint in (False, True):
            rows = []
            for label, metric in groups:
                cells = [label]
                for condition, levels in LEVELS.items():
                    selected = (max(levels),) if endpoint else levels
                    outcomes = []
                    for _, _, pcn, others in comparisons(condition, selected):
                        outcomes.extend([lower(pcn, others, metric)] if endpoint else
                                        [lower(pcn, [w], metric) for w in others])
                    cells.append(count(outcomes))
                rows.append(cells)
            if endpoint:
                lines += ['Most severe mismatch level, compared against the lowest-error WRN:', '']
            headers = [f'C{claim} layer group - channels (x2 / x4)'] + [
                name + (f' **{max(levels):.2f}**' if endpoint else '')
                for name, levels in zip(names, LEVELS.values())]
            lines += [table(headers, rows), '']
    lines += ['### PCN accuracy wins in Claim 2', '',
              'Accuracy is evaluated only in C2. The comparator is the highest-accuracy WRN '
              'among the three at each dataset/size/level, not necessarily the lowest-error WRN.', '',
              'All three evaluated mismatch levels:', '']
    accuracy_rows, endpoint_rows = [], []
    all_wins, all_total, endpoint_wins, endpoint_total = 0, 0, 0, 0
    for name, (condition, levels) in zip(names, LEVELS.items()):
        for level in levels:
            wins, losing = 0, []
            for dataset, size, pcn, others in comparisons(condition, (level,)):
                delta = pcn['accuracy'][0] - max(w['accuracy'][0] for w in others)
                wins += delta > 0
                if delta <= 0:
                    short = 'C10' if dataset == 'cifar10' else 'C100'
                    losing.append(f'{short} {size.replace("_", "-")}' + (' (tie)' if delta == 0 else ''))
            all_wins += wins
            all_total += len(pairs)
            accuracy_rows.append([name, f'{level:.2f}', f'**{wins}/{len(pairs)}**', ', '.join(losing) or 'None'])
            if level == max(levels):
                endpoint_wins += wins
                endpoint_total += len(pairs)
                endpoint_rows.append([f'{name} **{level:.2f}**', f'**{wins}/{len(pairs)}**'])
    accuracy_rows.append(['**Total**', '', f'**{all_wins}/{all_total}**', ''])
    endpoint_rows.append(['**Total**', f'**{endpoint_wins}/{endpoint_total}**'])
    lines += [table(['Mismatch type', 'Level', 'PCN accuracy wins', 'Losing cases / ties'], accuracy_rows), '',
              'Most severe mismatch levels:', '',
              table(['Most severe mismatch', 'PCN accuracy wins'], endpoint_rows), '']
    return lines


def select_top_abnormal(data, family_selector=None):
    """Best-accuracy WRN; error-gap top six, then accuracy-gap top three."""
    candidates = []
    for dataset in ('cifar10', 'cifar100'):
        for size in SIZES:
            for condition, levels in LEVELS.items():
                for level in levels:
                    prefix = dataset, size, condition, level
                    pcn = data[*prefix, 'pcn', 'no_bn']
                    families = (family_selector[dataset, condition] if family_selector is not None
                                else LEGACY_FAMILIES)
                    family = max(families,
                                 key=lambda f: data[*prefix, f, 'recalibrated']['accuracy'][0])
                    wrn = data[*prefix, family, 'recalibrated']
                    accuracy_gap = pcn['accuracy'][0] - wrn['accuracy'][0]
                    if pcn['logits'] is None or wrn['logits'] is None:
                        continue
                    error_gap = pcn['logits'][0] - wrn['logits'][0]
                    if accuracy_gap > 1 and error_gap > 0:
                        candidates.append(dict(prefix=prefix, family=family, pcn=pcn, wrn=wrn,
                                               accuracy_gap=accuracy_gap, error_gap=error_gap))
    shortlist = sorted(candidates, key=lambda r: -r['error_gap'])[:6]
    selected = sorted(shortlist, key=lambda r: -r['accuracy_gap'])[:3]
    return shortlist, selected


def top_abnormal_tables(data, family_selector=None):
    shortlist, selected = select_top_abnormal(data, family_selector)
    lines = ['## Selected top three high-logit-error accuracy wins', '',
             'For each dataset/size/type/level, select the highest-accuracy recalibrated WRN among the three. '
             'Require PCN accuracy advantage >1 percentage point and PCN original logit error > WRN error. '
             'Keep the six largest PCN-minus-WRN original logit-error differences; among those, select the '
             'three largest accuracy advantages. Adjacent levels are not deduplicated. Stable inventory order breaks ties. '
             'Selection uses original errors, not centered errors. All metrics below use the same comparator.', '',
             table(['Error-gap rank', 'Dataset / size / mismatch / level', 'WRN', 'Original error gap', 'Accuracy gap (pp)'],
                     [[i + 1, ' / '.join(map(str, r['prefix'])), FAMILY_LABELS[r['family']],
                     f"{r['error_gap']:.4f}", f"{r['accuracy_gap']:.2f}"] for i, r in enumerate(shortlist)]), '']
    for i, r in enumerate(selected, 1):
        lines += [f"### Selected case {i}: " + ' / '.join(map(str, r['prefix'])), '',
                  table(['Model', 'Final feature error', 'Original logit error', 'Centered logit error',
                         'Margin degradation', 'Correct->wrong (%)', 'Wrong->correct (%)', 'Accuracy (%)'],
                        [[label] + [fmt(v[k], k in ('accuracy', 'correct_to_wrong', 'wrong_to_correct'))
                                    for k in ('last2', 'logits', 'centered', 'margin', 'correct_to_wrong',
                                              'wrong_to_correct', 'accuracy')]
                         for label, v in [('PCN_noX', r['pcn']), (FAMILY_LABELS[r['family']] + ' (BN recal)', r['wrn'])]]), '']
        p, w = r['pcn']['centered'], r['wrn']['centered']
        lines += [('The higher-error ordering persists after centering; common offsets alone do not explain it.'
                   if p is not None and w is not None and p[0] > w[0] else
                   'PCN no longer has higher centered error; removing common offsets changes the error ordering.'
                   if p is not None and w is not None else 'Centered ordering is undefined due to zero reference energy.'), '',
                  'Centering does not remove positive scaling. Margin degradation remains scale-dependent; '
                  'prediction-transition rates describe the actual accuracy changes, not a causal proof.', '']
    return lines


def generate(source, output, logit_source=None, pcn_c2_source=None, selected_source=None):
    data, references, zero_rows = {}, [], 0
    family_selector = SELECTED_TOP3 if selected_source is not None else None
    # Explicit inventory prevents silently reporting only the runs found on disk.
    for dataset in ('cifar10', 'cifar100'):
        for size in SIZES:
            for condition, levels in LEVELS.items():
                for level in levels:
                    families = (('pcn',) + family_selector[dataset, condition]
                                if family_selector is not None else tuple(FAMILIES))
                    for family in families:
                        name = f'{dataset}_{size}_{family}'
                        run_source = (selected_source
                                      if selected_source is not None and family not in ('pcn', 'finaldrop')
                                      else source)
                        directory = run_source / 'runs' / f'{name}_{condition}_{level:g}' / name
                        supplement = (logit_source / 'runs' / directory.parent.name
                                      if logit_source is not None and family in FAMILIES else None)
                        replacement = None
                        if family == 'pcn' and pcn_c2_source is not None:
                            replacement = pcn_c2_source / 'runs' / directory.parent.name / name
                            supplement = None
                        spec, values, zeros = read_run(directory, condition, level, logit_directory=supplement,
                                                      pcn_c2_directory=replacement)
                        if (spec['dataset'], spec['size'], spec['family']) != (dataset, size, family):
                            raise ValueError(f'Wrong checkpoint identity: {directory}')
                        for policy, metrics in values.items():
                            data[dataset, size, condition, level, family, policy] = metrics
                        zero_rows += zeros
                        relative = os.path.relpath(directory, output.parent)
                        references.append(f'- {dataset} / {size} / {condition} {level:g} / {FAMILY_LABELS[family]}: '
                                          f'[metrics]({relative}/per_trial.csv), [checkpoint/protocol]({relative}/model.json), '
                                          f'[trial accuracy files]({relative}/claim2/network/{condition}/level_{level:g}/).')
                        if supplement is not None:
                            references[-1] += f' [Validated logit diagnostics]({os.path.relpath(supplement, output.parent)}/).'
                        if replacement is not None:
                            relative_new = os.path.relpath(replacement, output.parent)
                            references[-1] += (f' [Replacement C2 metrics]({relative_new}/per_trial.csv), '
                                               f'[old/new accuracy audit]({relative_new}/../old_vs_new.json).')
    lines = ['# Claim 1 and Claim 2: PCN versus WRN', '',
             '## Reading the tables', '',
             'Complete coverage: 48 comparison tables, both CIFAR datasets, '
             'four sizes, three levels per mismatch type, 10 trials and 10,000 test images per run.', '',
             'Main tables include PCN_noX (PCNetNoBatchNorm, ODEXInitFFFB) and recalibrated-BN WRNs only. '
             + ('WRNs are the fixed dataset/mismatch-specific top-three selections recorded in the run monitor.'
                if family_selector is not None else
                'WRN_original, WRN_wd1e-3, and std_BN_wd1e-3_lr1e-1_fd0p25 are the verified families.'), '',
             'C1 = isolated unit with clean input; C2 = full-network mismatch. Error = dataset relative L2, '
             '`sqrt(sum_image ||noisy-clean||^2 / sum_image ||clean||^2)`. '
             'Each cell is mean +/- population STD across 10 trials. Mean-layer error first equally averages '
             'the units within each trial (including the stem/first PCN layer), then averages across trials; '
             'final-feature and logit entries are not counted again in that layer mean.', '',
             'First = stem/first PCN layer. Last C1 = last unit output with only that unit perturbed; '
             'last C2 = final feature under full-network perturbation. PCN unit outputs are pre-external-pooling; '
             'WRN block outputs are after shortcut addition. Final WRN features precede final BN; '
             'PCN final features precede head ReLU/GAP. Each model uses its own clean reference '
             '(original BN statistics, including when the noisy WRN is recalibrated).', '',
             'Accuracy (%) comes from the same C2 trials, not from historical mismatch tables. '
             'There is no per-layer or C1 classification accuracy. Recalibration uses 5,120 training images; '
             'BN parameters and convolution biases are not perturbed. C2 includes classifier weight/bias.', '',
             f'Zero-reference images occurred in {zero_rows} per-trial metric rows. Undefined ratios are N/A, '
             'never epsilon-adjusted or silently omitted.', '',
             f'Generator: `{Path(__file__).resolve()}`. Base source: `{source}`. '
             + (f'Selected-WRN source: `{selected_source}`. ' if selected_source is not None else '') +
             'No inference, training, or raw-data changes are performed.', '']
    lines += ['Centered logit error uses the same dataset-relative L2 formula after subtracting each '
              'image\'s across-class mean separately from clean and noisy logits. Zero centered reference norms '
              'are flagged in trial JSON, never epsilon-adjusted. Margin degradation is the full-test-image mean '
              'of clean minus noisy true-class margin, where margin = true-class score minus the highest incorrect-class score. '
              'Positive degradation means worse margins; raw magnitudes are logit-scale-dependent. '
              'Transition percentages use all 10,000 images, not only initially correct/incorrect images. '
              'All four additional metrics are C2-only, with mean +/- population STD over ten trials.', '']
    if pcn_c2_source is not None:
        lines += ['PCN C2 uses a deterministic full-network rerun: all C2 layer errors, logits, margins and '
                  'accuracies come from the same replacement forward passes. Original PCN C1 and cached clean '
                  'references are unchanged; they were produced before deterministic cuDNN was enabled. '
                  'WRN results retain their validated replays. PCN differences from the original C2 runs are '
                  'recorded in the linked old/new audits, not hidden or required to be zero.', '']
    lines += summarization_tables(data, family_selector) + ['## Main comparisons', '']
    abnormal_rows = []
    columns = ('mean1', 'mean2', 'first1', 'first2', 'last1', 'last2', 'logits',
               'centered', 'margin', 'correct_to_wrong', 'wrong_to_correct', 'accuracy')
    for dataset in ('cifar10', 'cifar100'):
        for size in SIZES:
            for condition, levels in LEVELS.items():
                for level in levels:
                    prefix = (dataset, size, condition, level)
                    lines += [f'### {dataset} / WRN-{size.replace("_", "-")} / {condition} / {level:g}', '']
                    rows = []
                    pcn = data[*prefix, 'pcn', 'no_bn']
                    families = (('pcn',) + family_selector[dataset, condition]
                                if family_selector is not None else tuple(FAMILIES))
                    for family in families:
                        label = FAMILY_LABELS[family]
                        policy = 'no_bn' if family == 'pcn' else 'recalibrated'
                        values = data[*prefix, family, policy]
                        rows.append([label + (' (BN recal)' if family != 'pcn' else '')] +
                                    [fmt(values[k], k in ('accuracy', 'correct_to_wrong', 'wrong_to_correct')) for k in columns])
                        if family != 'pcn':
                            for bn in ('frozen', 'recalibrated'):
                                wrn = data[*prefix, family, bn]
                                types = anomalies(pcn, wrn)
                                if types:
                                    def pair(key):
                                        percent = key in ('accuracy', 'correct_to_wrong', 'wrong_to_correct')
                                        return fmt(pcn[key], percent) + ' / ' + fmt(wrn[key], percent)
                                    abnormal_rows.append([dataset, size, condition, f'{level:g}', label, bn,
                                                          ', '.join(types), *[pair(k) for k in
                                                           ('last2', 'logits', 'centered', 'margin', 'correct_to_wrong',
                                                            'wrong_to_correct', 'accuracy')]])
                    lines += [table(['Model', 'Mean C1', 'Mean C2', 'First C1', 'First C2',
                                     'Last C1', 'Last C2', 'Logits C2', 'Centered logits C2', 'Margin degradation C2',
                                     'Correct->wrong (%)', 'Wrong->correct (%)', 'Accuracy C2 (%)'], rows), '']
    if logit_source is not None:
        lines += top_abnormal_tables(data, family_selector)
    lines += ['## Abnormal cases (Claim 2)', '',
              'PCN is compared against each WRN under **both frozen and recalibrated BN**. '
              'All paired values below are **PCN / WRN** (mean +/- population STD). Labels overlap:', '',
              '- Type 1: lower final-feature error, higher logit error.',
              '- Type 2: lower logit error, lower accuracy.',
              '- Type 3: lower final-feature error, lower accuracy.', '',
              'Strict inequalities use unrounded means, without a minimum difference threshold. '
              'These are descriptive cases, not statistical-significance findings; equal displayed values may differ before rounding. '
              'Type 1 need not imply Type 3.', '',
              f'{len(abnormal_rows)} PCN-versus-WRN comparisons match at least one type.', '',
              table(['Dataset', 'Size', 'Mismatch', 'Level', 'WRN', 'BN', 'Types',
                     'Final error PCN / WRN', 'Logit error PCN / WRN', 'Centered error PCN / WRN',
                     'Margin degradation PCN / WRN', 'Correct->wrong (%) PCN / WRN',
                     'Wrong->correct (%) PCN / WRN', 'Accuracy (%) PCN / WRN'], abnormal_rows)
              if abnormal_rows else 'None.', '', '## Sources', '', *references, '']
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.md.tmp')
    temporary.write_text('\n'.join(lines))
    temporary.replace(output)
    print(f'Wrote {output}: 48 main tables, {len(abnormal_rows)} abnormal cases.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=EVIDENCE / 'activation_claims_20260912')
    parser.add_argument('--output', type=Path, default=EVIDENCE / 'claim1_claim2_comparison.md')
    parser.add_argument('--logit-source', type=Path, default=ROOT / 'logs/activation_logit_diagnostics',
                        help='Validated supplemental C2 replays; require complete coverage')
    parser.add_argument('--pcn-c2-source', type=Path,
                        help='Use deterministic replacement PCN C2, retaining original C1 and clean references')
    parser.add_argument('--selected-source', type=Path,
                        help='Selected-WRN run root; enables the fixed dataset/type top-three inventory')
    args = parser.parse_args()
    generate(args.source.resolve(), args.output.resolve(),
             args.logit_source.resolve() if args.logit_source else None,
             args.pcn_c2_source.resolve() if args.pcn_c2_source else None,
             args.selected_source.resolve() if args.selected_source else None)


if __name__ == '__main__':
    main()
