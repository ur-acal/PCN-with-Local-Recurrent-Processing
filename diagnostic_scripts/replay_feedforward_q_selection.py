"""Replay the production q selector from a recorded FT command, without FT."""
import argparse
import json
import os
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-log', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--candidates', nargs='+', type=float, required=True)
    options = parser.parse_args()
    os.chdir(ROOT)
    commands = []
    with options.source_log.open() as stream:
        for line in stream:
            if not line.startswith('Running: '):
                continue
            command = shlex.split(line.removeprefix('Running: '))
            if ('baseline/train_baseline_cifar.py' in command
                    and '--resume_checkpoint' in command):
                commands.append(command)
    if not commands:
        parser.error('No recorded feedforward FT command found')
    command = commands[-1]
    argv = command[command.index('baseline/train_baseline_cifar.py') + 1:]
    from diagnostic_scripts.select_feedforward_one_over_q import (
        args_from_ft_command, select)
    args = args_from_ft_command(argv)
    if not Path(args.checkpoint).is_file():
        parser.error(f'Source checkpoint is missing: {args.checkpoint}')
    output = options.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    args.result_path = str(output / 'sanity')
    # Normal selection is unexpanded; retain a separate cache location anyway.
    args.expanded_weight_dir = str(output / 'expanded_weights')
    (output / 'source_command.json').write_text(json.dumps(dict(
        log=str(options.source_log.resolve()), command=command), indent=2) + '\n')
    selected = select(args, options.candidates, output)
    print(f'REPLAY_SELECTED one_over_q={selected:g}', flush=True)


if __name__ == '__main__':
    main()
