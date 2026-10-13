#!/usr/bin/env python3
"""Show completion and the latest progress record without changing a run."""
import argparse
import json
from pathlib import Path
from datetime import datetime, timezone


def latest_record(path):
    if not path.is_file():
        return None
    with path.open('rb') as f:
        f.seek(max(0, path.stat().st_size - 32000))
        lines = f.read().decode(errors='replace').splitlines()
    for line in reversed(lines):
        try:
            row = json.loads(line)
        except (ValueError, TypeError):
            continue
        if isinstance(row, dict) and ('iteration' in row or 'step' in row):
            return row
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_root', type=Path)
    args = parser.parse_args()
    run = args.run_root
    status = {'checked_utc': datetime.now(timezone.utc).isoformat(), 'run_root': str(run),
              'generation_complete': (run / 'generators/COMPLETE').is_file(),
              'analysis_complete': (run / 'analysis/ANALYSIS_COMPLETE').is_file(),
              'report_complete': (run / 'analysis/COMPLETE').is_file(), 'replicas': {}}
    for i in (1, 2):
        replica = run / 'generators' / f'replicate_{i}'
        record = latest_record(replica / 'logs/mbgan_train.log')
        status['replicas'][str(i)] = {
            'completed': (replica / 'completion.json').is_file(),
            'icfm_sample_exists': (replica / 'sample_icfm.csv').is_file(),
            'mbgan_iteration': record.get('iteration') if record else None,
            'mbgan_remaining_hours': round(record.get('estimated_remaining_seconds', 0) / 3600, 2) if record else None}
    print(json.dumps(status, indent=2))


if __name__ == '__main__':
    main()
