#!/usr/bin/env python3
"""Archive the complete BMA sample comparison, including capped failures."""
import collections
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import tarfile

root = Path(__file__).resolve().parents[2]
base = root / "target/reconstruction-external"
out = root / "benches/results/reconstruction/bma"
out.mkdir(parents=True, exist_ok=True)
raw = (base / "bma-comparison.raw.csv").read_text()
clean = raw[raw.index("case,method,"):]
rows = list(csv.DictReader(io.StringIO(clean)))
assert len(rows) == 162
assert len({(r['case'], r['method'], r['seed']) for r in rows}) == 162
assert collections.Counter(r['status'] for r in rows) == {'ok': 159, 'ProbeLimit': 3}
assert all(r['prime'] == '2401514164751985937' for r in rows)
assert all(r['case'] == 'ibp_xbox2l2m_rank0001' and r['method'] == 'HuMonagan'
           and int(r['probes']) == 20000 for r in rows if r['status'] != 'ok')
(out / "comparison.csv").write_text(clean)
shutil.copyfile(base / "bma-comparison.log", out / "comparison.log")
tables = {'finite_field': rows}
for name in ['polynomial', 'rational']:
    path = base / f"bma-q-{name}.csv"
    qrows = list(csv.DictReader(path.open()))
    assert len(qrows) == 9 and all(r['status'] == 'ok' for r in qrows)
    for row in qrows:
        assert sum(int(p.split(':')[1]) for p in row['probes_by_prime'].split(';')) == int(row['probes'])
    tables[f'q_{name}'] = qrows
    shutil.copyfile(path, out / f"q-{name}.csv")
    logdir = out / f"q-{name}-logs"
    logdir.mkdir(exist_ok=True)
    for path in (base / f"bma-q-{name}-logs").iterdir():
        if path.suffix in ['.log', '.json', '.q-oracle']:
            shutil.copyfile(path, logdir / path.name)

summary = {
    'profile': 'dev, debug=0; timings are not release performance measurements',
    'finite_field_prime': 2401514164751985937,
    'seeds': [1, 2, 3], 'max_probes': 20000, 'max_attempts': 2,
    'soft_timeout_seconds': 120,
    'verification_points': 3, 'polynomial_pilot_limit': 32,
    'degree_bounds': {'polynomial_sparse3': 2000, 'rational_sparse3': 512, 'other': 128},
    'q_options': {'max_degree': 64, 'max_primes': 20, 'prime_policy': 'method native'},
    'tables': {},
}
for table, data in tables.items():
    groups = collections.defaultdict(list)
    for row in data:
        groups[row['case'], row['method']].append(row)
    summary['tables'][table] = [
        {'case': case, 'method': method, 'runs': len(group),
         'statuses': dict(collections.Counter(r['status'] for r in group)),
         'probes_min': min(int(r['probes']) for r in group),
         'probes_median': statistics.median(int(r['probes']) for r in group),
         'probes_max': max(int(r['probes']) for r in group)}
        for (case, method), group in sorted(groups.items())
    ]
(out / "summary.json").write_text(json.dumps(summary, indent=2) + '\n')

with tarfile.open(out / "inputs.tar.gz", 'w:gz') as archive:
    for family in ['box2l', 'diamond3l', 'xbox2l2m', 'tth2l_b16']:
        folder = base / 'ibp-inputs' / family
        for name in [f'ibp_{family}_rank0001', f'ibp_{family}_rank0001.variables',
                     'manifest.json', 'validation.json']:
            path = folder / name
            archive.add(path, arcname=path.relative_to(base))
    for path in sorted((base / 'bma-inputs').iterdir()):
        archive.add(path, arcname=path.relative_to(base))

for name in ['tests', 'build', 'check', 'clippy']:
    source = Path(f'/tmp/reconstruction-bma-validated-{name}.log')
    content = source.read_text()
    if name == 'tests':
        assert '50 passed; 0 failed' in content and '7 passed; 0 failed' in content
    else:
        assert 'Finished ' in content and '\nerror:' not in content
    shutil.copyfile(source, out / f'{name}.log')
shutil.copyfile('/tmp/reconstruction-bma-clippy-unfiltered.log', out / 'clippy-unfiltered.log')
# Normalize diagnostic whitespace so archived compiler suggestions pass diff checks.
for path in out.rglob('*.log'):
    path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()).rstrip() + '\n')

sources = [root / 'Cargo.toml', root / 'Cargo.lock', root / 'src/poly/gcd.rs',
           root / 'src/poly/reconstruction.rs', root / 'tests/reconstruction_bma.rs',
           root / 'examples/reconstruction_bma_benchmark.rs',
           root / 'examples/reconstruction_stress_benchmark.rs',
           root / 'examples/support/reconstruction_q.rs',
           root / 'benches/external/run_stress.py', root / 'benches/external/run_q_stress.py',
           Path(__file__).resolve(),
           root / 'target/debug/examples/reconstruction_bma_benchmark',
           root / 'target/debug/examples/reconstruction_stress_benchmark']
sources += sorted((root / 'src/poly/reconstruction').glob('*.rs'))
sources += [base / f'bma-q-{name}-logs/symbolica-stress' for name in ['polynomial', 'rational']]
(out / 'sources-and-binaries.sha256').write_text(''.join(
    f'{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root)}\n'
    for path in sources))
shutil.copyfile(root / 'Cargo.lock', out / 'Cargo.lock.snapshot')
(out / 'environment.txt').write_text('\n'.join(
    ' '.join(command) + '\n' + subprocess.check_output(command, cwd=root, text=True).strip()
    for command in [['git', 'rev-parse', 'HEAD'], ['git', 'branch', '--show-current'],
                    ['rustc', '--version'], ['cargo', '--version'], ['uname', '-sm']]
) + '\nCPU: AMD EPYC 9754 128-Core Processor\nProfile: dev, CARGO_PROFILE_DEV_DEBUG=0\n'
  'Measurements: serial; no explicit CPU affinity\n'
  'Commit above is the archive-time base; source hashes identify the measured changes.\n'
  'Clippy command: cargo clippy --test reconstruction_bma --example reconstruction_bma_benchmark -- -A clippy::never_loop\n')
print(f'Archived {sum(map(len, tables.values()))} measurements in {out}')
