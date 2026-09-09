import csv, hashlib, json, shutil, statistics, tarfile
from pathlib import Path
base=Path('target/reconstruction-external')
out=Path('benches/results/reconstruction/ordering-traces')
out.mkdir(parents=True,exist_ok=True)
required={'ordering-final-nb0':3319,'ordering-final-graph5':945,'ordering-final-sparse':4,'ordering-final-q':5,'ordering-final-sheared':4,'ordering-final-b16':36,'ordering-final-diamond':36,'ordering-final-box2l':12,'ordering-final-xbox2l2m':12,'trace-final-b16':36,'trace-final-xbox':24,'ordering-rare-b16':4,'ordering-fire7-dimension-last':4}
read=lambda p:list(csv.DictReader(p.open()))
final={}
for name,count in required.items():
    rows=read(base/(name+'.csv'))
    assert len(rows)==count,(name,len(rows),count)
    assert all(r['status']=='ok' for r in rows),name
    final[name]=rows
regressions=[]
old=Path('benches/results/reconstruction/independent-ibp')
for name,old_name in [('nb0','nb0'),('graph5','graph5'),('sparse','sparse'),('q','q'),('sheared','sheared'),('box2l','box'),('xbox2l2m','xbox')]:
    previous={(r['case'],r['method'],r['seed']):r for r in read(old/f'ibp-factor-final-{old_name}.csv')}
    for r in final['ordering-final-'+name]:
        reference=previous[(r['case'],r['method'],r['seed'])]
        for field in ['probes','probes_by_prime','selected_methods','selection_probes','sparse_rows','sparse_row_fallbacks']:
            if field in r and field in reference: assert r[field]==reference[field],(name,r['case'],field)
    regressions.append(dict(suite=name,runs=len(final['ordering-final-'+name]),unchanged=True))
for name,ref_file,rare_file in [('b16','ibp-tth2l_b16.csv','ordering-rare-b16.csv'),('diamond','ibp-diamond3l.csv',None)]:
    refs=read(base/ref_file)+(read(base/rare_file) if rare_file else [])
    for row in final['ordering-final-'+name]:
        if row['method']=='Automatic':
            competitors=[int(r['probes']) for r in refs if r['case']==row['case'] and r['method']!='Automatic']
            assert int(row['probes'])<min(competitors)
        else:
            expected=next(r for r in refs if r['case']==row['case'] and r['method']==row['method'])
            assert row['probes']==expected['probes']
trace_medians={}
for name,expanded in [('b16','ordering-final-b16'),('xbox','ordering-final-xbox2l2m')]:
    rows=final['trace-final-'+name]
    refs=final[expanded]
    if name=='xbox': refs+=read(base/'ibp-xbox2l2m.csv')
    for row in rows:
        assert row['oracle']=='ratracer_trace'
        expected=next(r for r in refs if r['case']==row['case'] and r['method']==row['method'])
        assert row['probes']==expected['probes']
    for case in sorted({r['case'] for r in rows}):
        methods={m:[r for r in rows if r['case']==case and r['method']==m] for m in {r['method'] for r in rows}}
        medians={m:statistics.median(float(r['elapsed_us']) for r in rs)/1e6 for m,rs in methods.items()}
        assert medians['Automatic']<medians['FireFly_scan']
        trace_medians[case]=medians
paths=sorted(set(base.glob('ordering-*.csv'))|set(base.glob('trace-*.csv'))|{base/'ibp-diamond3l.csv',base/'ibp-tth2l_b16.csv',base/'ibp-tth2l_b16-dimension-last.csv'})
statuses={}
for p in paths:
    rows=read(p)
    for row in rows:
        statuses[row['status']]=statuses.get(row['status'],0)+1
        assert row['status'] in {'ok','probe_limit','unsupported_trace_backend','unsupported_variable_count'},(p,row)
    shutil.copyfile(p,out/p.name)
logs=['ordering-final-tests.log','ordering-final-build.log','ordering-clippy.log','ordering-fmt.log','ordering-rare-build.log','ordering-rare-controls.log','trace-driver-controls.log','trace-check-b16.log','trace-check-xbox.log','trace-oracle-build.log','trace-firefly-final-build.log','trace-final-build.log','prepare-diamond3l.log','prepare-tth2l_b16.log','validate-diamond3l.log','validate-tth2l_b16.log']
for name in logs: shutil.copyfile(base/name,out/name)
with tarfile.open(out/'benchmark-logs.tar.gz','w:gz') as tar:
    for p in paths:
        folder=base/(p.stem+'-logs')
        if folder.is_dir(): tar.add(folder,arcname=folder.name)
        progress=p.with_suffix('.log')
        if progress.exists(): tar.add(progress,arcname=progress.name)
    tar.add(base/'trace-driver-controls'/'failure.c',arcname='trace-driver-controls/failure.c')
    for p in (base/'trace-driver-controls').glob('*.log'): tar.add(p,arcname='trace-driver-controls/'+p.name)
    tar.add(base/'rare-controls',arcname='rare-adapter-controls')
with tarfile.open(out/'inputs-traces-and-provenance.tar.gz','w:gz',dereference=True) as tar:
    for family in ['diamond3l','tth2l_b16','xbox2l2m']:
        inputs=base/'ibp-inputs'/family
        manifest=json.loads((inputs/'manifest.json').read_text())
        for name in ['manifest.json','validation.json','suite-cases.txt']:
            tar.add(inputs/name,arcname=f'ibp-inputs/{family}/{name}')
        if (inputs/'trace-abi-validation.json').exists(): tar.add(inputs/'trace-abi-validation.json',arcname=f'ibp-inputs/{family}/trace-abi-validation.json')
        for case in manifest['selected_cases']:
            for suffix in ['', '.variables','.trace']:
                tar.add(inputs/(case+suffix),arcname=f'ibp-inputs/{family}/{case+suffix}')
        if family!='xbox2l2m':
            work=base/'ibp-work'/family
            for name in ['preparation-logs','config','integrals','master-eqns','target-list','export-equations.yaml','top.outputs','selected.names','selected.outputs','selected.results']:
                tar.add(work/name,arcname=f'ibp-work/{family}/{name}')
files=[Path('src/poly/reconstruction.rs'),*Path('src/poly/reconstruction').glob('*.rs'),Path('tests/rational_reconstruction.rs'),Path('examples/support/reconstruction_q.rs'),Path('examples/support/trace_oracle.rs'),*Path('benches/external').glob('*trace*'),Path('benches/external/firefly_q_stress.cpp'),Path('benches/external/run_q_stress.py'),Path('benches/external/rare/src/main.rs'),Path('target/release/examples/reconstruction_stress_benchmark')]
files += [base/name for name in ['ordering-baseline','ordering-firefly-baseline','ordering-rare-baseline','firefly-q-stress','fire7-q-stress','rare-q-stress','libtrace-oracle.so']]
(out/'sources-and-binaries.sha256').write_text(''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p}\n' for p in files))
summary=dict(baseline='e9e6fe92',final_successful_runs=sum(required.values()),archived_statuses=statuses,regressions=regressions,trace_median_seconds=trace_medians,rare_adapter_variables=8,trace_backends=['Symbolica','FireFly'],native_prime_policies_unchanged=True,limitations=['Scalar, single-output reconstruction; not a complete multi-output IBP reduction.','Shared host, not isolated; timings exclude trace loading and input preparation.','FIRE7 native 64-bit primes exceed interpreter range; rare has no trace binding.','ordering-rare-unsupported.csv predates the extension to eight variables and is superseded by ordering-rare-b16.csv.'])
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
