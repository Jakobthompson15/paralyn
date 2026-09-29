#!/usr/bin/env python3
"""Exercise complete native GPU applications through the actual CLI, including export."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
p=argparse.ArgumentParser();p.add_argument('--paralyn',type=Path,required=True);p.add_argument('--artifacts',type=Path)
a=p.parse_args();root=Path(__file__).resolve().parents[2];cli=a.paralyn.resolve()
def run(work):
    work.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,PARALYN_PYTHON=sys.executable)
    for name in ('PYTHONPATH','PARALYN_LIBRARY','PARALYN_OPERATORS','PARALYN_ARTIFACT_DIR','PARALYN_RUNTIME_LOG','PARALYN_EVENT_LOG'):
        env.pop(name,None)
    reports=[]
    for extension in ('cpp','py'):
        source=work/('native example ü.'+extension);shutil.copyfile(root/'examples/native'/('arrays.'+extension),source)
        target=work/extension
        result=subprocess.run([str(cli),'run',str(source),'--json','--device','metal:0','--artifacts',str(target)],env=env,cwd=work,capture_output=True,text=True,check=False,timeout=90)
        (work/(extension+'.stdout')).write_text(result.stdout);(work/(extension+'.stderr')).write_text(result.stderr)
        assert result.returncode==0,(result.returncode,result.stdout,result.stderr)
        report=json.loads(result.stdout)
        assert report['status']=='completed' and report['verification']['status']=='not_requested'
        assert report['application']['exit_code']==0 and report['failure_origin'] is None
        assert report['runtime_errors_observed'] is False
        assert 'Verification: PASS native' in Path(report['application']['stdout']).read_text()
        assert Path(report['application']['stderr']).read_text()==''
        evidence=report['runtime_evidence'];assert evidence['cpu_fallback'] is False and evidence['backend']=='Metal'
        assert len(evidence['launches'])==2 and evidence['runtime_owned_current_buffer_bytes']==0
        path=Path(report['runtime_evidence_path']);assert path.parent==target/'native' and path.is_file()
        for event in evidence['launches']:
            assert event['command_status']=='completed' and not event['error']
            assert event['gpu_duration_valid'] and 0<event['gpu_start_seconds']<event['gpu_end_seconds']
            assert (path.parent/event['source_file']).is_file()
        reports.append({'language':extension,'gpu_events':2,'device':evidence['device'],'source_revision':evidence['paralyn_commit'],'report':str(target/'report.json')})
    (work/'verification.json').write_text(json.dumps({'verification':'PASS','gpu_events':4,'cases':reports},indent=2)+'\n')
    print('Native CLI arrays: PASS (4 physical GPU events; copied standalone C++ and Python references)')
if a.artifacts:
    if a.artifacts.exists():raise RuntimeError('artifact directory must be new')
    run(a.artifacts.resolve())
else:
    with tempfile.TemporaryDirectory(prefix='paralyn-native-cli-') as temporary:run(Path(temporary))
