#!/usr/bin/env python3
"""Run independent array qualification in isolated artifact directories."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile
p=argparse.ArgumentParser();p.add_argument('--kind',choices=['cpp','python'],required=True);p.add_argument('--build',type=Path,required=True)
a=p.parse_args();root=Path(__file__).resolve().parents[2];build=a.build.resolve()
with tempfile.TemporaryDirectory(prefix='paralyn-arrays-') as temporary:
    evidence=str(Path(temporary)/'proof')
    if a.kind=='cpp':
        command=[str(build/'array_tests'),str(build/'operators.prk'),evidence]
    else:
        command=[sys.executable,str(root/'tests/native/test_arrays.py'),'--module',str(build/'operators.prk'),'--artifacts',evidence]
    env=dict(os.environ,PARALYN_LIBRARY=str(build/'libparalyn_native.dylib'),PYTHONPATH=str(root/'bindings/python'))
    subprocess.run(command,env=env,check=True)
