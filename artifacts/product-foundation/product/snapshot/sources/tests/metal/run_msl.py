#!/usr/bin/env python3
"""Qualified public MSL input through container and native runtime, no test hook."""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile
p=argparse.ArgumentParser()
p.add_argument('--paralyn',required=True);p.add_argument('--driver',required=True);p.add_argument('--source',required=True);p.add_argument('--manifest',required=True)
a=p.parse_args()
with tempfile.TemporaryDirectory(prefix='paralyn-msl-') as temporary:
    root=Path(temporary);module=root/'module.prx'
    subprocess.run([a.paralyn,'compile',a.source,'--manifest',a.manifest,'--output',str(module)],check=True)
    subprocess.run([a.driver,str(module),str(root/'proof')],check=True)
