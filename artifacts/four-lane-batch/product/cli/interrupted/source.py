import signal,sys
from pathlib import Path
def stop(number,frame):
    print('child observed SIGINT',flush=True)
    raise SystemExit(77)
signal.signal(signal.SIGINT,stop)
Path('/Users/jakob/Desktop/paralyn/.claude/worktrees/wf_2f802cff-5d1-1/artifacts/runs/capture-cccb177/product/cli/interrupt-ready').write_text('ready')
while True: signal.pause()
