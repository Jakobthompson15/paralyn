import signal,sys
from pathlib import Path
def stop(number,frame):
    print('child observed SIGINT',flush=True)
    raise SystemExit(77)
signal.signal(signal.SIGINT,stop)
Path('/Users/jakob/Documents/Codex/2026-09-25/files-pasted-by-the-user-unicuda/outputs/paralyn/artifacts/runs/product-87978b1/product/cli/interrupt-ready').write_text('ready')
while True: signal.pause()
