import json,sys
print(json.dumps(sys.argv[1:], ensure_ascii=False), flush=True)
print('Verification: PASS', flush=True)
print('child stderr Ω', file=sys.stderr, flush=True)
raise SystemExit(int(sys.argv[1]))
