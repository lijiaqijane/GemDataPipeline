set -e
if [ -f _input.tar.gz ]; then tar -xzf _input.tar.gz; fi
URL=http://www.zigpoll.com/manifest.json TARGET=data/raw/manifest-json.json MAX_BYTES=20000000 TIMEOUT=15 python - <<'PY'
import json, os, pathlib, urllib.request
url = os.environ.get('URL', '')
target = os.environ.get('TARGET', '')
max_bytes = int(os.environ.get('MAX_BYTES', '6000000'))
timeout = float(os.environ.get('TIMEOUT', '15'))
errors = []
headers_list = [
  {'User-Agent': 'Mozilla/5.0', 'Accept': '*/*', 'Accept-Language': 'en-US,en;q=0.9'},
  {'User-Agent': 'Mozilla/5.0', 'Accept': 'text/csv,application/json;q=0.9,*/*;q=0.8'},
]
try:
    if not url or not target:
        raise ValueError('Missing URL or TARGET')
    path = pathlib.Path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        print(json.dumps({'ok': True, 'path': str(path)}))
        raise SystemExit(0)
    def attempt(headers: dict[str, str]) -> None:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content_type = (resp.headers.get('Content-Type') or '').lower()
            size = 0
            with open(path, 'wb') as f:
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    if size == 0:
                        lower = chunk[:512].lower()
                        if 'text/html' in content_type or b'<html' in lower or b'access denied' in lower:
                            raise ValueError('HTML/block page')
                    size += len(chunk)
                    if size > max_bytes:
                        raise ValueError('File too large')
                    f.write(chunk)
            if size == 0:
                raise ValueError('Empty file')
    ok = False
    for headers in headers_list:
        try:
            attempt(headers)
            ok = True
            break
        except Exception as e:
            errors.append(str(e)[:120])
            if path.exists():
                path.unlink(missing_ok=True)
    if not ok:
        raise ValueError('; '.join(errors)[:200])
    print(json.dumps({'ok': True, 'path': str(path)}))
except Exception as e:
    if 'path' in locals() and path.exists():
        path.unlink(missing_ok=True)
    print(json.dumps({'ok': False, 'error': str(e)[:200]}))
PY
tar -czf _output.tar.gz --warning=no-file-changed --warning=no-file-removed --ignore-failed-read --exclude=_output.tar.gz --exclude=_input.tar.gz .