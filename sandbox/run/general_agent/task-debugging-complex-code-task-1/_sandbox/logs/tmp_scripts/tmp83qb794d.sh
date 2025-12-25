set -e
if [ -f _input.tar.gz ]; then tar -xzf _input.tar.gz; fi
SERPER_API_KEY=359135a8666e6c3934dc758cd2c48fdb21621fc9 SEARCH_QUERY="Debugging complex code modules with highest bug recurrence" SEARCH_MAX=6 python - <<'PY'
import json, urllib.request, os
query = os.environ.get('SEARCH_QUERY', '')
max_results = int(os.environ.get('SEARCH_MAX', '5'))
api_key = os.environ.get('SERPER_API_KEY', '')
if not api_key:
    print(json.dumps({'error': 'missing_api_key'}))
    raise SystemExit(0)
url = 'https://google.serper.dev/search'
payload = json.dumps({'q': query}).encode('utf-8')
req = urllib.request.Request(url, data=payload, headers={'X-API-KEY': api_key, 'Content-Type': 'application/json'})
try:
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode('utf-8', 'replace'))
    results = []
    organic = data.get('organic') or []
    for item in organic[:max_results]:
        if not isinstance(item, dict):
            continue
        title = str(item.get('title') or '').strip()
        link = str(item.get('link') or item.get('url') or '').strip()
        snippet = str(item.get('snippet') or item.get('description') or '').strip()
        if title or link:
            results.append({'title': title, 'url': link, 'summary': snippet})
    if not results and data.get('answerBox'):
        box = data['answerBox']
        results.append({'title': str(box.get('title') or query), 'url': str(box.get('link') or ''), 'summary': str(box.get('answer') or box.get('snippet') or '')})
    print(json.dumps({'results': results}))
except Exception as e:
    print(json.dumps({'error': str(e)[:100], 'results': []}))
PY
tar -czf _output.tar.gz --warning=no-file-changed --warning=no-file-removed --ignore-failed-read --exclude=_output.tar.gz --exclude=_input.tar.gz .