set -e
if [ -f _input.tar.gz ]; then tar -xzf _input.tar.gz; fi
JINA_API_KEY=jina_b7ac238911474f91a7c06eddede292d7qFimJoGGPvlsGNfxxUU8duLXqjRi JINA_QUERY_B64=RGVidWdnaW5nIGNvbXBsZXggY29kZSBkYXRhc2V0IHNjaGVtYSBjb2x1bW5zIGZpZWxkcyBjc3YganNvbiB0YWJsZSBsYXRpdHVkZSBsb25naXR1ZGUgcHJpY2UgcmF0aW5nIHRpbWUgaWQgbmFtZSBhZGRyZXNz JINA_DOCS_B64=WyJUaXRsZTogSnVzdCBhIG1vbWVudC4uLiBVUkwgU291cmNlOiBodHRwczovL3d3dy5xdW9yYS5jb20vSG93LWNhbi1JLWVmZmVjdGl2ZWx5LWRlYnVnLWFuZC10cm91Ymxlc2hvb3QtY29tcGxleC1jb2RlLWlzc3VlcyBXYXJuaW5nOiBUYXJnZXQgVVJMIHJldHVybmVkIGVycm9yIDQwMzogRm9yYmlkZGVuIFdhcm5pbmc6IFRoaXMgcGFnZSBtYXliZSByZXF1aXJpbmcgQ0FQVENIQSwgcGxlYXNlIG1ha2Ugc3VyZSB5b3UgYXJlIGF1dGhvcml6ZWQgdG8gYWNjZXNzIHRoaXMgcGFnZS4gIVtJbWFnZSAxOiBJY29uIGZvciB3d3cucXVvcmEuY29tXShodHRwczovL3d3dy5xdW9yYS5jb20vZmF2aWNvbi5pY28pd3d3LnF1b3JhLmNvbSIsICItLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tIFZlcmlmeSB5b3UgYXJlIGh1bWFuIGJ5IGNvbXBsZXRpbmcgdGhlIGFjdGlvbiBiZWxvdy4gd3d3LnF1b3JhLmNvbSBuZWVkcyB0byByZXZpZXcgdGhlIHNlY3VyaXR5IG9mIHlvdXIgY29ubmVjdGlvbiBiZWZvcmUgcHJvY2VlZGluZy4iXQ== python - <<'PY'
import base64, json, os, urllib.request
query_b64 = os.environ.get('JINA_QUERY_B64', '')
docs_b64 = os.environ.get('JINA_DOCS_B64', '')
query = base64.b64decode(query_b64.encode('ascii')).decode('utf-8', 'replace') if query_b64 else ''
docs_json = base64.b64decode(docs_b64.encode('ascii')).decode('utf-8', 'replace') if docs_b64 else '[]'
docs = json.loads(docs_json)
api_key = os.environ.get('JINA_API_KEY', '')
payload = json.dumps({
  'model': 'jina-reranker-v2-base-multilingual',
  'query': query,
  'documents': docs,
  'top_n': min(12, len(docs))
}).encode('utf-8')
req = urllib.request.Request(
  'https://api.jina.ai/v1/rerank',
  data=payload,
  headers={'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}
)
try:
  with urllib.request.urlopen(req, timeout=15) as resp:
    data = json.loads(resp.read().decode('utf-8', 'replace'))
  print(json.dumps({'data': data.get('data', [])}))
except Exception as e:
  print(json.dumps({'error': str(e)[:200]}))
PY
tar -czf _output.tar.gz --warning=no-file-changed --warning=no-file-removed --ignore-failed-read --exclude=_output.tar.gz --exclude=_input.tar.gz .