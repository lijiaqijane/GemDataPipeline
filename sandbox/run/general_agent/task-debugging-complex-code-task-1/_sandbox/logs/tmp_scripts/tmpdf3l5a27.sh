set -e
if [ -f _input.tar.gz ]; then tar -xzf _input.tar.gz; fi
JINA_API_KEY=jina_b7ac238911474f91a7c06eddede292d7qFimJoGGPvlsGNfxxUU8duLXqjRi JINA_QUERY_B64=RGVidWdnaW5nIGNvbXBsZXggY29kZSBkYXRhc2V0IHNjaGVtYSBjb2x1bW5zIGZpZWxkcyBjc3YganNvbiB0YWJsZSBsYXRpdHVkZSBsb25naXR1ZGUgcHJpY2UgcmF0aW5nIHRpbWUgaWQgbmFtZSBhZGRyZXNz JINA_DOCS_B64=WyJUaXRsZTogSnVzdCBhIG1vbWVudC4uLiBVUkwgU291cmNlOiBodHRwczovL3d3dy5xdW9yYS5jb20vV2hhdC1hcHByb2FjaGVzLXdvcmstYmVzdC1mb3ItZGVidWdnaW5nLWNvbXBsZXgtY29kZS1pc3N1ZXMtZHVyaW5nLXRoZS1kZXZlbG9wbWVudC1wcm9jZXNzIFdhcm5pbmc6IFRhcmdldCBVUkwgcmV0dXJuZWQgZXJyb3IgNDAzOiBGb3JiaWRkZW4gV2FybmluZzogVGhpcyBwYWdlIG1heWJlIHJlcXVpcmluZyBDQVBUQ0hBLCBwbGVhc2UgbWFrZSBzdXJlIHlvdSBhcmUgYXV0aG9yaXplZCB0byBhY2Nlc3MgdGhpcyBwYWdlLiAhW0ltYWdlIDE6IEljb24gZm9yIiwgInd3dy5xdW9yYS5jb21dKGh0dHBzOi8vd3d3LnF1b3JhLmNvbS9mYXZpY29uLmljbyl3d3cucXVvcmEuY29tIC0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0gVmVyaWZ5IHlvdSBhcmUgaHVtYW4gYnkgY29tcGxldGluZyB0aGUgYWN0aW9uIGJlbG93LiB3d3cucXVvcmEuY29tIG5lZWRzIHRvIHJldmlldyB0aGUgc2VjdXJpdHkgb2YgeW91ciBjb25uZWN0aW9uIGJlZm9yZSBwcm9jZWVkaW5nLiJd python - <<'PY'
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