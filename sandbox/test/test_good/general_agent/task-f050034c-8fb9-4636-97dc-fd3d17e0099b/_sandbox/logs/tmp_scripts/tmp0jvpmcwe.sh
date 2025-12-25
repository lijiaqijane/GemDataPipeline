set -e
if [ -f _input.tar.gz ]; then tar -xzf _input.tar.gz; fi
JINA_API_KEY=jina_b7ac238911474f91a7c06eddede292d7qFimJoGGPvlsGNfxxUU8duLXqjRi JINA_QUERY_B64=UGFyaXMgVHJhdmVsIFBsYW5uaW5nIGRhdGFzZXQgc2NoZW1hIGNvbHVtbnMgZmllbGRzIGNzdiBqc29uIHRhYmxlIGxhdGl0dWRlIGxvbmdpdHVkZSBwcmljZSByYXRpbmcgdGltZSBpZCBuYW1lIGFkZHJlc3M= JINA_DOCS_B64=WyJVUkwgU291cmNlOiBodHRwczovL3d3dy5leHBlZGlhLmNvbS9QYXJpcy5kMTc5ODk4LkRlc3RpbmF0aW9uLVRyYXZlbC1HdWlkZXMgV2FybmluZzogVGFyZ2V0IFVSTCByZXR1cm5lZCBlcnJvciA0Mjk6IFRvbyBNYW55IFJlcXVlc3RzIFdhcm5pbmc6IFRoaXMgcGFnZSBtYXliZSByZXF1aXJpbmcgQ0FQVENIQSwgcGxlYXNlIG1ha2Ugc3VyZSB5b3UgYXJlIGF1dGhvcml6ZWQgdG8gYWNjZXNzIHRoaXMgcGFnZS4gU2hvdyB1cyB5b3VyIGh1bWFuIHNpZGUuLi4gLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0gV2UgY2FuJ3QgdGVsbCBpZiB5b3UncmUgYSBodW1hbiBvciBhIGJvdC4iLCAiMTJjNjY4M2UtNDNmNi00NjcwLWJmMWMtZmVmMzBiYWM4OWQzIl0= python - <<'PY'
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