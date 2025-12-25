set -e
if [ -f _input.tar.gz ]; then tar -xzf _input.tar.gz; fi
JINA_API_KEY=jina_b7ac238911474f91a7c06eddede292d7qFimJoGGPvlsGNfxxUU8duLXqjRi JINA_QUERY_B64=RGVidWdnaW5nIGNvbXBsZXggY29kZSBkYXRhc2V0IHNjaGVtYSBjb2x1bW5zIGZpZWxkcyBjc3YganNvbiB0YWJsZSBsYXRpdHVkZSBsb25naXR1ZGUgcHJpY2UgcmF0aW5nIHRpbWUgaWQgbmFtZSBhZGRyZXNz JINA_DOCS_B64=WyJUaXRsZTogSnVzdCBhIG1vbWVudC4uLiBVUkwgU291cmNlOiBodHRwczovL21lZGl1bS5jb20vQG9uc2l0ZXIvZGVidWdnaW5nLWNvbXBsZXgtY29kZWJhc2VzLWEtY29tcHJlaGVuc2l2ZS1ndWlkZS01Zjg1MjhjNDhjZTQgV2FybmluZzogVGFyZ2V0IFVSTCByZXR1cm5lZCBlcnJvciA0MDM6IEZvcmJpZGRlbiBXYXJuaW5nOiBUaGlzIHBhZ2UgbWF5YmUgcmVxdWlyaW5nIENBUFRDSEEsIHBsZWFzZSBtYWtlIHN1cmUgeW91IGFyZSBhdXRob3JpemVkIHRvIGFjY2VzcyB0aGlzIHBhZ2UuICFbSW1hZ2UgMTogSWNvbiBmb3IgbWVkaXVtLmNvbV0oaHR0cHM6Ly9tZWRpdW0uY29tL2Zhdmljb24uaWNvKW1lZGl1bS5jb20iLCAiLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLSBWZXJpZnkgeW91IGFyZSBodW1hbiBieSBjb21wbGV0aW5nIHRoZSBhY3Rpb24gYmVsb3cuIG1lZGl1bS5jb20gbmVlZHMgdG8gcmV2aWV3IHRoZSBzZWN1cml0eSBvZiB5b3VyIGNvbm5lY3Rpb24gYmVmb3JlIHByb2NlZWRpbmcuIl0= python - <<'PY'
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