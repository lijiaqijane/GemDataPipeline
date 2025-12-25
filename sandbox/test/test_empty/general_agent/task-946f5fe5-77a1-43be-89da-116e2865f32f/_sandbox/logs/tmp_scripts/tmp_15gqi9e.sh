set -e
if [ -f _input.tar.gz ]; then tar -xzf _input.tar.gz; fi
URL=https://www.ricksteves.com/europe/france/paris TIMEOUT=8 MAX_BYTES=2000000 DATA_EXTS=.csv,.db,.json,.jsonl,.ndjson,.parquet,.sqlite,.sqlite3,.tsv,.txt SAVE_DIR=data/raw_pages SAVE_RAW=0 python - <<'PY'
import hashlib, html, json, os, pathlib, re, urllib.parse, urllib.request
from html.parser import HTMLParser
url = os.environ.get('URL', '')
timeout = float(os.environ.get('TIMEOUT', '10'))
max_bytes = int(os.environ.get('MAX_BYTES', '2000000'))
exts = [e for e in os.environ.get('DATA_EXTS', '').split(',') if e]
save_dir = os.environ.get('SAVE_DIR', 'data/raw_pages')
save_raw = os.environ.get('SAVE_RAW', '1')
errors = []
def is_data_file(u: str) -> bool:
    path = urllib.parse.urlparse(u).path.lower()
    return any(path.endswith(ext) for ext in exts)
def extract(text: str, base_url: str) -> tuple[str, list[str], list[list[list[str]]]]:
    text = re.sub(r'<script[^>]*>.*?</script>', ' ', text, flags=re.I|re.S)
    text = re.sub(r'<style[^>]*>.*?</style>', ' ', text, flags=re.I|re.S)
    cleaned = re.sub(r'<[^>]+>', ' ', text)
    lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
    filtered = []
    for ln in lines:
        if len(ln) < 20:
            continue
        if re.search(r'[{};<>]|function\s*\(|var\s+|let\s+|const\s+|@keyframes', ln):
            continue
        if re.search(r'\b(document|window|navigator|jquery|\$)\b', ln, flags=re.I):
            continue
        filtered.append(ln)
    content = '\n'.join(filtered[:400])
    hrefs = re.findall(r'href=["\'](.*?)["\']', text, flags=re.I)
    links = []
    for href in hrefs:
        full = urllib.parse.urljoin(base_url, href)
        if is_data_file(full):
            links.append(full)
    tables = []
    try:
        class TableParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.tables = []
                self._table = []
                self._row = []
                self._cell = []
                self._in_table = False
                self._in_row = False
                self._in_cell = False
            def handle_starttag(self, tag, attrs):
                if tag == 'table':
                    self._in_table = True
                    self._table = []
                elif self._in_table and tag == 'tr':
                    self._in_row = True
                    self._row = []
                elif self._in_row and tag in ('td', 'th'):
                    self._in_cell = True
                    self._cell = []
            def handle_endtag(self, tag):
                if tag in ('td', 'th') and self._in_cell:
                    cell_text = ' '.join(self._cell).strip()
                    cell_text = re.sub(r'\s+', ' ', cell_text)
                    self._row.append(cell_text)
                    self._in_cell = False
                elif tag == 'tr' and self._in_row:
                    if self._row and any(c for c in self._row):
                        self._table.append(self._row)
                    self._in_row = False
                elif tag == 'table' and self._in_table:
                    if self._table:
                        self.tables.append(self._table)
                    self._in_table = False
            def handle_data(self, data):
                if self._in_cell:
                    self._cell.append(data)
        parser = TableParser()
        parser.feed(text)
        tables = parser.tables[:6]
    except Exception:
        tables = []
    return content, links, tables
def _safe_save(raw_text: str, url_value: str) -> tuple[str, int]:
    try:
        if save_raw in {'0', 'false', 'no'}:
            return '', 0
        base = pathlib.Path(save_dir)
        if base.is_absolute():
            base = pathlib.Path('data/raw_pages')
        base.mkdir(parents=True, exist_ok=True)
        slug = hashlib.md5(url_value.encode('utf-8')).hexdigest()[:16]
        target = base / f'page_{slug}.html'
        target.write_text(raw_text, encoding='utf-8', errors='replace')
        return str(target.as_posix()), len(raw_text.encode('utf-8', errors='replace'))
    except Exception:
        return '', 0
def fetch_once(u: str, headers: dict[str, str]) -> tuple[bool, str, list[str], list[list[list[str]]], str, str, int]:
    req = urllib.request.Request(u, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raw = raw[:max_bytes]
    text = raw.decode('utf-8', 'replace')
    raw_path, raw_size = _safe_save(text, u)
    content, links, tables = extract(text, u)
    return True, content[:10000], links[:20], tables[:6], '', raw_path, raw_size
def try_fetch(u: str) -> tuple[bool, str, list[str], list[list[list[str]]], str, str, int]:
    headers_list = [
        {
            'User-Agent': 'Mozilla/5.0',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
        },
        {
            'User-Agent': 'Mozilla/5.0',
            'Accept': '*/*',
            'Accept-Language': 'en-US,en;q=0.9',
        },
    ]
    for headers in headers_list:
        try:
            return fetch_once(u, headers)
        except Exception as e:
            errors.append(str(e)[:120])
    return False, '', [], [], errors[-1] if errors else 'fetch_failed', '', 0
def jina_proxy(u: str) -> str:
    parsed = urllib.parse.urlparse(u)
    scheme = 'http' if parsed.scheme == 'http' else 'https'
    return f"https://r.jina.ai/{scheme}://{parsed.netloc}{parsed.path}"
try:
    if not url.startswith(('http://', 'https://')):
        raise ValueError('Invalid URL')
    ok, content, links, tables, err, raw_path, raw_size = try_fetch(url)
    if not ok:
        proxy = jina_proxy(url)
        ok, content, links, tables, err, raw_path, raw_size = try_fetch(proxy)
    if ok:
        out = {'success': True, 'content': content, 'tables': tables, 'links': links, 'raw_path': raw_path, 'raw_size': raw_size, 'error': None, 'error_detail': None}
        print(json.dumps(out, ensure_ascii=False))
    else:
        out = {'success': False, 'content': '', 'tables': [], 'links': [], 'raw_path': '', 'raw_size': 0, 'error': err or 'fetch_failed', 'error_detail': '; '.join(errors)[:300]}
        print(json.dumps(out, ensure_ascii=False))
except Exception as e:
    print(json.dumps({'success': False, 'content': '', 'tables': [], 'links': [], 'raw_path': '', 'raw_size': 0, 'error': str(e)[:100], 'error_detail': '; '.join(errors)[:300]}, ensure_ascii=False))
PY
tar -czf _output.tar.gz --warning=no-file-changed --warning=no-file-removed --ignore-failed-read --exclude=_output.tar.gz --exclude=_input.tar.gz .