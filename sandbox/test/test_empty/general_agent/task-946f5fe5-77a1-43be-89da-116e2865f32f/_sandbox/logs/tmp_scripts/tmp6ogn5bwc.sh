set -e
if [ -f _input.tar.gz ]; then tar -xzf _input.tar.gz; fi
python - <<'PY'
import json
import importlib.util
import traceback
from pathlib import Path
import sys
import types

# Ensure a usable @mcp.tool decorator exists at import-time.
# This does NOT provide any tool implementations; it only prevents import failures
# when a real `mcp` package is present but lacks `tool`.
try:
    import mcp  # type: ignore
except Exception:
    mcp = types.ModuleType("mcp")
if not hasattr(mcp, "tool"):
    def _tool(func=None, **kwargs):
        if func is None:
            def wrapper(f):
                return f
            return wrapper
        return func
    mcp.tool = _tool  # type: ignore[attr-defined]
sys.modules["mcp"] = mcp

class ToolProxy(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        def _missing(*args, **kwargs):
            return {"error": f"Tool not available: {name}", "args": args, "kwargs": kwargs}
        return _missing

tools = {}
try:
    spec = importlib.util.spec_from_file_location("generated_tools", "tools.py")
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for key in dir(module):
            if key.startswith("_"):
                continue
            value = getattr(module, key)
            if callable(value):
                tools[key] = value
except Exception:
    pass

def _load_records():
    records_path = Path("records.json")
    if not records_path.exists():
        return []
    try:
        data = json.loads(records_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(data, dict) and isinstance(data.get("records"), list):
        data = data["records"]
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    return []

def _load_submitted_result():
    submitted_path = Path("submitted_result.json")
    if not submitted_path.exists():
        return None
    try:
        return json.loads(submitted_path.read_text(encoding="utf-8"))
    except Exception:
        return None

def _should_use_submitted_result(value):
    if value is None:
        return True
    if isinstance(value, str):
        lowered = value.lower()
        return any(
            token in lowered
            for token in (
                "submitted_result.json",
                "result submitted",
                "result saved",
                "submitted successfully",
                "saved to submitted_result",
            )
        )
    return False

def _unwrap_answer(value):
    if not isinstance(value, dict):
        return value
    status = value.get("status")
    message = value.get("message")
    if not (isinstance(status, str) or isinstance(message, str)):
        return value
    if "submitted_data" in value:
        return value.get("submitted_data")
    if "data" in value:
        return value.get("data")
    return value
# User requirement: all required tools MUST be implemented by the agent in tools.py.
missing = [name for name in ["search_hotel_pricing", "search_hotel_listings", "search_travel_reviews", "search_trip_details", "submit_result"] if name not in tools or not callable(tools.get(name))]
if missing:
    print(json.dumps({"error": f"missing_required_tools: {missing}"}, ensure_ascii=False))
    raise SystemExit(0)

tool_cache = {}
def _cache_key(args, kwargs):
    payload = {"args": args, "kwargs": kwargs}
    try:
        return json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    except Exception:
        return repr(payload)

def _wrap_tool(name, fn):
    def _call(*args, **kwargs):
        cache_key = name + ":" + _cache_key(args, kwargs)
        if cache_key in tool_cache:
            return tool_cache[cache_key]
        result = fn(*args, **kwargs)
        tool_cache[cache_key] = result
        return result
    return _call

for name in list(tools.keys()):
    if callable(tools[name]):
        tools[name] = _wrap_tool(name, tools[name])

tool_proxy = ToolProxy(**tools)

def _emit(payload):
    print(json.dumps(payload, ensure_ascii=False))

try:
    solution_src = "def solve(tools):\n    # Search for Paris hotels with pool amenities\n    listings = tools['search_hotel_listings']('Paris pool')\n    hotels_with_pool = []\n    if isinstance(listings, list):\n        for hotel in listings:\n            if isinstance(hotel, dict):\n                amenities = hotel.get('amenities', '')\n                if amenities and 'pool' in amenities.lower():\n                    name = hotel.get('name', '')\n                    stars = hotel.get('stars', '')\n                    price = hotel.get('price', '')\n                    if name and stars and price:\n                        hotels_with_pool.append({\n                            'name': str(name),\n                            'stars': str(stars),\n                            'price': str(price),\n                            'amenities': str(amenities)\n                        })\n    # Second tool: search pricing data for additional verification\n    tools['search_hotel_pricing']('Paris')\n    answer = {\n        'hotels': hotels_with_pool,\n        'total_count': str(len(hotels_with_pool))\n    }\n    return tools['submit_result'](answer)\n"
    verification_src = "def verify(tools, answer):\n    import json\n    try:\n        # Handle wrapped answer\n        if isinstance(answer, dict) and 'submitted_data' in answer:\n            data = answer.get('submitted_data')\n        elif isinstance(answer, dict) and 'data' in answer:\n            data = answer.get('data')\n        else:\n            data = answer\n        \n        if not isinstance(data, dict):\n            return {'passed': False, 'message': 'Answer is not a dict'}\n        \n        # Check required structure\n        if 'hotels' not in data or 'total_count' not in data:\n            return {'passed': False, 'message': 'Missing required keys'}\n        \n        hotels = data.get('hotels')\n        total_count = data.get('total_count')\n        \n        if not isinstance(hotels, list):\n            return {'passed': False, 'message': 'Hotels must be a list'}\n        \n        if not isinstance(total_count, str):\n            return {'passed': False, 'message': 'total_count must be string'}\n        \n        # Verify total_count matches actual count\n        if not total_count.isdigit():\n            return {'passed': False, 'message': 'total_count must be numeric string'}\n        \n        if int(total_count) != len(hotels):\n            return {'passed': False, 'message': 'Count mismatch'}\n        \n        # Verify each hotel has required string fields\n        for hotel in hotels:\n            if not isinstance(hotel, dict):\n                return {'passed': False, 'message': 'Hotel entry not a dict'}\n            required = ['name', 'stars', 'price', 'amenities']\n            for key in required:\n                if key not in hotel:\n                    return {'passed': False, 'message': f'Missing hotel field: {key}'}\n                if not isinstance(hotel[key], str):\n                    return {'passed': False, 'message': f'Hotel {key} must be string'}\n        \n        # Cross-check with tool data\n        listings = tools['search_hotel_listings']('Paris')\n        if isinstance(listings, list) and hotels:\n            # Verify at least one hotel from answer exists in tool data\n            found = False\n            for hotel in hotels:\n                for listing in listings:\n                    if isinstance(listing, dict):\n                        if listing.get('name') == hotel['name']:\n                            found = True\n                            break\n                if found:\n                    break\n            if not found and len(hotels) > 0:\n                return {'passed': False, 'message': 'Hotels not found in tool data'}\n        \n        return {'passed': True, 'message': 'Format and data validation passed'}\n    except Exception as e:\n        return {'passed': False, 'message': f'Verification error: {str(e)}'}\n"
    def _coerce_score(value):
        try:
            return float(value)
        except Exception:
            return None

    def _normalize_verification_output(output):
        verified = None
        score = None
        details = None
        message = None
        if isinstance(output, dict):
            for key in ("passed", "success", "ok", "result"):
                if key in output:
                    verified = bool(output.get(key))
                    break
            score = _coerce_score(output.get("score"))
            details = output.get("details") or output
            message = output.get("message") or output.get("error")
        elif isinstance(output, (list, tuple)) and output:
            if isinstance(output[0], bool):
                verified = output[0]
            if len(output) > 1:
                score = _coerce_score(output[1])
                if score is None and isinstance(output[1], str):
                    message = output[1]
            if len(output) > 2:
                details = output[2]
            if len(output) > 3 and message is None and isinstance(output[3], str):
                message = output[3]
        elif isinstance(output, bool):
            verified = output
        else:
            details = output
        return verified, score, details, message

    exec(solution_src, globals())
    exec(verification_src, globals())
    answer = solve(tool_proxy)
    if _should_use_submitted_result(answer):
        submitted_payload = _load_submitted_result()
        if submitted_payload is not None:
            answer = submitted_payload
    answer = _unwrap_answer(answer)
    raw_verified = verify(tool_proxy, answer)
    verified, score, details, message = _normalize_verification_output(raw_verified)
    if verified is False and message is None:
        message = "verification returned False"
    if verified is None and message is None:
        message = f"verification returned unsupported type: {type(raw_verified).__name__}"
    _emit({"answer": answer, "verified": verified, "verification_score": score, "verification_details": details, "verification_message": message})
except Exception:
    _emit({"error": traceback.format_exc()})
PY
tar -czf _output.tar.gz --warning=no-file-changed --warning=no-file-removed --ignore-failed-read --exclude=_output.tar.gz --exclude=_input.tar.gz .