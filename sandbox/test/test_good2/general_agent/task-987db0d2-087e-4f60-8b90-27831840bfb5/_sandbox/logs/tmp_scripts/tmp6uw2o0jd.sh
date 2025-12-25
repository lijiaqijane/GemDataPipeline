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
missing = [name for name in ["search_hotel_pricing", "search_travel_recommendations", "search_tripadvisor_attractions", "search_paris_hotel_guides", "submit_result"] if name not in tools or not callable(tools.get(name))]
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
    solution_src = "def solve(tools):\n    # Search for hotel pricing data with special offers in Paris\n    hotel_data = tools['search_hotel_pricing']('Paris special offer')\n    \n    # Search Paris hotel guides for additional context\n    guide_data = tools['search_paris_hotel_guides']('hotel')\n    \n    # Filter hotels: must be Available and have a non-empty special offer\n    available_hotels = []\n    \n    if isinstance(hotel_data, list):\n        for hotel in hotel_data:\n            if isinstance(hotel, dict) and 'error' not in hotel:\n                availability = hotel.get('availability', '')\n                special_offer = hotel.get('special_offer', '')\n                hotel_name = hotel.get('hotel_name', '')\n                room_type = hotel.get('room_type', '')\n                rate = hotel.get('rate', '')\n                \n                # Check criteria: Available and has a real special offer\n                if (availability == 'Available' and \n                    special_offer and \n                    special_offer.lower() != 'none' and \n                    hotel_name and room_type and rate):\n                    \n                    # Format rate as string with currency if needed\n                    rate_str = str(rate)\n                    if not rate_str.startswith('$') and rate_str.replace('.', '').isdigit():\n                        rate_str = f'${rate_str}'\n                    \n                    available_hotels.append({\n                        'hotel_name': hotel_name,\n                        'room_type': room_type,\n                        'rate': rate_str,\n                        'special_offer': special_offer\n                    })\n    \n    # Build answer matching required format\n    answer = {\n        'available_hotels_with_offers': available_hotels,\n        'total_count': len(available_hotels)  # integer, not string\n    }\n    \n    return tools['submit_result'](answer)\n"
    verification_src = "def verify(tools, answer):\n    import json\n    \n    try:\n        # Unwrap answer if needed\n        if isinstance(answer, dict):\n            if 'status' in answer and 'data' in answer:\n                actual_data = answer.get('data')\n            elif 'status' in answer and 'submitted_data' in answer:\n                actual_data = answer.get('submitted_data')\n            else:\n                actual_data = answer\n        else:\n            actual_data = answer\n        \n        if actual_data is None:\n            return {'passed': False, 'message': 'Answer is None'}\n        \n        if not isinstance(actual_data, dict):\n            return {'passed': False, 'message': 'Answer is not a dictionary'}\n        \n        # Check required keys\n        required_keys = ['available_hotels_with_offers', 'total_count']\n        for key in required_keys:\n            if key not in actual_data:\n                return {'passed': False, 'message': f'Missing required key: {key}'}\n        \n        hotels_list = actual_data.get('available_hotels_with_offers', [])\n        total_count = actual_data.get('total_count')\n        \n        # total_count must be integer\n        if not isinstance(total_count, int):\n            return {'passed': False, 'message': 'total_count must be integer, not string'}\n        \n        # Consistency check: total_count should match list length\n        if total_count != len(hotels_list):\n            return {'passed': False, 'message': 'total_count does not match list length'}\n        \n        # Non\u2011emptiness check: if there are hotels, they must have content\n        if total_count > 0:\n            if not hotels_list:\n                return {'passed': False, 'message': 'total_count > 0 but list is empty'}\n            \n            required_fields = ['hotel_name', 'room_type', 'rate', 'special_offer']\n            for hotel in hotels_list:\n                if not isinstance(hotel, dict):\n                    return {'passed': False, 'message': 'Hotel entry is not a dict'}\n                \n                for field in required_fields:\n                    if field not in hotel:\n                        return {'passed': False, 'message': f'Missing field {field} in hotel'}\n                    \n                    value = hotel.get(field)\n                    if not isinstance(value, str) or not value.strip():\n                        return {'passed': False, 'message': f'Field {field} is empty or not a string'}\n                \n                # Special offer must not be 'None'\n                if hotel.get('special_offer', '').lower() == 'none':\n                    return {'passed': False, 'message': 'special_offer contains \"None\"'}\n        \n        # Cross\u2011check with a data tool\n        check_data = tools['search_hotel_pricing']('Available')\n        if isinstance(check_data, list) and len(check_data) > 0:\n            # Verify that at least some hotels in the dataset have availability 'Available'\n            available_in_dataset = any(\n                isinstance(h, dict) and h.get('availability') == 'Available'\n                for h in check_data if isinstance(h, dict)\n            )\n            \n            # If dataset has Available hotels but answer has none, that's suspicious\n            if available_in_dataset and total_count == 0:\n                return {'passed': False, 'message': 'Dataset has Available hotels but answer shows none'}\n        \n        return {'passed': True, 'message': 'Verification passed'}\n    \n    except Exception as e:\n        return {'passed': False, 'message': f'Exception during verification: {str(e)}'}\n"
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