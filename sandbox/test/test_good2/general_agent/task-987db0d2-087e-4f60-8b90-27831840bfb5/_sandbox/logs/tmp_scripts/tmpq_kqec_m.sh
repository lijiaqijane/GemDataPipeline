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
    solution_src = "def solve(tools):\n    # First tool: search hotel pricing data for Paris hotels with special offers\n    hotel_data = tools['search_hotel_pricing']('Paris special offer')\n    \n    # Second tool: search Paris hotel guides for additional information\n    guide_data = tools['search_paris_hotel_guides']('hotel')\n    \n    # Process hotel data to find available hotels with special offers\n    available_hotels = []\n    \n    # Check if hotel_data is valid and not an error\n    if isinstance(hotel_data, list) and len(hotel_data) > 0:\n        for hotel in hotel_data:\n            # Skip if hotel is an error record\n            if isinstance(hotel, dict) and 'error' not in hotel:\n                # Check if hotel is available and has a special offer\n                availability = hotel.get('availability', '')\n                special_offer = hotel.get('special_offer', '')\n                hotel_name = hotel.get('hotel_name', '')\n                room_type = hotel.get('room_type', '')\n                rate = hotel.get('rate', '')\n                \n                # Filter: must be available and have a non-empty special offer (not 'None')\n                if (availability == 'Available' and \n                    special_offer and \n                    special_offer.lower() != 'none' and \n                    special_offer.strip() != '' and\n                    hotel_name and room_type and rate):\n                    \n                    available_hotels.append({\n                        'hotel_name': hotel_name,\n                        'room_type': room_type,\n                        'rate': rate,\n                        'special_offer': special_offer\n                    })\n    \n    # Create answer according to required format\n    answer = {\n        'available_hotels_with_offers': available_hotels,\n        'total_count': len(available_hotels)\n    }\n    \n    return tools['submit_result'](answer)\n"
    verification_src = "def verify(tools, answer):\n    import json\n    \n    try:\n        # Handle wrapped answer from submit_result\n        if isinstance(answer, dict):\n            # Check if answer is wrapped with status/message/data\n            if 'status' in answer and 'data' in answer:\n                actual_data = answer.get('data')\n            # Or if it's wrapped with status/message/submitted_data\n            elif 'status' in answer and 'submitted_data' in answer:\n                actual_data = answer.get('submitted_data')\n            else:\n                actual_data = answer\n        else:\n            actual_data = answer\n        \n        # Check if answer is None\n        if actual_data is None:\n            return {'passed': False, 'message': 'Answer is None'}\n        \n        # Check required keys\n        if not isinstance(actual_data, dict):\n            return {'passed': False, 'message': 'Answer is not a dictionary'}\n        \n        required_keys = ['available_hotels_with_offers', 'total_count']\n        for key in required_keys:\n            if key not in actual_data:\n                return {'passed': False, 'message': f'Missing required key: {key}'}\n        \n        hotels_list = actual_data.get('available_hotels_with_offers')\n        total_count = actual_data.get('total_count')\n        \n        # Check types\n        if not isinstance(hotels_list, list):\n            return {'passed': False, 'message': 'available_hotels_with_offers is not a list'}\n        \n        if not isinstance(total_count, int):\n            return {'passed': False, 'message': 'total_count is not an integer'}\n        \n        # Check non-empty requirement\n        if len(hotels_list) == 0:\n            return {'passed': False, 'message': 'No hotels found - list is empty'}\n        \n        if total_count == 0:\n            return {'passed': False, 'message': 'total_count is zero but should have hotels'}\n        \n        if total_count != len(hotels_list):\n            return {'passed': False, 'message': f'total_count ({total_count}) does not match list length ({len(hotels_list)})'}\n        \n        # Check each hotel entry\n        required_hotel_keys = ['hotel_name', 'room_type', 'rate', 'special_offer']\n        for i, hotel in enumerate(hotels_list):\n            if not isinstance(hotel, dict):\n                return {'passed': False, 'message': f'Hotel entry {i} is not a dictionary'}\n            \n            for key in required_hotel_keys:\n                if key not in hotel:\n                    return {'passed': False, 'message': f'Hotel {i} missing key: {key}'}\n                \n                value = hotel.get(key)\n                if not isinstance(value, str):\n                    return {'passed': False, 'message': f'Hotel {i} {key} is not a string'}\n                \n                if value.strip() == '':\n                    return {'passed': False, 'message': f'Hotel {i} {key} is empty string'}\n            \n            # Special check: special_offer should not be 'None'\n            special_offer = hotel.get('special_offer', '')\n            if special_offer.lower() == 'none':\n                return {'passed': False, 'message': f'Hotel {i} special_offer is \\'None\\''}\n        \n        # Use a tool to cross-check data\n        # Search for Paris hotels to verify consistency\n        tool_result = tools['search_hotel_pricing']('Paris')\n        \n        # Even if tool returns error, we can still verify our answer structure\n        # Count how many hotels in tool result match our answer\n        if isinstance(tool_result, list):\n            tool_hotel_names = set()\n            for item in tool_result:\n                if isinstance(item, dict) and 'hotel_name' in item:\n                    tool_hotel_names.add(item['hotel_name'].strip().lower())\n            \n            # Check if any hotels in answer exist in tool results\n            answer_hotel_names = [h['hotel_name'].strip().lower() for h in hotels_list]\n            matching_count = sum(1 for name in answer_hotel_names if name in tool_hotel_names)\n            \n            if matching_count == 0 and len(tool_hotel_names) > 0:\n                return {'passed': False, 'message': 'No hotels in answer match tool search results', 'details': {'answer_hotels': answer_hotel_names[:3], 'tool_hotels': list(tool_hotel_names)[:3]}}\n        \n        # All checks passed\n        return {'passed': True, 'message': 'Verification successful', 'details': {'hotels_count': total_count}}\n        \n    except Exception as e:\n        # Exception-safe: return False with error message\n        return {'passed': False, 'message': f'Verification error: {str(e)}'}\n"
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