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
missing = [name for name in ["search_hotel_pricing", "search_hotel_directory", "search_travel_recommendations", "search_tripadvisor_attractions", "search_hotel_star_ratings", "submit_result"] if name not in tools or not callable(tools.get(name))]
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
    solution_src = "def solve(tools):\n    hotels = tools['search_hotel_directory']('Paris')\n    luxury_hotels = []\n    for hotel in hotels:\n        if isinstance(hotel, dict):\n            amenities = hotel.get('amenities', '').lower()\n            stars = hotel.get('stars', '')\n            if 'pool' in amenities and 'jacuzzi' in amenities and stars == '5':\n                name = hotel.get('name', '')\n                if name:\n                    luxury_hotels.append({\n                        'name': name,\n                        'stars': stars,\n                        'amenities': hotel.get('amenities', ''),\n                        'price': hotel.get('price', '')\n                    })\n    pricing = tools['search_hotel_pricing']('Paris')\n    for hotel in luxury_hotels:\n        seasonal_rate = 'Not available'\n        for price_rec in pricing:\n            if isinstance(price_rec, dict):\n                if price_rec.get('hotel_name', '') == hotel['name']:\n                    rate = price_rec.get('rate', '')\n                    if rate:\n                        seasonal_rate = str(rate)\n                        hotel['price'] = seasonal_rate\n                    break\n        hotel['seasonal_rate'] = seasonal_rate\n        try:\n            rate_val = float(seasonal_rate.replace('$', '').replace(',', '')) if seasonal_rate != 'Not available' else 0\n            hotel['3_night_cost'] = f'${rate_val * 3:.2f}'\n        except:\n            hotel['3_night_cost'] = 'Not available'\n    attractions = tools['search_tripadvisor_attractions']('Paris')\n    top_attractions = []\n    for attr in attractions:\n        if isinstance(attr, dict):\n            rating = attr.get('rating', '')\n            entry_fee = attr.get('entry_fee', '')\n            try:\n                rating_val = float(rating)\n                if rating_val >= 4.5 and entry_fee and 'free' not in entry_fee.lower():\n                    top_attractions.append({\n                        'name': attr.get('sub_category', ''),\n                        'description': attr.get('description', ''),\n                        'rating': rating,\n                        'entry_fee': entry_fee\n                    })\n            except:\n                continue\n    for attr in top_attractions:\n        fee_str = attr['entry_fee']\n        try:\n            fee_val = float(''.join(c for c in fee_str if c.isdigit() or c == '.'))\n            attr['cost_for_2'] = f'${fee_val * 2:.2f}'\n        except:\n            attr['cost_for_2'] = 'Not available'\n    total_hotel = 0\n    for hotel in luxury_hotels:\n        try:\n            total_hotel += float(hotel['3_night_cost'].replace('$', '').replace(',', '')) if hotel['3_night_cost'] != 'Not available' else 0\n        except:\n            pass\n    total_attractions = 0\n    for attr in top_attractions:\n        try:\n            total_attractions += float(attr['cost_for_2'].replace('$', '').replace(',', '')) if attr['cost_for_2'] != 'Not available' else 0\n        except:\n            pass\n    answer = {\n        'package_name': 'Luxury Paris Experience',\n        'hotels': [{\n            'name': h['name'],\n            'stars': h['stars'],\n            'amenities': h['amenities'],\n            'seasonal_rate': h['seasonal_rate'],\n            '3_night_cost': h['3_night_cost']\n        } for h in luxury_hotels],\n        'attractions': [{\n            'name': a['name'],\n            'description': a['description'],\n            'rating': a['rating'],\n            'entry_fee': a['entry_fee'],\n            'cost_for_2': a['cost_for_2']\n        } for a in top_attractions],\n        'cost_summary': {\n            'total_hotel_3_nights': f'${total_hotel:.2f}',\n            'total_attractions_2_people': f'${total_attractions:.2f}',\n            'grand_total': f'${total_hotel + total_attractions:.2f}'\n        }\n    }\n    return tools['submit_result'](answer)\n"
    verification_src = "def verify(tools, answer):\n    try:\n        import json\n        if answer is None:\n            return {'passed': False, 'message': 'Answer is None'}\n        data = answer\n        if isinstance(answer, dict):\n            if 'status' in answer and 'submitted_data' in answer:\n                data = answer['submitted_data']\n            elif 'status' in answer and 'data' in answer:\n                data = answer['data']\n        if not isinstance(data, dict):\n            return {'passed': False, 'message': 'Answer data is not a dict'}\n        required_keys = ['package_name', 'hotels', 'attractions', 'cost_summary']\n        for key in required_keys:\n            if key not in data:\n                return {'passed': False, 'message': f'Missing key: {key}'}\n        if not isinstance(data['hotels'], list) or not isinstance(data['attractions'], list):\n            return {'passed': False, 'message': 'Hotels or attractions not a list'}\n        cost_keys = ['total_hotel_3_nights', 'total_attractions_2_people', 'grand_total']\n        for key in cost_keys:\n            if key not in data['cost_summary']:\n                return {'passed': False, 'message': f'Missing cost key: {key}'}\n        hotels = tools['search_hotel_directory']('Paris')\n        paris_hotel_count = len([h for h in hotels if isinstance(h, dict) and h.get('city', '').lower() == 'paris'])\n        if paris_hotel_count == 0:\n            return {'passed': False, 'message': 'No Paris hotels found in directory'}\n        if len(data['hotels']) == 0:\n            return {'passed': False, 'message': 'No hotels in package'}\n        for hotel in data['hotels']:\n            if not isinstance(hotel, dict):\n                return {'passed': False, 'message': 'Hotel entry not a dict'}\n            hotel_keys = ['name', 'stars', 'amenities', 'seasonal_rate', '3_night_cost']\n            for key in hotel_keys:\n                if key not in hotel:\n                    return {'passed': False, 'message': f'Hotel missing key: {key}'}\n            if hotel['stars'] != '5':\n                return {'passed': False, 'message': f'Hotel {hotel[\"name\"]} not 5-star'}\n        for attr in data['attractions']:\n            if not isinstance(attr, dict):\n                return {'passed': False, 'message': 'Attraction entry not a dict'}\n            attr_keys = ['name', 'description', 'rating', 'entry_fee', 'cost_for_2']\n            for key in attr_keys:\n                if key not in attr:\n                    return {'passed': False, 'message': f'Attraction missing key: {key}'}\n            try:\n                rating = float(attr['rating'])\n                if rating < 4.5:\n                    return {'passed': False, 'message': f'Attraction rating {rating} < 4.5'}\n            except:\n                return {'passed': False, 'message': f'Invalid rating: {attr[\"rating\"]}'}\n        total_hotel = data['cost_summary']['total_hotel_3_nights']\n        total_attr = data['cost_summary']['total_attractions_2_people']\n        grand = data['cost_summary']['grand_total']\n        if not total_hotel.startswith('$') or not total_attr.startswith('$') or not grand.startswith('$'):\n            return {'passed': False, 'message': 'Costs not formatted as currency'}\n        return {'passed': True, 'message': 'Verification passed', 'details': {'hotels_count': len(data['hotels']), 'attractions_count': len(data['attractions'])}}\n    except Exception as e:\n        return {'passed': False, 'message': f'Verification exception: {str(e)}'}\n"
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