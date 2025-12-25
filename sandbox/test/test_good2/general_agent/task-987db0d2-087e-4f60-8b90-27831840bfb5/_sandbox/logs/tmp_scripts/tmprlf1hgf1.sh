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
    solution_src = "def solve(tools):\n    # Step 1: Search for American female travelers to Paris\n    travel_data = tools['search_travel_recommendations']('Paris')\n    \n    # Filter for American female travelers aged 25-35\n    american_female_travelers = []\n    total_accommodation_cost = 0\n    total_nights = 0\n    accommodation_types = {}\n    \n    if isinstance(travel_data, list):\n        for trip in travel_data:\n            if isinstance(trip, dict) and 'error' not in trip:\n                destination = trip.get('destination', '').lower()\n                nationality = trip.get('traveler_nationality', '').lower()\n                gender = trip.get('traveler_gender', '').lower()\n                age_str = trip.get('traveler_age', '')\n                \n                # Check if trip is to Paris and traveler is American female\n                if ('paris' in destination and \n                    'american' in nationality and \n                    'female' in gender):\n                    \n                    try:\n                        age = int(age_str)\n                        if 25 <= age <= 35:\n                            american_female_travelers.append(trip)\n                            \n                            # Process accommodation cost\n                            acc_cost_str = trip.get('accommodation_cost', '0')\n                            # Clean cost string\n                            acc_cost_clean = acc_cost_str.replace('$', '').replace(',', '').replace(' ', '')\n                            try:\n                                acc_cost = float(acc_cost_clean)\n                                total_accommodation_cost += acc_cost\n                            except (ValueError, TypeError):\n                                pass\n                            \n                            # Process duration\n                            duration_str = trip.get('duration_days', '0')\n                            try:\n                                duration = int(duration_str)\n                                total_nights += duration\n                            except (ValueError, TypeError):\n                                pass\n                            \n                            # Track accommodation types\n                            acc_type = trip.get('accommodation_type', '')\n                            if acc_type:\n                                accommodation_types[acc_type] = accommodation_types.get(acc_type, 0) + 1\n                    except (ValueError, TypeError):\n                        continue\n    \n    # Calculate average cost per night\n    avg_cost_per_night = 0.0\n    if total_nights > 0 and total_accommodation_cost > 0:\n        avg_cost_per_night = round(total_accommodation_cost / total_nights, 2)\n    \n    # Find most common accommodation type\n    most_common_type = 'Not available'\n    if accommodation_types:\n        most_common_type = max(accommodation_types.items(), key=lambda x: x[1])[0]\n    \n    # Step 2: Search for Paris hotels with high ratings and special offers\n    hotel_data = tools['search_hotel_pricing']('Paris')\n    \n    top_hotels = []\n    if isinstance(hotel_data, list):\n        for hotel in hotel_data:\n            if isinstance(hotel, dict) and 'error' not in hotel:\n                availability = hotel.get('availability', '')\n                special_offer = hotel.get('special_offer', '')\n                guest_rating = hotel.get('guest_rating', 0)\n                \n                # Check criteria: Available, has special offer, high rating\n                if (availability == 'Available' and \n                    special_offer and \n                    special_offer.lower() != 'none' and \n                    isinstance(guest_rating, (int, float)) and \n                    guest_rating >= 4.5):\n                    \n                    # Convert rate to string\n                    rate = hotel.get('rate', '')\n                    if not isinstance(rate, str):\n                        rate = str(rate)\n                    \n                    top_hotels.append({\n                        'hotel_name': hotel.get('hotel_name', ''),\n                        'room_type': hotel.get('room_type', ''),\n                        'rate': rate,\n                        'guest_rating': float(guest_rating),\n                        'special_offer': special_offer\n                    })\n    \n    # Sort hotels by rating (highest first)\n    top_hotels.sort(key=lambda x: x['guest_rating'], reverse=True)\n    \n    # Prepare final answer\n    answer = {\n        'average_accommodation_cost_per_night': str(avg_cost_per_night),\n        'top_rated_hotels_with_offers': top_hotels,\n        'most_common_accommodation_type': most_common_type,\n        'data_summary': {\n            'total_american_female_travelers_analyzed': str(len(american_female_travelers)),\n            'hotels_with_high_ratings_found': str(len(top_hotels))\n        }\n    }\n    \n    return tools['submit_result'](str(answer))\n"
    verification_src = "def verify(tools, answer):\n    import json\n    import ast\n    \n    try:\n        # Parse the answer string back to dictionary\n        if isinstance(answer, str):\n            try:\n                answer_dict = ast.literal_eval(answer)\n            except:\n                # Try JSON parsing\n                answer_dict = json.loads(answer)\n        else:\n            answer_dict = answer\n        \n        if not isinstance(answer_dict, dict):\n            return {'passed': False, 'message': 'Answer is not a dictionary'}\n        \n        # Check required top-level keys\n        required_keys = ['average_accommodation_cost_per_night', \n                        'top_rated_hotels_with_offers', \n                        'most_common_accommodation_type',\n                        'data_summary']\n        \n        for key in required_keys:\n            if key not in answer_dict:\n                return {'passed': False, 'message': f'Missing required key: {key}'}\n        \n        # Verify average cost format\n        avg_cost = answer_dict.get('average_accommodation_cost_per_night')\n        if not isinstance(avg_cost, str):\n            return {'passed': False, 'message': 'average_accommodation_cost_per_night must be string'}\n        \n        try:\n            avg_cost_float = float(avg_cost)\n            # Check if rounded to 2 decimals\n            if len(avg_cost.split('.')[-1]) > 2:\n                return {'passed': False, 'message': 'Average cost should be rounded to 2 decimals'}\n        except ValueError:\n            return {'passed': False, 'message': 'Average cost is not a valid float string'}\n        \n        # Verify hotels list\n        hotels_list = answer_dict.get('top_rated_hotels_with_offers', [])\n        if not isinstance(hotels_list, list):\n            return {'passed': False, 'message': 'top_rated_hotels_with_offers must be a list'}\n        \n        # Verify each hotel entry\n        required_hotel_keys = ['hotel_name', 'room_type', 'rate', 'guest_rating', 'special_offer']\n        for i, hotel in enumerate(hotels_list):\n            if not isinstance(hotel, dict):\n                return {'passed': False, 'message': f'Hotel entry {i} is not a dictionary'}\n            \n            for key in required_hotel_keys:\n                if key not in hotel:\n                    return {'passed': False, 'message': f'Hotel entry {i} missing key: {key}'}\n                \n                value = hotel.get(key)\n                if key == 'guest_rating':\n                    if not isinstance(value, (int, float)):\n                        return {'passed': False, 'message': f'Hotel entry {i} guest_rating must be numeric'}\n                    if value < 4.5:\n                        return {'passed': False, 'message': f'Hotel entry {i} guest_rating must be \u22654.5'}\n                elif key == 'special_offer':\n                    if not isinstance(value, str):\n                        return {'passed': False, 'message': f'Hotel entry {i} special_offer must be string'}\n                    if value.lower() == 'none' or not value.strip():\n                        return {'passed': False, 'message': f'Hotel entry {i} has invalid special_offer'}\n                else:\n                    if not isinstance(value, str):\n                        return {'passed': False, 'message': f'Hotel entry {i} key {key} must be string'}\n                    if not value.strip():\n                        return {'passed': False, 'message': f'Hotel entry {i} key {key} is empty'}\n        \n        # Verify most common accommodation type\n        acc_type = answer_dict.get('most_common_accommodation_type')\n        if not isinstance(acc_type, str):\n            return {'passed': False, 'message': 'most_common_accommodation_type must be string'}\n        \n        # Verify data summary\n        data_summary = answer_dict.get('data_summary', {})\n        if not isinstance(data_summary, dict):\n            return {'passed': False, 'message': 'data_summary must be a dictionary'}\n        \n        summary_keys = ['total_american_female_travelers_analyzed', 'hotels_with_high_ratings_found']\n        for key in summary_keys:\n            if key not in data_summary:\n                return {'passed': False, 'message': f'data_summary missing key: {key}'}\n            \n            value = data_summary.get(key)\n            if not isinstance(value, str):\n                return {'passed': False, 'message': f'data_summary key {key} must be string'}\n            \n            try:\n                int_value = int(value)\n                if int_value < 0:\n                    return {'passed': False, 'message': f'data_summary key {key} must be non-negative'}\n            except ValueError:\n                return {'passed': False, 'message': f'data_summary key {key} must be integer string'}\n        \n        # Cross-verify with actual data\n        # Check travel recommendations for American female travelers\n        travel_check = tools['search_travel_recommendations']('American female Paris')\n        if isinstance(travel_check, list) and len(travel_check) > 0:\n            # At least some data should exist\n            pass\n        \n        # Check hotel pricing for high-rated hotels\n        hotel_check = tools['search_hotel_pricing']('Available')\n        if isinstance(hotel_check, list) and len(hotel_check) > 0:\n            # Verify at least one hotel meets criteria\n            high_rated_found = False\n            for hotel in hotel_check:\n                if isinstance(hotel, dict) and 'error' not in hotel:\n                    rating = hotel.get('guest_rating', 0)\n                    offer = hotel.get('special_offer', '')\n                    avail = hotel.get('availability', '')\n                    \n                    if (isinstance(rating, (int, float)) and \n                        rating >= 4.5 and \n                        offer and offer.lower() != 'none' and \n                        avail == 'Available'):\n                        high_rated_found = True\n                        break\n            \n            if not high_rated_found and len(hotels_list) > 0:\n                return {'passed': False, 'message': 'Hotels listed but none found in verification'}\n        \n        return {'passed': True, 'message': 'All verification checks passed'}\n        \n    except Exception as e:\n        return {'passed': False, 'message': f'Verification exception: {str(e)}'}\n"
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