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
missing = [name for name in ["search_hotels", "search_attractions", "search_travel_recommendations", "get_hotel_star_ratings", "submit_result"] if name not in tools or not callable(tools.get(name))]
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
    solution_src = "def solve(tools):\n    # Step 1: Find 4-star Paris hotel with pool and wifi\n    hotels = tools['search_hotels']('Paris', 50)\n    target_hotel = None\n    \n    for hotel in hotels:\n        if isinstance(hotel, dict):\n            stars = str(hotel.get('stars', '')).strip()\n            city = str(hotel.get('city', '')).lower()\n            amenities = str(hotel.get('amenities', '')).lower()\n            \n            if ('4' in stars or 'four' in stars.lower()) and 'paris' in city:\n                if 'pool' in amenities and 'wifi' in amenities:\n                    target_hotel = {\n                        'name': hotel.get('name', ''),\n                        'stars': '4',\n                        'amenities': hotel.get('amenities', ''),\n                        'price': hotel.get('price', ''),\n                        'city': hotel.get('city', '')\n                    }\n                    break\n    \n    # Step 2: Find Airbnb experience in Paris with rating >= 4.5\n    attractions = tools['search_attractions']('Paris', 50)\n    airbnb_exp = None\n    \n    for attr in attractions:\n        if isinstance(attr, dict):\n            location = str(attr.get('location', '')).lower()\n            rating_str = str(attr.get('rating', '0'))\n            \n            # Check if it's an Airbnb experience (has experience_title)\n            if 'paris' in location and 'experience_title' in attr:\n                try:\n                    rating = float(rating_str)\n                    if rating >= 4.5:\n                        airbnb_exp = {\n                            'title': attr.get('experience_title', ''),\n                            'rating': rating_str,\n                            'price': attr.get('price', ''),\n                            'location': attr.get('location', '')\n                        }\n                        break\n                except:\n                    continue\n    \n    # Step 3: Find Paris attraction with entry fee < $30\n    attraction_target = None\n    for attr in attractions:\n        if isinstance(attr, dict):\n            location = str(attr.get('location', '')).lower()\n            entry_fee = str(attr.get('entry_fee', '$100'))\n            \n            if 'paris' in location and 'entry_fee' in attr:\n                # Extract numeric value from entry fee\n                import re\n                fee_match = re.search(r'\\$?([0-9]+\\.?[0-9]*)', entry_fee)\n                if fee_match:\n                    fee_value = float(fee_match.group(1))\n                    if fee_value < 30:\n                        attraction_target = {\n                            'name': attr.get('sub_category', attr.get('experience_title', '')),\n                            'entry_fee': entry_fee,\n                            'rating': attr.get('rating', ''),\n                            'location': attr.get('location', '')\n                        }\n                        break\n    \n    # Step 4: Find historical American travelers to Paris who stayed in hotels\n    travel_data = tools['search_travel_recommendations']('', 100)\n    american_travelers = []\n    \n    for trip in travel_data:\n        if isinstance(trip, dict):\n            nationality = str(trip.get('traveler_nationality', '')).lower()\n            destination = str(trip.get('destination', '')).lower()\n            accommodation = str(trip.get('accommodation_type', '')).lower()\n            \n            if 'american' in nationality and 'paris' in destination and 'hotel' in accommodation:\n                american_travelers.append({\n                    'name': trip.get('traveler_name', ''),\n                    'nationality': trip.get('traveler_nationality', ''),\n                    'destination': trip.get('destination', ''),\n                    'accommodation_type': trip.get('accommodation_type', '')\n                })\n                \n                if len(american_travelers) >= 3:\n                    break\n    \n    # Step 5: Verify all criteria are met\n    verification = []\n    \n    if target_hotel:\n        verification.append(\"\u2713 Found 4-star Paris hotel with pool and wifi\")\n    else:\n        verification.append(\"\u2717 No suitable hotel found\")\n        \n    if airbnb_exp:\n        verification.append(f\"\u2713 Found Airbnb experience with rating {airbnb_exp['rating']} (>= 4.5)\")\n    else:\n        verification.append(\"\u2717 No suitable Airbnb experience found\")\n        \n    if attraction_target:\n        verification.append(f\"\u2713 Found attraction with entry fee {attraction_target['entry_fee']} (< $30)\")\n    else:\n        verification.append(\"\u2717 No suitable attraction found\")\n        \n    if len(american_travelers) >= 3:\n        verification.append(f\"\u2713 Found {len(american_travelers)} American travelers to Paris who stayed in hotels\")\n    else:\n        verification.append(f\"\u2717 Only found {len(american_travelers)} American travelers (need 3)\")\n    \n    # Step 6: Create final package\n    package_name = \"Paris Premium Experience Package\"\n    \n    answer = {\n        'package_name': package_name,\n        'hotel': target_hotel if target_hotel else {\n            'name': 'No suitable hotel found',\n            'stars': '',\n            'amenities': '',\n            'price': '',\n            'city': ''\n        },\n        'airbnb_experience': airbnb_exp if airbnb_exp else {\n            'title': 'No suitable experience found',\n            'rating': '',\n            'price': '',\n            'location': ''\n        },\n        'attraction': attraction_target if attraction_target else {\n            'name': 'No suitable attraction found',\n            'entry_fee': '',\n            'rating': '',\n            'location': ''\n        },\n        'historical_travelers': american_travelers if american_travelers else [],\n        'verification_summary': ' | '.join(verification)\n    }\n    \n    return tools['submit_result'](answer)\n"
    verification_src = "def verify(tools, answer):\n    try:\n        # Handle wrapped answer\n        if answer is None:\n            return {'passed': False, 'message': 'Answer is None'}\n        \n        if isinstance(answer, dict):\n            if 'status' in answer and 'data' in answer:\n                answer = answer['data']\n            elif 'submitted_data' in answer:\n                answer = answer['submitted_data']\n            elif 'data' in answer:\n                answer = answer['data']\n        \n        if not isinstance(answer, dict):\n            return {'passed': False, 'message': 'Answer is not a dict'}\n        \n        # Check required keys\n        required_keys = ['package_name', 'hotel', 'airbnb_experience', 'attraction', 'historical_travelers', 'verification_summary']\n        for key in required_keys:\n            if key not in answer:\n                return {'passed': False, 'message': f'Missing required key: {key}'}\n        \n        # Verify hotel criteria\n        hotel = answer['hotel']\n        if not isinstance(hotel, dict):\n            return {'passed': False, 'message': 'Hotel is not a dict'}\n        \n        if hotel.get('stars') != '4':\n            return {'passed': False, 'message': f'Hotel stars is not 4: {hotel.get(\"stars\")}'}\n        \n        if 'paris' not in str(hotel.get('city', '')).lower():\n            return {'passed': False, 'message': f'Hotel city is not Paris: {hotel.get(\"city\")}'}\n        \n        amenities = str(hotel.get('amenities', '')).lower()\n        if 'pool' not in amenities or 'wifi' not in amenities:\n            return {'passed': False, 'message': f'Hotel missing required amenities (pool and wifi): {amenities}'}\n        \n        # Verify Airbnb experience criteria\n        airbnb = answer['airbnb_experience']\n        if not isinstance(airbnb, dict):\n            return {'passed': False, 'message': 'Airbnb experience is not a dict'}\n        \n        if 'paris' not in str(airbnb.get('location', '')).lower():\n            return {'passed': False, 'message': f'Airbnb location is not Paris: {airbnb.get(\"location\")}'}\n        \n        try:\n            rating = float(str(airbnb.get('rating', '0')))\n            if rating < 4.5:\n                return {'passed': False, 'message': f'Airbnb rating {rating} is less than 4.5'}\n        except:\n            return {'passed': False, 'message': f'Invalid Airbnb rating: {airbnb.get(\"rating\")}'}\n        \n        # Verify attraction criteria\n        attraction = answer['attraction']\n        if not isinstance(attraction, dict):\n            return {'passed': False, 'message': 'Attraction is not a dict'}\n        \n        if 'paris' not in str(attraction.get('location', '')).lower():\n            return {'passed': False, 'message': f'Attraction location is not Paris: {attraction.get(\"location\")}'}\n        \n        entry_fee = str(attraction.get('entry_fee', '$100'))\n        import re\n        fee_match = re.search(r'\\$?([0-9]+\\.?[0-9]*)', entry_fee)\n        if fee_match:\n            fee_value = float(fee_match.group(1))\n            if fee_value >= 30:\n                return {'passed': False, 'message': f'Attraction entry fee ${fee_value} is not under $30'}\n        else:\n            return {'passed': False, 'message': f'Could not parse attraction entry fee: {entry_fee}'}\n        \n        # Verify historical travelers criteria\n        travelers = answer['historical_travelers']\n        if not isinstance(travelers, list):\n            return {'passed': False, 'message': 'Historical travelers is not a list'}\n        \n        if len(travelers) < 3:\n            return {'passed': False, 'message': f'Only {len(travelers)} historical travelers found (need 3)'}\n        \n        for i, traveler in enumerate(travelers):\n            if not isinstance(traveler, dict):\n                return {'passed': False, 'message': f'Traveler {i} is not a dict'}\n            \n            if 'american' not in str(traveler.get('nationality', '')).lower():\n                return {'passed': False, 'message': f'Traveler {i} nationality is not American: {traveler.get(\"nationality\")}'}\n            \n            if 'paris' not in str(traveler.get('destination', '')).lower():\n                return {'passed': False, 'message': f'Traveler {i} destination is not Paris: {traveler.get(\"destination\")}'}\n            \n            if 'hotel' not in str(traveler.get('accommodation_type', '')).lower():\n                return {'passed': False, 'message': f'Traveler {i} accommodation is not hotel: {traveler.get(\"accommodation_type\")}'}\n        \n        # Verify summary exists\n        summary = answer['verification_summary']\n        if not isinstance(summary, str) or not summary.strip():\n            return {'passed': False, 'message': 'Verification summary is missing or empty'}\n        \n        # Cross-check with actual data sources\n        # Verify hotel exists in dataset\n        hotels_data = tools['search_hotels']('Paris', 20)\n        hotel_found = False\n        for h in hotels_data:\n            if isinstance(h, dict) and h.get('name') == hotel.get('name'):\n                hotel_found = True\n                break\n        \n        if not hotel_found and hotel.get('name') != 'No suitable hotel found':\n            return {'passed': False, 'message': f'Hotel {hotel.get(\"name\")} not found in dataset'}\n        \n        return {\n            'passed': True, \n            'message': 'All package criteria verified successfully', \n            'details': {\n                'hotel_verified': hotel_found or hotel.get('name') == 'No suitable hotel found',\n                'airbnb_rating_verified': rating >= 4.5,\n                'attraction_fee_verified': fee_value < 30,\n                'historical_travelers_count': len(travelers)\n            }\n        }\n        \n    except Exception as e:\n        return {'passed': False, 'message': f'Verification error: {str(e)}'}\n"
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