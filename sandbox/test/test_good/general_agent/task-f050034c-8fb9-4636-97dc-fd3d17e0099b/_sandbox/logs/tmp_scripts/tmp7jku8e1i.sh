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
    solution_src = "def solve(tools):\n    # Step 1: Find 5-star Paris hotels with pool and jacuzzi\n    hotels = tools['search_hotel_directory']('Paris')\n    luxury_hotels = []\n    for hotel in hotels:\n        if isinstance(hotel, dict):\n            amenities = hotel.get('amenities', '').lower()\n            stars = hotel.get('stars', '')\n            if 'pool' in amenities and 'jacuzzi' in amenities and stars == '5':\n                luxury_hotels.append({\n                    'name': hotel.get('name', ''),\n                    'stars': stars,\n                    'amenities': hotel.get('amenities', ''),\n                    'directory_price': hotel.get('price', '')\n                })\n    \n    # Step 2: Get seasonal pricing for these hotels\n    pricing_data = tools['search_hotel_pricing']('Paris')\n    for hotel in luxury_hotels:\n        seasonal_rate = 'Not available'\n        special_offer = 'None'\n        for price_rec in pricing_data:\n            if isinstance(price_rec, dict):\n                if price_rec.get('hotel_name', '') == hotel['name']:\n                    rate = price_rec.get('rate', '')\n                    if rate:\n                        seasonal_rate = str(rate)\n                    offer = price_rec.get('special_offer', '')\n                    if offer and offer.lower() != 'none':\n                        special_offer = offer\n                    break\n        hotel['seasonal_rate'] = seasonal_rate\n        hotel['special_offer'] = special_offer\n        \n        # Calculate 3-night cost\n        try:\n            rate_val = float(seasonal_rate.replace('$', '').replace(',', '')) if seasonal_rate != 'Not available' else 0\n            hotel['3_night_cost'] = f'${rate_val * 3:.2f}'\n        except:\n            hotel['3_night_cost'] = 'Not available'\n    \n    # Step 3: Find top attractions in Paris\n    attractions = tools['search_tripadvisor_attractions']('Paris')\n    top_attractions = []\n    for attr in attractions:\n        if isinstance(attr, dict):\n            rating = attr.get('rating', '')\n            entry_fee = attr.get('entry_fee', '')\n            try:\n                rating_val = float(rating)\n                if rating_val >= 4.5 and entry_fee and 'free' not in entry_fee.lower():\n                    top_attractions.append({\n                        'name': attr.get('sub_category', ''),\n                        'description': attr.get('description', ''),\n                        'rating': rating,\n                        'entry_fee': entry_fee\n                    })\n            except:\n                continue\n    \n    # Calculate attraction costs for 2 people\n    for attr in top_attractions:\n        fee_str = attr['entry_fee']\n        try:\n            # Extract first number from fee string\n            import re\n            numbers = re.findall(r'\\d+\\.?\\d*', fee_str)\n            if numbers:\n                fee_val = float(numbers[0])\n                attr['cost_for_2'] = f'${fee_val * 2:.2f}'\n            else:\n                attr['cost_for_2'] = 'Not available'\n        except:\n            attr['cost_for_2'] = 'Not available'\n    \n    # Step 4: Analyze travel recommendations for Paris hotel trips\n    trips = tools['search_travel_recommendations']('Paris')\n    paris_hotel_trips = []\n    total_accommodation_cost = 0\n    total_nights = 0\n    \n    for trip in trips:\n        if isinstance(trip, dict):\n            accommodation = trip.get('accommodation_type', '')\n            if accommodation.lower() == 'hotel':\n                cost_str = trip.get('accommodation_cost', '')\n                duration_str = trip.get('duration_days', '')\n                try:\n                    # Clean cost string\n                    cost_clean = ''.join(c for c in cost_str if c.isdigit() or c == '.')\n                    if cost_clean:\n                        cost = float(cost_clean)\n                        duration = int(duration_str) if duration_str.isdigit() else 1\n                        total_accommodation_cost += cost\n                        total_nights += duration\n                        paris_hotel_trips.append(trip)\n                except:\n                    continue\n    \n    # Calculate average nightly cost\n    avg_nightly_cost = total_accommodation_cost / total_nights if total_nights > 0 else 0\n    \n    # Step 5: Cross-reference hotels with travel data\n    booked_hotel_names = set()\n    for trip in paris_hotel_trips:\n        # Extract hotel name from trip data (simplified - in real scenario would need hotel name field)\n        # For this exercise, we'll check if any hotel names appear in destination or other fields\n        destination = trip.get('destination', '').lower()\n        for hotel in luxury_hotels:\n            if hotel['name'].lower() in destination:\n                booked_hotel_names.add(hotel['name'])\n    \n    # Mark which hotels have been booked\n    for hotel in luxury_hotels:\n        hotel['booked_in_past_year'] = hotel['name'] in booked_hotel_names\n    \n    # Step 6: Create recommended package\n    # Select first available hotel with seasonal rate\n    selected_hotel = None\n    for hotel in luxury_hotels:\n        if hotel['seasonal_rate'] != 'Not available':\n            selected_hotel = hotel\n            break\n    \n    if not selected_hotel and luxury_hotels:\n        selected_hotel = luxury_hotels[0]\n    \n    # Select top 3 attractions\n    selected_attractions = top_attractions[:3] if len(top_attractions) >= 3 else top_attractions\n    \n    # Calculate total costs\n    hotel_cost = 0\n    try:\n        if selected_hotel and selected_hotel['3_night_cost'] != 'Not available':\n            hotel_cost = float(selected_hotel['3_night_cost'].replace('$', '').replace(',', ''))\n    except:\n        hotel_cost = 0\n    \n    attractions_cost = 0\n    for attr in selected_attractions:\n        try:\n            if attr['cost_for_2'] != 'Not available':\n                attractions_cost += float(attr['cost_for_2'].replace('$', '').replace(',', ''))\n        except:\n            continue\n    \n    total_cost = hotel_cost + attractions_cost\n    \n    # Prepare final answer\n    answer = {\n        'analysis_summary': f'Analysis of {len(luxury_hotels)} luxury hotels and {len(top_attractions)} top attractions in Paris',\n        'available_luxury_hotels': [{\n            'name': h['name'],\n            'stars': h['stars'],\n            'amenities': h['amenities'],\n            'seasonal_rate': h['seasonal_rate'],\n            'special_offer': h['special_offer'],\n            '3_night_cost': h['3_night_cost'],\n            'booked_in_past_year': h['booked_in_past_year']\n        } for h in luxury_hotels],\n        'top_attractions': [{\n            'name': a['name'],\n            'description': a['description'],\n            'rating': a['rating'],\n            'entry_fee': a['entry_fee'],\n            'cost_for_2': a['cost_for_2']\n        } for a in top_attractions],\n        'historical_data': {\n            'average_hotel_nightly_cost': f'${avg_nightly_cost:.2f}',\n            'total_paris_hotel_trips': len(paris_hotel_trips),\n            'matching_hotels_booked': len(booked_hotel_names)\n        },\n        'recommended_package': {\n            'package_name': 'Executive Paris Luxury Retreat',\n            'selected_hotel': selected_hotel['name'] if selected_hotel else 'None available',\n            'selected_attractions': [a['name'] for a in selected_attractions],\n            'cost_breakdown': {\n                'hotel_3_nights': selected_hotel['3_night_cost'] if selected_hotel else '$0',\n                'attractions_2_people': f'${attractions_cost:.2f}',\n                'total_estimated': f'${total_cost:.2f}'\n            }\n        }\n    }\n    \n    return tools['submit_result'](answer)\n"
    verification_src = "def verify(tools, answer):\n    try:\n        import json\n        import re\n        \n        if answer is None:\n            return {'passed': False, 'message': 'Answer is None'}\n        \n        data = answer\n        if isinstance(answer, dict):\n            if 'status' in answer and 'submitted_data' in answer:\n                data = answer['submitted_data']\n            elif 'status' in answer and 'data' in answer:\n                data = answer['data']\n        \n        if not isinstance(data, dict):\n            return {'passed': False, 'message': 'Answer data is not a dict'}\n        \n        # Check required top-level keys\n        required_keys = ['analysis_summary', 'available_luxury_hotels', 'top_attractions', \n                        'historical_data', 'recommended_package']\n        for key in required_keys:\n            if key not in data:\n                return {'passed': False, 'message': f'Missing key: {key}'}\n        \n        # Verify hotels structure\n        if not isinstance(data['available_luxury_hotels'], list):\n            return {'passed': False, 'message': 'Hotels not a list'}\n        \n        for hotel in data['available_luxury_hotels']:\n            if not isinstance(hotel, dict):\n                return {'passed': False, 'message': 'Hotel entry not a dict'}\n            \n            hotel_keys = ['name', 'stars', 'amenities', 'seasonal_rate', \n                         'special_offer', '3_night_cost', 'booked_in_past_year']\n            for key in hotel_keys:\n                if key not in hotel:\n                    return {'passed': False, 'message': f'Hotel missing key: {key}'}\n            \n            # Verify 5-star rating\n            if hotel['stars'] != '5':\n                return {'passed': False, 'message': f'Hotel {hotel[\"name\"]} not 5-star'}\n            \n            # Verify amenities contain pool and jacuzzi\n            amenities = hotel.get('amenities', '').lower()\n            if 'pool' not in amenities or 'jacuzzi' not in amenities:\n                return {'passed': False, 'message': f'Hotel {hotel[\"name\"]} missing required amenities'}\n        \n        # Verify attractions structure\n        if not isinstance(data['top_attractions'], list):\n            return {'passed': False, 'message': 'Attractions not a list'}\n        \n        for attr in data['top_attractions']:\n            if not isinstance(attr, dict):\n                return {'passed': False, 'message': 'Attraction entry not a dict'}\n            \n            attr_keys = ['name', 'description', 'rating', 'entry_fee', 'cost_for_2']\n            for key in attr_keys:\n                if key not in attr:\n                    return {'passed': False, 'message': f'Attraction missing key: {key}'}\n            \n            # Verify rating >= 4.5\n            try:\n                rating = float(attr['rating'])\n                if rating < 4.5:\n                    return {'passed': False, 'message': f'Attraction rating {rating} < 4.5'}\n            except:\n                return {'passed': False, 'message': f'Invalid rating: {attr[\"rating\"]}'}\n            \n            # Verify entry fee exists and not free\n            entry_fee = attr['entry_fee'].lower()\n            if not entry_fee or 'free' in entry_fee:\n                return {'passed': False, 'message': f'Attraction has free or no entry fee'}\n        \n        # Verify historical data structure\n        hist_keys = ['average_hotel_nightly_cost', 'total_paris_hotel_trips', 'matching_hotels_booked']\n        for key in hist_keys:\n            if key not in data['historical_data']:\n                return {'passed': False, 'message': f'Historical data missing key: {key}'}\n        \n        # Verify package structure\n        pkg_keys = ['package_name', 'selected_hotel', 'selected_attractions', 'cost_breakdown']\n        for key in pkg_keys:\n            if key not in data['recommended_package']:\n                return {'passed': False, 'message': f'Package missing key: {key}'}\n        \n        cost_keys = ['hotel_3_nights', 'attractions_2_people', 'total_estimated']\n        for key in cost_keys:\n            if key not in data['recommended_package']['cost_breakdown']:\n                return {'passed': False, 'message': f'Cost breakdown missing key: {key}'}\n        \n        # Verify cost formatting\n        for cost_key in cost_keys:\n            cost_val = data['recommended_package']['cost_breakdown'][cost_key]\n            if cost_val and not (cost_val.startswith('$') or cost_val == 'Not available'):\n                return {'passed': False, 'message': f'Cost {cost_key} not properly formatted: {cost_val}'}\n        \n        # Verify data integration by checking tools were used\n        hotels_dir = tools['search_hotel_directory']('Paris')\n        if not hotels_dir:\n            return {'passed': False, 'message': 'No hotel directory data found'}\n        \n        pricing = tools['search_hotel_pricing']('Paris')\n        attractions = tools['search_tripadvisor_attractions']('Paris')\n        trips = tools['search_travel_recommendations']('Paris')\n        \n        # Basic verification that multiple data sources were accessed\n        if not (hotels_dir and attractions):\n            return {'passed': False, 'message': 'Insufficient data sources accessed'}\n        \n        return {\n            'passed': True, \n            'message': 'Verification passed', \n            'details': {\n                'hotels_count': len(data['available_luxury_hotels']),\n                'attractions_count': len(data['top_attractions']),\n                'historical_trips': data['historical_data']['total_paris_hotel_trips']\n            }\n        }\n        \n    except Exception as e:\n        return {'passed': False, 'message': f'Verification exception: {str(e)}'}\n"
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