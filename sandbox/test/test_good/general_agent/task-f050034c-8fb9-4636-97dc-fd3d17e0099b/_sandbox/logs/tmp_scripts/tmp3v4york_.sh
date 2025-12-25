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
    solution_src = "def solve(tools):\n    # Step 1: Find 5-star Paris hotels with pool and jacuzzi\n    hotels_dir = tools['search_hotel_directory']('Paris')\n    luxury_hotels = []\n    hotel_names = []\n    for hotel in hotels_dir:\n        if isinstance(hotel, dict):\n            amenities = hotel.get('amenities', '').lower()\n            stars = hotel.get('stars', '')\n            if 'pool' in amenities and 'jacuzzi' in amenities and stars == '5':\n                hotel_name = hotel.get('name', '')\n                luxury_hotels.append({\n                    'name': hotel_name,\n                    'stars': stars,\n                    'amenities': hotel.get('amenities', ''),\n                    'directory_price': hotel.get('price', '')\n                })\n                hotel_names.append(hotel_name)\n    \n    # Step 2: Get seasonal pricing for these hotels\n    pricing_data = []\n    for name in hotel_names:\n        pricing = tools['search_hotel_pricing'](name)\n        if pricing and not (isinstance(pricing, list) and len(pricing) == 1 and 'error' in pricing[0]):\n            pricing_data.extend(pricing)\n    if not pricing_data:\n        pricing_data = tools['search_hotel_pricing']('sample')\n    \n    # Step 3: Find TripAdvisor attractions in Paris with rating >= 4.5 and entry fee\n    attractions_raw = tools['search_tripadvisor_attractions']('Paris')\n    top_attractions = []\n    if attractions_raw:\n        for attr in attractions_raw:\n            if isinstance(attr, dict):\n                rating_str = attr.get('rating', '0')\n                try:\n                    rating = float(rating_str)\n                except:\n                    rating = 0.0\n                entry_fee = attr.get('entry_fee', '')\n                if rating >= 4.5 and entry_fee and entry_fee.lower() != 'free' and 'free' not in entry_fee.lower():\n                    top_attractions.append({\n                        'name': attr.get('sub_category', ''),\n                        'description': attr.get('description', ''),\n                        'rating': rating_str,\n                        'entry_fee': entry_fee\n                    })\n    \n    # Step 4: Analyze travel recommendations for Paris hotel trips\n    travel_recs = tools['search_travel_recommendations']('Paris')\n    paris_hotel_trips = []\n    total_cost = 0\n    total_nights = 0\n    booked_hotel_names = set()\n    \n    if travel_recs:\n        for trip in travel_recs:\n            if isinstance(trip, dict):\n                acc_type = trip.get('accommodation_type', '')\n                dest = trip.get('destination', '')\n                if 'hotel' in acc_type.lower() and ('paris' in dest.lower() or 'france' in dest.lower()):\n                    cost_str = trip.get('accommodation_cost', '0')\n                    duration_str = trip.get('duration_days', '0')\n                    try:\n                        cost = float(cost_str.replace('$', '').replace(',', '').strip())\n                        nights = int(float(duration_str))\n                        if nights > 0:\n                            total_cost += cost\n                            total_nights += nights\n                            paris_hotel_trips.append(trip)\n                            # Try to extract hotel name from traveler name or other fields\n                            traveler = trip.get('traveler_name', '')\n                            if traveler:\n                                booked_hotel_names.add(traveler)\n                    except:\n                        pass\n    \n    # Step 5: Process luxury hotels with pricing and booking info\n    available_luxury = []\n    for hotel in luxury_hotels:\n        seasonal_rate = 'Not available'\n        special_offer = 'None'\n        nightly_rate = 0\n        \n        # Find matching pricing data\n        for price_rec in pricing_data:\n            if isinstance(price_rec, dict):\n                price_hotel_name = price_rec.get('hotel_name', '')\n                if price_hotel_name and hotel['name'].lower() in price_hotel_name.lower():\n                    rate = price_rec.get('rate', '')\n                    if rate:\n                        try:\n                            nightly_rate = float(str(rate).replace('$', '').replace(',', '').strip())\n                            seasonal_rate = f'${nightly_rate:.2f}'\n                        except:\n                            seasonal_rate = str(rate)\n                    offer = price_rec.get('special_offer', '')\n                    if offer and offer.lower() != 'none':\n                        special_offer = offer\n                    break\n        \n        # Calculate 3-night cost\n        if nightly_rate > 0:\n            three_night_cost = f'${nightly_rate * 3:.2f}'\n        else:\n            three_night_cost = '$0.00'\n        \n        # Check if booked in past year\n        booked = False\n        for booked_name in booked_hotel_names:\n            if hotel['name'].lower() in booked_name.lower() or booked_name.lower() in hotel['name'].lower():\n                booked = True\n                break\n        \n        available_luxury.append({\n            'name': hotel['name'],\n            'stars': hotel['stars'],\n            'amenities': hotel['amenities'],\n            'seasonal_rate': seasonal_rate,\n            'special_offer': special_offer,\n            '3_night_cost': three_night_cost,\n            'booked_in_past_year': booked\n        })\n    \n    # Step 6: Calculate historical averages\n    avg_nightly = '$0.00'\n    if total_nights > 0:\n        avg = total_cost / total_nights\n        avg_nightly = f'${avg:.2f}'\n    \n    # Step 7: Prepare top attractions with cost for 2\n    top_attractions_final = []\n    for attr in top_attractions[:3]:  # Take up to 3\n        fee_str = attr['entry_fee']\n        cost_for_2 = 'Not available'\n        # Try to extract numeric value\n        import re\n        numbers = re.findall(r'\\d+\\.?\\d*', fee_str)\n        if numbers:\n            try:\n                fee = float(numbers[0])\n                cost_for_2 = f'${fee * 2:.2f}'\n            except:\n                pass\n        top_attractions_final.append({\n            'name': attr['name'],\n            'description': attr['description'],\n            'rating': attr['rating'],\n            'entry_fee': attr['entry_fee'],\n            'cost_for_2': cost_for_2\n        })\n    \n    # Step 8: Create recommended package\n    selected_hotel = 'George V'\n    if available_luxury:\n        selected_hotel = available_luxury[0]['name']\n    \n    selected_attractions = [attr['name'] for attr in top_attractions_final]\n    \n    # Calculate package costs\n    hotel_3_nights = '$0.00'\n    for hotel in available_luxury:\n        if hotel['name'] == selected_hotel:\n            hotel_3_nights = hotel['3_night_cost']\n            break\n    \n    attractions_cost = '$0.00'\n    total_attr = 0\n    for attr in top_attractions_final:\n        if attr['name'] in selected_attractions:\n            fee_str = attr['entry_fee']\n            numbers = re.findall(r'\\d+\\.?\\d*', fee_str)\n            if numbers:\n                try:\n                    total_attr += float(numbers[0]) * 2\n                except:\n                    pass\n    if total_attr > 0:\n        attractions_cost = f'${total_attr:.2f}'\n    \n    total_estimated = '$0.00'\n    if hotel_3_nights != '$0.00' and attractions_cost != '$0.00':\n        hotel_val = float(hotel_3_nights.replace('$', ''))\n        attr_val = float(attractions_cost.replace('$', ''))\n        total_estimated = f'${hotel_val + attr_val:.2f}'\n    \n    # Step 9: Build final answer\n    answer = {\n        'analysis_summary': f'Analysis of {len(available_luxury)} luxury hotels and {len(top_attractions_final)} top attractions in Paris',\n        'available_luxury_hotels': available_luxury,\n        'top_attractions': top_attractions_final,\n        'historical_data': {\n            'average_hotel_nightly_cost': avg_nightly,\n            'total_paris_hotel_trips': len(paris_hotel_trips),\n            'matching_hotels_booked': sum(1 for h in available_luxury if h['booked_in_past_year'])\n        },\n        'recommended_package': {\n            'package_name': 'Executive Paris Luxury Retreat',\n            'selected_hotel': selected_hotel,\n            'selected_attractions': selected_attractions,\n            'cost_breakdown': {\n                'hotel_3_nights': hotel_3_nights,\n                'attractions_2_people': attractions_cost,\n                'total_estimated': total_estimated\n            }\n        }\n    }\n    \n    return tools['submit_result'](answer)\n"
    verification_src = "def verify(tools, answer):\n    try:\n        import json\n        import re\n        \n        if answer is None:\n            return {'passed': False, 'message': 'Answer is None'}\n        \n        data = answer\n        if isinstance(answer, dict):\n            if 'status' in answer and 'submitted_data' in answer:\n                data = answer['submitted_data']\n            elif 'status' in answer and 'data' in answer:\n                data = answer['data']\n        \n        if not isinstance(data, dict):\n            return {'passed': False, 'message': 'Answer data is not a dict'}\n        \n        # Check required top-level keys\n        required_keys = ['analysis_summary', 'available_luxury_hotels', 'top_attractions', \n                        'historical_data', 'recommended_package']\n        for key in required_keys:\n            if key not in data:\n                return {'passed': False, 'message': f'Missing key: {key}'}\n        \n        # Verify hotels structure\n        if not isinstance(data['available_luxury_hotels'], list):\n            return {'passed': False, 'message': 'Hotels not a list'}\n        \n        hotel_required = ['name', 'stars', 'amenities', 'seasonal_rate', 'special_offer', '3_night_cost', 'booked_in_past_year']\n        for hotel in data['available_luxury_hotels']:\n            if not isinstance(hotel, dict):\n                return {'passed': False, 'message': 'Hotel entry not a dict'}\n            for key in hotel_required:\n                if key not in hotel:\n                    return {'passed': False, 'message': f'Hotel missing key: {key}'}\n            if hotel['stars'] != '5':\n                return {'passed': False, 'message': 'Hotel not 5-star'}\n            amenities = hotel.get('amenities', '').lower()\n            if 'pool' not in amenities or 'jacuzzi' not in amenities:\n                return {'passed': False, 'message': 'Hotel missing required amenities'}\n        \n        # Verify attractions structure\n        if not isinstance(data['top_attractions'], list):\n            return {'passed': False, 'message': 'Attractions not a list'}\n        \n        attr_required = ['name', 'description', 'rating', 'entry_fee', 'cost_for_2']\n        for attr in data['top_attractions']:\n            if not isinstance(attr, dict):\n                return {'passed': False, 'message': 'Attraction entry not a dict'}\n            for key in attr_required:\n                if key not in attr:\n                    return {'passed': False, 'message': f'Attraction missing key: {key}'}\n            try:\n                rating = float(attr['rating'])\n                if rating < 4.5:\n                    return {'passed': False, 'message': 'Attraction rating below 4.5'}\n            except:\n                pass\n            if not attr['entry_fee'] or 'free' in attr['entry_fee'].lower():\n                return {'passed': False, 'message': 'Attraction has no entry fee or is free'}\n        \n        # Verify historical data\n        hist = data['historical_data']\n        if not isinstance(hist, dict):\n            return {'passed': False, 'message': 'Historical data not a dict'}\n        hist_keys = ['average_hotel_nightly_cost', 'total_paris_hotel_trips', 'matching_hotels_booked']\n        for key in hist_keys:\n            if key not in hist:\n                return {'passed': False, 'message': f'Historical missing key: {key}'}\n        \n        # Verify package\n        pkg = data['recommended_package']\n        if not isinstance(pkg, dict):\n            return {'passed': False, 'message': 'Package not a dict'}\n        pkg_keys = ['package_name', 'selected_hotel', 'selected_attractions', 'cost_breakdown']\n        for key in pkg_keys:\n            if key not in pkg:\n                return {'passed': False, 'message': f'Package missing key: {key}'}\n        \n        cost = pkg['cost_breakdown']\n        cost_keys = ['hotel_3_nights', 'attractions_2_people', 'total_estimated']\n        for key in cost_keys:\n            if key not in cost:\n                return {'passed': False, 'message': f'Cost breakdown missing key: {key}'}\n        \n        # Cross-check with actual data sources\n        # Check hotel directory data\n        hotels_check = tools['search_hotel_directory']('sample')\n        if hotels_check and isinstance(hotels_check, list) and len(hotels_check) > 0:\n            found_hotels = False\n            for h in hotels_check:\n                if isinstance(h, dict) and h.get('city', '').lower() == 'paris':\n                    found_hotels = True\n                    break\n            if not found_hotels and len(data['available_luxury_hotels']) > 0:\n                return {'passed': False, 'message': 'No Paris hotels found in directory data'}\n        \n        # Check attractions data\n        attractions_check = tools['search_tripadvisor_attractions']('sample')\n        if attractions_check and isinstance(attractions_check, list) and len(attractions_check) > 0:\n            found_attractions = False\n            for a in attractions_check:\n                if isinstance(a, dict) and a.get('country', '').lower() == 'france':\n                    found_attractions = True\n                    break\n            if not found_attractions and len(data['top_attractions']) > 0:\n                return {'passed': False, 'message': 'No France attractions found in TripAdvisor data'}\n        \n        # Check travel recommendations\n        travel_check = tools['search_travel_recommendations']('sample')\n        if travel_check and isinstance(travel_check, list) and len(travel_check) > 0:\n            found_travel = False\n            for t in travel_check:\n                if isinstance(t, dict) and 'hotel' in t.get('accommodation_type', '').lower():\n                    found_travel = True\n                    break\n            if not found_travel and data['historical_data']['total_paris_hotel_trips'] > 0:\n                return {'passed': False, 'message': 'No hotel trips found in travel data'}\n        \n        # Final validation\n        if len(data['available_luxury_hotels']) == 0:\n            return {'passed': False, 'message': 'No luxury hotels found'}\n        \n        return {'passed': True, 'message': 'All verification checks passed'}\n    \n    except Exception as e:\n        return {'passed': False, 'message': f'Verification error: {str(e)}'}\n"
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