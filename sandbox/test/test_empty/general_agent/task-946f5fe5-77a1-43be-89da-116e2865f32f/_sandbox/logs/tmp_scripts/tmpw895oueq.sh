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
    solution_src = "def solve(tools):\n    # Step 1: Find travelers aged 30-45 who visited Paris in 2023\n    trip_query = 'Paris 2023'\n    trips = tools['search_trip_details'](trip_query)\n    \n    travelers = []\n    if isinstance(trips, list):\n        for trip in trips:\n            if isinstance(trip, dict):\n                try:\n                    age = int(trip.get('traveler_age', '0'))\n                    destination = trip.get('destination', '')\n                    year = trip.get('start_date', '').split('/')[-1] if '/' in trip.get('start_date', '') else ''\n                    \n                    if 30 <= age <= 45 and 'paris' in destination.lower() and '2023' in year:\n                        # Calculate total trip cost\n                        acc_cost_str = trip.get('accommodation_cost', '0').replace('$', '').replace(',', '').strip()\n                        trans_cost_str = trip.get('transportation_cost', '0').replace('$', '').replace(',', '').strip()\n                        \n                        try:\n                            acc_cost = float(acc_cost_str) if acc_cost_str else 0.0\n                        except:\n                            acc_cost = 0.0\n                        \n                        try:\n                            trans_cost = float(trans_cost_str) if trans_cost_str else 0.0\n                        except:\n                            trans_cost = 0.0\n                        \n                        total_cost = acc_cost + trans_cost\n                        \n                        travelers.append({\n                            'name': trip.get('traveler_name', ''),\n                            'age': str(age),\n                            'nationality': trip.get('traveler_nationality', ''),\n                            'accommodation_type': trip.get('accommodation_type', ''),\n                            'total_cost': total_cost\n                        })\n                except:\n                    continue\n    \n    analysis_list = []\n    highest_overall_ratio = -float('inf')\n    highest_traveler = ''\n    \n    for traveler in travelers:\n        traveler_name = traveler['name']\n        acc_type = traveler['accommodation_type']\n        total_cost = traveler['total_cost']\n        \n        # Step 2: Find matching hotels in Paris based on accommodation type\n        hotels_query = f'Paris {acc_type}'\n        hotels = tools['search_hotel_listings'](hotels_query)\n        \n        matching_hotels = []\n        if isinstance(hotels, list):\n            for hotel in hotels:\n                if isinstance(hotel, dict):\n                    city = hotel.get('city', '')\n                    country = hotel.get('country', '')\n                    if 'paris' in city.lower() and 'france' in country.lower():\n                        hotel_name = hotel.get('name', '')\n                        stars = hotel.get('stars', '')\n                        \n                        # Step 3: Find most expensive deluxe/executive rate\n                        pricing_query = f'{hotel_name} Deluxe Executive'\n                        pricing = tools['search_hotel_pricing'](pricing_query)\n                        \n                        max_rate = 'N/A'\n                        max_rate_num = 0.0\n                        if isinstance(pricing, list):\n                            for price in pricing:\n                                if isinstance(price, dict):\n                                    room_type = price.get('room_type', '')\n                                    rate_str = price.get('rate', '')\n                                    hotel_name_price = price.get('hotel_name', '')\n                                    \n                                    if hotel_name.lower() in hotel_name_price.lower():\n                                        if 'deluxe' in room_type.lower() or 'executive' in room_type.lower():\n                                            try:\n                                                rate_num = float(rate_str.replace('$', '').replace(',', '').strip())\n                                                if rate_num > max_rate_num:\n                                                    max_rate_num = rate_num\n                                                    max_rate = rate_str\n                                            except:\n                                                pass\n                        \n                        # Step 4: Find highest review rating for this hotel\n                        reviews_query = f'{hotel_name} Paris'\n                        reviews = tools['search_travel_reviews'](reviews_query)\n                        \n                        highest_rating = 0.0\n                        if isinstance(reviews, list):\n                            for review in reviews:\n                                if isinstance(review, dict):\n                                    provider = review.get('provider_name', '')\n                                    dest = review.get('destination_location', '')\n                                    rating_str = review.get('rating', '')\n                                    \n                                    if hotel_name.lower() in provider.lower() or hotel_name.lower() in dest.lower():\n                                        try:\n                                            rating = float(rating_str)\n                                            if rating > highest_rating:\n                                                highest_rating = rating\n                                        except:\n                                            pass\n                        \n                        matching_hotels.append({\n                            'hotel_name': hotel_name,\n                            'stars': stars,\n                            'most_expensive_deluxe_rate': max_rate,\n                            'highest_review_rating': round(highest_rating, 2)\n                        })\n        \n        # Step 5: Calculate cost-to-quality ratio for each hotel and find best pair\n        best_pair = {'hotel_name': '', 'cost_to_quality_ratio': 0.0}\n        \n        for hotel in matching_hotels:\n            stars_str = hotel['stars']\n            highest_rating = hotel['highest_review_rating']\n            \n            try:\n                stars_float = float(stars_str)\n                if total_cost > 0 and highest_rating > 0:\n                    ratio = (stars_float * highest_rating) / total_cost\n                    if ratio > best_pair['cost_to_quality_ratio']:\n                        best_pair = {\n                            'hotel_name': hotel['hotel_name'],\n                            'cost_to_quality_ratio': round(ratio, 4)\n                        }\n            except:\n                continue\n        \n        # Track highest overall ratio\n        if best_pair['cost_to_quality_ratio'] > highest_overall_ratio:\n            highest_overall_ratio = best_pair['cost_to_quality_ratio']\n            highest_traveler = traveler_name\n        \n        analysis_entry = {\n            'traveler_name': traveler_name,\n            'traveler_age': traveler['age'],\n            'traveler_nationality': traveler['nationality'],\n            'accommodation_type': acc_type,\n            'total_trip_cost': round(total_cost, 2),\n            'matching_hotels': matching_hotels,\n            'best_hotel_pair': best_pair\n        }\n        \n        analysis_list.append(analysis_entry)\n    \n    # Prepare final answer\n    answer = {\n        'analysis': analysis_list,\n        'traveler_with_highest_overall_ratio': highest_traveler if highest_overall_ratio > 0 else '',\n        'total_travelers_analyzed': str(len(analysis_list))\n    }\n    \n    return tools['submit_result'](answer)\n"
    verification_src = "def verify(tools, answer):\n    import json\n    try:\n        # Handle wrapped answer\n        if isinstance(answer, dict):\n            if 'submitted_data' in answer:\n                data = answer.get('submitted_data')\n            elif 'data' in answer:\n                data = answer.get('data')\n            else:\n                data = answer\n        else:\n            return {'passed': False, 'message': 'Answer is not a dict'}\n        \n        if data is None:\n            return {'passed': False, 'message': 'Answer data is None'}\n        \n        # Check required structure\n        required_keys = ['analysis', 'traveler_with_highest_overall_ratio', 'total_travelers_analyzed']\n        for key in required_keys:\n            if key not in data:\n                return {'passed': False, 'message': f'Missing required key: {key}'}\n        \n        analysis = data.get('analysis')\n        if not isinstance(analysis, list):\n            return {'passed': False, 'message': 'Analysis must be a list'}\n        \n        total_travelers = data.get('total_travelers_analyzed')\n        if not isinstance(total_travelers, str):\n            return {'passed': False, 'message': 'total_travelers_analyzed must be string'}\n        \n        # Verify total_travelers matches analysis length\n        if int(total_travelers) != len(analysis):\n            return {'passed': False, 'message': f'total_travelers_analyzed ({total_travelers}) does not match analysis length ({len(analysis)})'}\n        \n        # Use tools to cross-check data\n        trips = tools['search_trip_details']('Paris 2023')\n        if isinstance(trips, list):\n            paris_travelers_count = 0\n            for trip in trips:\n                if isinstance(trip, dict):\n                    try:\n                        age = int(trip.get('traveler_age', '0'))\n                        destination = trip.get('destination', '')\n                        year = trip.get('start_date', '').split('/')[-1] if '/' in trip.get('start_date', '') else ''\n                        \n                        if 30 <= age <= 45 and 'paris' in destination.lower() and '2023' in year:\n                            paris_travelers_count += 1\n                    except:\n                        continue\n            \n            # Check if analysis is empty when travelers exist\n            if paris_travelers_count > 0 and len(analysis) == 0:\n                return {'passed': False, 'message': f'Found {paris_travelers_count} travelers but analysis is empty'}\n            \n            # Check if analysis has entries when no travelers found\n            if paris_travelers_count == 0 and len(analysis) > 0:\n                return {'passed': False, 'message': 'Analysis has entries but no travelers found matching criteria'}\n        \n        # Validate each analysis entry structure\n        for i, entry in enumerate(analysis):\n            if not isinstance(entry, dict):\n                return {'passed': False, 'message': f'Analysis entry {i} is not a dict'}\n            \n            entry_keys = ['traveler_name', 'traveler_age', 'traveler_nationality', 'accommodation_type', \n                         'total_trip_cost', 'matching_hotels', 'best_hotel_pair']\n            for key in entry_keys:\n                if key not in entry:\n                    return {'passed': False, 'message': f'Analysis entry {i} missing key: {key}'}\n            \n            # Check matching_hotels structure\n            hotels = entry.get('matching_hotels')\n            if not isinstance(hotels, list):\n                return {'passed': False, 'message': f'Analysis entry {i} matching_hotels must be a list'}\n            \n            for j, hotel in enumerate(hotels):\n                if not isinstance(hotel, dict):\n                    return {'passed': False, 'message': f'Analysis entry {i} hotel {j} is not a dict'}\n                \n                hotel_keys = ['hotel_name', 'stars', 'most_expensive_deluxe_rate', 'highest_review_rating']\n                for key in hotel_keys:\n                    if key not in hotel:\n                        return {'passed': False, 'message': f'Analysis entry {i} hotel {j} missing key: {key}'}\n                \n                # Check data types\n                if not isinstance(hotel.get('highest_review_rating'), (int, float)):\n                    return {'passed': False, 'message': f'Analysis entry {i} hotel {j} highest_review_rating must be numeric'}\n            \n            # Check best_hotel_pair structure\n            best_pair = entry.get('best_hotel_pair')\n            if not isinstance(best_pair, dict):\n                return {'passed': False, 'message': f'Analysis entry {i} best_hotel_pair must be a dict'}\n            \n            pair_keys = ['hotel_name', 'cost_to_quality_ratio']\n            for key in pair_keys:\n                if key not in best_pair:\n                    return {'passed': False, 'message': f'Analysis entry {i} best_hotel_pair missing key: {key}'}\n            \n            if not isinstance(best_pair.get('cost_to_quality_ratio'), (int, float)):\n                return {'passed': False, 'message': f'Analysis entry {i} best_hotel_pair cost_to_quality_ratio must be numeric'}\n            \n            # Check total_trip_cost type\n            if not isinstance(entry.get('total_trip_cost'), (int, float)):\n                return {'passed': False, 'message': f'Analysis entry {i} total_trip_cost must be numeric'}\n        \n        # Check traveler_with_highest_overall_ratio\n        highest_traveler = data.get('traveler_with_highest_overall_ratio', '')\n        if not isinstance(highest_traveler, str):\n            return {'passed': False, 'message': 'traveler_with_highest_overall_ratio must be string'}\n        \n        # If highest traveler is specified, verify it exists in analysis\n        if highest_traveler:\n            found = False\n            for entry in analysis:\n                if entry.get('traveler_name') == highest_traveler:\n                    found = True\n                    # Verify best_hotel_pair has positive ratio\n                    if entry.get('best_hotel_pair', {}).get('cost_to_quality_ratio', 0) <= 0:\n                        return {'passed': False, 'message': f'Highest traveler {highest_traveler} does not have positive cost-to-quality ratio'}\n                    break\n            if not found:\n                return {'passed': False, 'message': f'Highest traveler {highest_traveler} not found in analysis'}\n        \n        return {'passed': True, 'message': 'Verification passed'}\n    \n    except Exception as e:\n        return {'passed': False, 'message': f'Verification exception: {str(e)}'}\n"
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