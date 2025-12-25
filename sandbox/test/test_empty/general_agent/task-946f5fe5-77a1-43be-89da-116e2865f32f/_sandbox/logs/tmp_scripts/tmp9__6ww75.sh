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
    solution_src = "def solve(tools):\n    # Step 1: Find Paris hotels with both 'pool' and 'wifi' amenities\n    listings = tools['search_hotel_listings']('Paris, France')\n    paris_hotels = []\n    if isinstance(listings, list):\n        for hotel in listings:\n            if isinstance(hotel, dict):\n                amenities = hotel.get('amenities', '')\n                city = hotel.get('city', '')\n                country = hotel.get('country', '')\n                if amenities and city and country:\n                    amenities_lower = amenities.lower()\n                    if 'pool' in amenities_lower and 'wifi' in amenities_lower:\n                        if 'paris' in city.lower() and 'france' in country.lower():\n                            paris_hotels.append({\n                                'name': str(hotel.get('name', '')),\n                                'stars': str(hotel.get('stars', '')),\n                                'amenities': amenities,\n                                'city': city,\n                                'country': country\n                            })\n    \n    # Sort hotels alphabetically\n    paris_hotels.sort(key=lambda x: x['name'].lower())\n    \n    analysis_list = []\n    \n    for hotel in paris_hotels:\n        hotel_name = hotel['name']\n        stars = hotel['stars']\n        amenities = hotel['amenities']\n        \n        # Step 2: Search travel reviews for this hotel\n        reviews = tools['search_travel_reviews'](hotel_name)\n        all_reviews = []\n        most_recent_review = None\n        \n        if isinstance(reviews, list):\n            for review in reviews:\n                if isinstance(review, dict):\n                    # Check if review mentions hotel name or provider_name\n                    provider = review.get('provider_name', '')\n                    if hotel_name.lower() in provider.lower() or hotel_name.lower() in str(review.get('review_text', '')).lower():\n                        all_reviews.append(review)\n        \n        # Filter reviews: require at least 3 reviews before calculating average\n        if len(all_reviews) >= 3:\n            # Find most recent review by review_date\n            valid_reviews = []\n            for rev in all_reviews:\n                if rev.get('review_date'):\n                    valid_reviews.append(rev)\n            \n            if valid_reviews:\n                valid_reviews.sort(key=lambda x: x.get('review_date', ''), reverse=True)\n                most_recent = valid_reviews[0]\n                most_recent_review = {\n                    'review_date': str(most_recent.get('review_date', '')),\n                    'review_title': str(most_recent.get('review_title', '')),\n                    'rating': str(most_recent.get('rating', '')),\n                    'provider_name': str(most_recent.get('provider_name', ''))\n                }\n            \n            # Calculate average rating from all reviews\n            total_rating = 0\n            count = 0\n            for rev in all_reviews:\n                rating_str = rev.get('rating', '')\n                if rating_str and rating_str.replace('.', '').isdigit():\n                    try:\n                        total_rating += float(rating_str)\n                        count += 1\n                    except:\n                        pass\n            \n            average_rating = round(total_rating / count, 2) if count > 0 else 0.0\n            \n            # Step 3: Search hotel pricing for Deluxe or Executive room\n            pricing = tools['search_hotel_pricing'](hotel_name)\n            current_rate = ''\n            \n            if isinstance(pricing, list):\n                deluxe_rates = []\n                executive_rates = []\n                \n                for price in pricing:\n                    if isinstance(price, dict):\n                        room_type = str(price.get('room_type', '')).lower()\n                        rate_val = price.get('rate', '')\n                        if 'deluxe' in room_type and rate_val:\n                            deluxe_rates.append(rate_val)\n                        elif 'executive' in room_type and rate_val:\n                            executive_rates.append(rate_val)\n                \n                # Prioritize Deluxe over Executive\n                if deluxe_rates:\n                    current_rate = str(deluxe_rates[0])\n                elif executive_rates:\n                    current_rate = str(executive_rates[0])\n            \n            # Calculate rating discrepancy\n            try:\n                star_float = float(stars) if stars else 0.0\n                discrepancy = round(average_rating - star_float, 2)\n            except:\n                discrepancy = 0.0\n            \n            analysis_list.append({\n                'hotel_name': hotel_name,\n                'stars': stars,\n                'amenities': amenities,\n                'most_recent_review': most_recent_review if most_recent_review else {\n                    'review_date': '',\n                    'review_title': '',\n                    'rating': '',\n                    'provider_name': ''\n                },\n                'current_deluxe_rate': current_rate,\n                'average_review_rating': average_rating,\n                'rating_discrepancy': discrepancy\n            })\n    \n    # Find hotel with highest positive discrepancy\n    highest_hotel = ''\n    highest_discrepancy = -999.0\n    \n    for item in analysis_list:\n        disc = item.get('rating_discrepancy', 0.0)\n        if disc > highest_discrepancy:\n            highest_discrepancy = disc\n            highest_hotel = item.get('hotel_name', '')\n    \n    # Prepare final answer with proper string types\n    answer = {\n        'analysis': analysis_list,\n        'hotel_with_highest_positive_discrepancy': str(highest_hotel),\n        'total_hotels_analyzed': str(len(analysis_list))\n    }\n    \n    return tools['submit_result'](answer)\n"
    verification_src = "def verify(tools, answer):\n    import json\n    try:\n        # Handle wrapped answer\n        if isinstance(answer, dict):\n            if 'submitted_data' in answer:\n                data = answer.get('submitted_data')\n            elif 'data' in answer:\n                data = answer.get('data')\n            else:\n                data = answer\n        else:\n            return {'passed': False, 'message': 'Answer is not a dict'}\n        \n        if data is None:\n            return {'passed': False, 'message': 'Answer data is None'}\n        \n        # Check required structure\n        required_keys = ['analysis', 'hotel_with_highest_positive_discrepancy', 'total_hotels_analyzed']\n        for key in required_keys:\n            if key not in data:\n                return {'passed': False, 'message': f'Missing required key: {key}'}\n        \n        analysis = data.get('analysis')\n        if not isinstance(analysis, list):\n            return {'passed': False, 'message': 'Analysis must be a list'}\n        \n        total_hotels = data.get('total_hotels_analyzed')\n        if not isinstance(total_hotels, str):\n            return {'passed': False, 'message': 'total_hotels_analyzed must be string'}\n        \n        # Verify total_hotels matches analysis length\n        try:\n            total_int = int(total_hotels)\n            if total_int != len(analysis):\n                return {'passed': False, 'message': f'total_hotels_analyzed {total_hotels} does not match analysis length {len(analysis)}'}\n        except:\n            return {'passed': False, 'message': 'total_hotels_analyzed must be convertible to integer'}\n        \n        # Verify each analysis item has correct structure\n        for idx, item in enumerate(analysis):\n            if not isinstance(item, dict):\n                return {'passed': False, 'message': f'Analysis item {idx} is not a dict'}\n            \n            required_item_keys = ['hotel_name', 'stars', 'amenities', 'most_recent_review', \n                                 'current_deluxe_rate', 'average_review_rating', 'rating_discrepancy']\n            for key in required_item_keys:\n                if key not in item:\n                    return {'passed': False, 'message': f'Analysis item {idx} missing key: {key}'}\n            \n            # Check types\n            if not isinstance(item['hotel_name'], str):\n                return {'passed': False, 'message': f'Hotel name at index {idx} must be string'}\n            if not isinstance(item['stars'], str):\n                return {'passed': False, 'message': f'Stars at index {idx} must be string'}\n            if not isinstance(item['amenities'], str):\n                return {'passed': False, 'message': f'Amenities at index {idx} must be string'}\n            if not isinstance(item['current_deluxe_rate'], str):\n                return {'passed': False, 'message': f'Current deluxe rate at index {idx} must be string'}\n            if not isinstance(item['average_review_rating'], float):\n                return {'passed': False, 'message': f'Average review rating at index {idx} must be float'}\n            if not isinstance(item['rating_discrepancy'], float):\n                return {'passed': False, 'message': f'Rating discrepancy at index {idx} must be float'}\n            \n            # Check most_recent_review structure\n            review = item['most_recent_review']\n            if not isinstance(review, dict):\n                return {'passed': False, 'message': f'Most recent review at index {idx} must be dict'}\n            \n            review_keys = ['review_date', 'review_title', 'rating', 'provider_name']\n            for rkey in review_keys:\n                if rkey not in review:\n                    return {'passed': False, 'message': f'Review at index {idx} missing key: {rkey}'}\n                if not isinstance(review[rkey], str):\n                    return {'passed': False, 'message': f'Review {rkey} at index {idx} must be string'}\n        \n        # Verify hotel_with_highest_positive_discrepancy is string\n        if not isinstance(data['hotel_with_highest_positive_discrepancy'], str):\n            return {'passed': False, 'message': 'hotel_with_highest_positive_discrepancy must be string'}\n        \n        # Cross-check with hotel listings tool\n        listings = tools['search_hotel_listings']('Paris')\n        if isinstance(listings, list):\n            # Count hotels with pool and wifi\n            pool_wifi_count = 0\n            for hotel in listings:\n                if isinstance(hotel, dict):\n                    amenities = hotel.get('amenities', '')\n                    if amenities and 'pool' in amenities.lower() and 'wifi' in amenities.lower():\n                        pool_wifi_count += 1\n            \n            # If we found hotels with pool+wifi, verify analysis is not empty\n            if pool_wifi_count > 0 and len(analysis) == 0:\n                return {'passed': False, 'message': f'Found {pool_wifi_count} hotels with pool+wifi but analysis is empty'}\n        \n        return {'passed': True, 'message': 'Verification passed'}\n    except Exception as e:\n        return {'passed': False, 'message': f'Verification error: {str(e)}'}\n"
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