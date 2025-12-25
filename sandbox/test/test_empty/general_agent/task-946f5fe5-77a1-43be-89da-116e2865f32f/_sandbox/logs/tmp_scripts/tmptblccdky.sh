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
    solution_src = "def solve(tools):\n    # Step 1: Find Paris hotels with both 'pool' and 'wifi' amenities\n    listings = tools['search_hotel_listings']('Paris, France')\n    paris_hotels = []\n    if isinstance(listings, list):\n        for hotel in listings:\n            if isinstance(hotel, dict):\n                amenities = hotel.get('amenities', '')\n                city = hotel.get('city', '')\n                country = hotel.get('country', '')\n                if amenities and city and country:\n                    amenities_lower = amenities.lower()\n                    if 'pool' in amenities_lower and 'wifi' in amenities_lower:\n                        if 'paris' in city.lower() and 'france' in country.lower():\n                            paris_hotels.append({\n                                'name': str(hotel.get('name', '')),\n                                'stars': str(hotel.get('stars', '')),\n                                'amenities': amenities,\n                                'city': city,\n                                'country': country\n                            })\n    \n    analysis_list = []\n    \n    for hotel in paris_hotels:\n        hotel_name = hotel['name']\n        stars = hotel['stars']\n        amenities = hotel['amenities']\n        \n        # Step 2: Find most recent review for this hotel\n        reviews = tools['search_travel_reviews'](hotel_name)\n        most_recent_review = None\n        all_ratings = []\n        \n        if isinstance(reviews, list):\n            for review in reviews:\n                if isinstance(review, dict):\n                    provider = review.get('provider_name', '')\n                    rating_str = review.get('rating', '')\n                    review_date = review.get('review_date', '')\n                    \n                    # Check if review mentions hotel name or provider matches\n                    if hotel_name.lower() in provider.lower() or hotel_name.lower() in str(review.get('review_text', '')).lower():\n                        try:\n                            rating = float(rating_str)\n                            all_ratings.append(rating)\n                        except:\n                            pass\n                        \n                        # Find most recent by date\n                        if review_date:\n                            if most_recent_review is None:\n                                most_recent_review = review\n                            else:\n                                # Simple date comparison (assuming format like '1/10/2025')\n                                try:\n                                    new_date_parts = review_date.split('/')\n                                    old_date_parts = most_recent_review.get('review_date', '').split('/')\n                                    if len(new_date_parts) == 3 and len(old_date_parts) == 3:\n                                        new_year = int(new_date_parts[2])\n                                        new_month = int(new_date_parts[0])\n                                        old_year = int(old_date_parts[2])\n                                        old_month = int(old_date_parts[0])\n                                        if new_year > old_year or (new_year == old_year and new_month > old_month):\n                                            most_recent_review = review\n                                except:\n                                    pass\n        \n        # Step 3: Find current deluxe/executive room rate\n        pricing = tools['search_hotel_pricing'](hotel_name)\n        current_rate = 'Not found'\n        \n        if isinstance(pricing, list):\n            for price_entry in pricing:\n                if isinstance(price_entry, dict):\n                    room_type = price_entry.get('room_type', '')\n                    rate = price_entry.get('rate', '')\n                    date = price_entry.get('date', '')\n                    \n                    if room_type and ('deluxe' in room_type.lower() or 'executive' in room_type.lower()):\n                        if rate and rate != 'Sold Out' and rate != 'Not available':\n                            current_rate = str(rate)\n                            break\n        \n        # Step 4: Calculate average rating and discrepancy\n        avg_rating = 0.0\n        discrepancy = 0.0\n        \n        if all_ratings:\n            avg_rating = sum(all_ratings) / len(all_ratings)\n            try:\n                star_float = float(stars)\n                discrepancy = avg_rating - star_float\n            except:\n                discrepancy = avg_rating\n        \n        # Prepare most recent review details\n        review_details = {\n            'review_date': '',\n            'review_title': '',\n            'rating': '',\n            'provider_name': ''\n        }\n        \n        if most_recent_review:\n            review_details = {\n                'review_date': str(most_recent_review.get('review_date', '')),\n                'review_title': str(most_recent_review.get('review_title', '')),\n                'rating': str(most_recent_review.get('rating', '')),\n                'provider_name': str(most_recent_review.get('provider_name', ''))\n            }\n        \n        analysis_list.append({\n            'hotel_name': hotel_name,\n            'stars': stars,\n            'amenities': amenities,\n            'most_recent_review': review_details,\n            'current_deluxe_rate': current_rate,\n            'average_review_rating': round(avg_rating, 2),\n            'rating_discrepancy': round(discrepancy, 2)\n        })\n    \n    # Step 5: Find hotel with highest positive discrepancy\n    highest_hotel = ''\n    highest_discrepancy = -999.0\n    \n    for item in analysis_list:\n        disc = item.get('rating_discrepancy', 0.0)\n        if disc > highest_discrepancy:\n            highest_discrepancy = disc\n            highest_hotel = item.get('hotel_name', '')\n    \n    answer = {\n        'analysis': analysis_list,\n        'hotel_with_highest_positive_discrepancy': highest_hotel,\n        'total_hotels_analyzed': len(analysis_list)\n    }\n    \n    return tools['submit_result'](answer)\n"
    verification_src = "def verify(tools, answer):\n    import json\n    try:\n        # Handle wrapped answer\n        if isinstance(answer, dict):\n            if 'submitted_data' in answer:\n                data = answer.get('submitted_data')\n            elif 'data' in answer:\n                data = answer.get('data')\n            else:\n                data = answer\n        else:\n            return {'passed': False, 'message': 'Answer is not a dict'}\n        \n        if data is None:\n            return {'passed': False, 'message': 'Answer data is None'}\n        \n        # Check required structure\n        required_keys = ['analysis', 'hotel_with_highest_positive_discrepancy', 'total_hotels_analyzed']\n        for key in required_keys:\n            if key not in data:\n                return {'passed': False, 'message': f'Missing required key: {key}'}\n        \n        analysis = data.get('analysis')\n        if not isinstance(analysis, list):\n            return {'passed': False, 'message': 'Analysis must be a list'}\n        \n        total_hotels = data.get('total_hotels_analyzed')\n        if not isinstance(total_hotels, int):\n            return {'passed': False, 'message': 'total_hotels_analyzed must be integer'}\n        \n        if len(analysis) != total_hotels:\n            return {'passed': False, 'message': f'Analysis count {len(analysis)} != total_hotels {total_hotels}'}\n        \n        # Verify each hotel in analysis\n        for idx, hotel_analysis in enumerate(analysis):\n            if not isinstance(hotel_analysis, dict):\n                return {'passed': False, 'message': f'Hotel analysis {idx} is not dict'}\n            \n            required_hotel_keys = ['hotel_name', 'stars', 'amenities', 'most_recent_review', \n                                  'current_deluxe_rate', 'average_review_rating', 'rating_discrepancy']\n            for key in required_hotel_keys:\n                if key not in hotel_analysis:\n                    return {'passed': False, 'message': f'Hotel {idx} missing key: {key}'}\n            \n            # Check types\n            if not isinstance(hotel_analysis['hotel_name'], str):\n                return {'passed': False, 'message': f'Hotel {idx} name not string'}\n            \n            if not isinstance(hotel_analysis['stars'], str):\n                return {'passed': False, 'message': f'Hotel {idx} stars not string'}\n            \n            if not isinstance(hotel_analysis['average_review_rating'], (int, float)):\n                return {'passed': False, 'message': f'Hotel {idx} average rating not numeric'}\n            \n            if not isinstance(hotel_analysis['rating_discrepancy'], (int, float)):\n                return {'passed': False, 'message': f'Hotel {idx} discrepancy not numeric'}\n            \n            # Verify most_recent_review structure\n            review = hotel_analysis.get('most_recent_review')\n            if not isinstance(review, dict):\n                return {'passed': False, 'message': f'Hotel {idx} review not dict'}\n            \n            review_keys = ['review_date', 'review_title', 'rating', 'provider_name']\n            for rkey in review_keys:\n                if rkey not in review:\n                    return {'passed': False, 'message': f'Hotel {idx} review missing {rkey}'}\n                if not isinstance(review[rkey], str):\n                    return {'passed': False, 'message': f'Hotel {idx} review {rkey} not string'}\n        \n        # Cross-check with actual data using a tool\n        if analysis:\n            # Use search_hotel_listings to verify at least one hotel exists\n            sample_hotel = analysis[0]['hotel_name']\n            listings_check = tools['search_hotel_listings'](sample_hotel)\n            \n            if isinstance(listings_check, list) and len(listings_check) > 0:\n                # Found some data\n                pass\n            else:\n                # No data found - might be okay if hotel doesn't exist in listings\n                pass\n        \n        # Verify hotel_with_highest_positive_discrepancy\n        highest_hotel = data.get('hotel_with_highest_positive_discrepancy')\n        if not isinstance(highest_hotel, str):\n            return {'passed': False, 'message': 'Highest hotel name not string'}\n        \n        # If there are hotels, verify highest hotel is in analysis\n        if analysis and highest_hotel:\n            hotel_names = [h.get('hotel_name', '') for h in analysis]\n            if highest_hotel not in hotel_names and highest_hotel != '':\n                return {'passed': False, 'message': 'Highest hotel not in analysis list'}\n        \n        return {'passed': True, 'message': 'Verification passed', 'details': {'hotels_analyzed': total_hotels}}\n        \n    except Exception as e:\n        return {'passed': False, 'message': f'Verification error: {str(e)}'}\n"
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