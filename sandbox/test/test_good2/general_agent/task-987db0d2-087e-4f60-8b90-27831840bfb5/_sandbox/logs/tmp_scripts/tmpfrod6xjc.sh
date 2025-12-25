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
    solution_src = "def solve(tools):\n    # Get hotel pricing data\n    hotel_data = tools['search_hotel_pricing']('')\n    \n    # Get travel recommendations data for Paris trips\n    travel_data = tools['search_travel_recommendations']('Paris')\n    \n    # Initialize data structures\n    high_rated_rates = []\n    low_rated_rates = []\n    all_rates = []\n    all_ratings = []\n    premium_hotels = []\n    \n    # Process hotel data\n    if isinstance(hotel_data, list):\n        for item in hotel_data:\n            if isinstance(item, dict):\n                try:\n                    hotel_name = str(item.get('hotel_name', '')).strip()\n                    rate_str = str(item.get('rate', '0')).replace('$', '').replace(',', '').strip()\n                    rating_str = str(item.get('guest_rating', '0')).strip()\n                    special_offer = str(item.get('special_offer', '')).strip()\n                    \n                    if rate_str and rating_str:\n                        rate = float(rate_str)\n                        rating = float(rating_str)\n                        \n                        # Collect for correlation\n                        all_rates.append(rate)\n                        all_ratings.append(rating)\n                        \n                        # Categorize by rating for ADR premium\n                        if rating >= 4.5:\n                            high_rated_rates.append(rate)\n                        else:\n                            low_rated_rates.append(rate)\n                        \n                        # Check for premium hotel criteria\n                        if rating >= 4.5 and special_offer and special_offer.lower() != 'none' and special_offer.strip() != '':\n                            premium_hotels.append({\n                                'hotel_name': hotel_name,\n                                'guest_rating': rating,\n                                'rate': rate,\n                                'special_offer': special_offer\n                            })\n                except (ValueError, TypeError):\n                    continue\n    \n    # Calculate ADR premium percentage\n    adr_premium = 0.0\n    if high_rated_rates and low_rated_rates:\n        avg_high = sum(high_rated_rates) / len(high_rated_rates)\n        avg_low = sum(low_rated_rates) / len(low_rated_rates)\n        if avg_low > 0:\n            adr_premium = ((avg_high - avg_low) / avg_low) * 100\n    \n    # Round and convert to string\n    adr_premium_str = str(round(adr_premium, 1))\n    \n    # Calculate 75th percentile rate for premium hotels\n    if all_rates:\n        sorted_rates = sorted(all_rates)\n        idx = int(0.75 * len(sorted_rates))\n        percentile_75 = sorted_rates[idx] if idx < len(sorted_rates) else sorted_rates[-1]\n    else:\n        percentile_75 = float('inf')\n    \n    # Filter premium hotels below 75th percentile\n    filtered_premium_hotels = []\n    for hotel in premium_hotels:\n        if hotel['rate'] < percentile_75:\n            filtered_premium_hotels.append(hotel)\n    \n    # Sort premium hotels by rating descending, then rate ascending\n    filtered_premium_hotels.sort(key=lambda x: (-x['guest_rating'], x['rate']))\n    \n    # Calculate correlation coefficient\n    correlation = 0.0\n    if len(all_rates) >= 2 and len(all_ratings) >= 2:\n        n = len(all_rates)\n        sum_x = sum(all_rates)\n        sum_y = sum(all_ratings)\n        sum_xy = sum(all_rates[i] * all_ratings[i] for i in range(n))\n        sum_x2 = sum(r * r for r in all_rates)\n        sum_y2 = sum(r * r for r in all_ratings)\n        \n        numerator = n * sum_xy - sum_x * sum_y\n        denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)) ** 0.5\n        \n        if denominator != 0:\n            correlation = numerator / denominator\n    \n    # Round correlation to 3 decimals and convert to string\n    correlation_str = str(round(correlation, 3))\n    \n    # Process travel data for most expensive accommodation type\n    accommodation_costs = {}\n    american_trips = 0\n    \n    if isinstance(travel_data, list):\n        for item in travel_data:\n            if isinstance(item, dict):\n                try:\n                    destination = str(item.get('destination', '')).strip()\n                    nationality = str(item.get('traveler_nationality', '')).strip()\n                    acc_type = str(item.get('accommodation_type', '')).strip()\n                    cost_str = str(item.get('accommodation_cost', '0')).replace('$', '').replace(',', '').strip()\n                    duration_str = str(item.get('duration_days', '0')).strip()\n                    \n                    # Check if it's an American traveler to Paris\n                    if 'paris' in destination.lower() and 'american' in nationality.lower():\n                        cost = float(cost_str) if cost_str else 0\n                        duration = float(duration_str) if duration_str else 0\n                        \n                        if cost > 0 and duration > 0:\n                            american_trips += 1\n                            cost_per_night = cost / duration\n                            \n                            if acc_type:\n                                if acc_type not in accommodation_costs:\n                                    accommodation_costs[acc_type] = []\n                                accommodation_costs[acc_type].append(cost_per_night)\n                except (ValueError, TypeError):\n                    continue\n    \n    # Find most expensive accommodation type\n    most_expensive_type = ''\n    max_avg_cost = 0\n    \n    for acc_type, costs in accommodation_costs.items():\n        if costs:\n            avg_cost = sum(costs) / len(costs)\n            if avg_cost > max_avg_cost:\n                max_avg_cost = avg_cost\n                most_expensive_type = acc_type\n    \n    # If no accommodation type found, set default\n    if not most_expensive_type:\n        most_expensive_type = 'Hotel'\n    \n    # Prepare data quality notes\n    data_quality_notes = {\n        'high_rated_hotels_analyzed': str(len(high_rated_rates)),\n        'low_rated_hotels_analyzed': str(len(low_rated_rates)),\n        'american_traveler_trips_analyzed': str(american_trips),\n        'total_hotels_in_correlation': str(len(all_rates))\n    }\n    \n    # Prepare final answer with proper string conversions\n    answer = {\n        'adr_premium_percentage': adr_premium_str,\n        'most_expensive_accommodation_type': most_expensive_type,\n        'premium_hotels_below_75th_percentile': filtered_premium_hotels,\n        'rating_rate_correlation': correlation_str,\n        'data_quality_notes': data_quality_notes\n    }\n    \n    return tools['submit_result'](answer)\n"
    verification_src = "def verify(tools, answer):\n    import json\n    \n    try:\n        # Handle wrapped answer\n        if isinstance(answer, dict):\n            if 'status' in answer and 'data' in answer:\n                answer_data = answer.get('data')\n            elif 'submitted_data' in answer:\n                answer_data = answer.get('submitted_data')\n            else:\n                answer_data = answer\n        else:\n            return {'passed': False, 'message': 'Answer is not a dictionary'}\n        \n        if not isinstance(answer_data, dict):\n            return {'passed': False, 'message': 'Answer data is not a dictionary'}\n        \n        # Check required keys\n        required_keys = [\n            'adr_premium_percentage',\n            'most_expensive_accommodation_type',\n            'premium_hotels_below_75th_percentile',\n            'rating_rate_correlation',\n            'data_quality_notes'\n        ]\n        \n        for key in required_keys:\n            if key not in answer_data:\n                return {'passed': False, 'message': f'Missing required key: {key}'}\n        \n        # Check data quality notes structure\n        notes = answer_data.get('data_quality_notes', {})\n        if not isinstance(notes, dict):\n            return {'passed': False, 'message': 'data_quality_notes must be a dictionary'}\n        \n        required_note_keys = [\n            'high_rated_hotels_analyzed',\n            'low_rated_hotels_analyzed',\n            'american_traveler_trips_analyzed',\n            'total_hotels_in_correlation'\n        ]\n        \n        for key in required_note_keys:\n            if key not in notes:\n                return {'passed': False, 'message': f'Missing data_quality_notes key: {key}'}\n            \n            # Check that values are strings\n            if not isinstance(notes[key], str):\n                return {'passed': False, 'message': f'data_quality_notes[{key}] must be a string'}\n        \n        # Check field types as per format specification\n        adr_premium = answer_data.get('adr_premium_percentage')\n        if not isinstance(adr_premium, str):\n            return {'passed': False, 'message': 'adr_premium_percentage must be a string'}\n        \n        # Try to parse as float to validate format\n        try:\n            float_val = float(adr_premium)\n            # Check if it's properly formatted with 1 decimal\n            if '.' in adr_premium:\n                decimal_part = adr_premium.split('.')[1]\n                if len(decimal_part) != 1:\n                    return {'passed': False, 'message': 'adr_premium_percentage must have exactly 1 decimal place'}\n        except ValueError:\n            return {'passed': False, 'message': 'adr_premium_percentage must be a valid numeric string'}\n        \n        accommodation_type = answer_data.get('most_expensive_accommodation_type')\n        if not isinstance(accommodation_type, str):\n            return {'passed': False, 'message': 'most_expensive_accommodation_type must be a string'}\n        \n        if not accommodation_type:\n            return {'passed': False, 'message': 'most_expensive_accommodation_type cannot be empty'}\n        \n        correlation = answer_data.get('rating_rate_correlation')\n        if not isinstance(correlation, str):\n            return {'passed': False, 'message': 'rating_rate_correlation must be a string'}\n        \n        # Validate correlation format\n        try:\n            corr_val = float(correlation)\n            if not (-1.0 <= corr_val <= 1.0):\n                return {'passed': False, 'message': 'rating_rate_correlation must be between -1.0 and 1.0'}\n            \n            # Check decimal places\n            if '.' in correlation:\n                decimal_part = correlation.split('.')[1]\n                if len(decimal_part) != 3:\n                    return {'passed': False, 'message': 'rating_rate_correlation must have exactly 3 decimal places'}\n        except ValueError:\n            return {'passed': False, 'message': 'rating_rate_correlation must be a valid numeric string'}\n        \n        premium_hotels = answer_data.get('premium_hotels_below_75th_percentile')\n        if not isinstance(premium_hotels, list):\n            return {'passed': False, 'message': 'premium_hotels_below_75th_percentile must be a list'}\n        \n        # Check that list is not empty (meaningful content requirement)\n        if not premium_hotels:\n            return {'passed': False, 'message': 'premium_hotels_below_75th_percentile list cannot be empty'}\n        \n        # Validate each hotel entry\n        for i, hotel in enumerate(premium_hotels):\n            if not isinstance(hotel, dict):\n                return {'passed': False, 'message': f'Hotel entry {i} must be a dictionary'}\n            \n            required_hotel_keys = ['hotel_name', 'guest_rating', 'rate', 'special_offer']\n            for key in required_hotel_keys:\n                if key not in hotel:\n                    return {'passed': False, 'message': f'Hotel entry {i} missing key: {key}'}\n            \n            # Check hotel name is non-empty string\n            hotel_name = hotel.get('hotel_name')\n            if not isinstance(hotel_name, str) or not hotel_name.strip():\n                return {'passed': False, 'message': f'Hotel entry {i} must have non-empty hotel_name'}\n            \n            # Check special offer is non-empty string\n            special_offer = hotel.get('special_offer')\n            if not isinstance(special_offer, str) or not special_offer.strip():\n                return {'passed': False, 'message': f'Hotel entry {i} must have non-empty special_offer'}\n            \n            # Check rating and rate are numeric\n            try:\n                rating = float(hotel.get('guest_rating', 0))\n                rate = float(hotel.get('rate', 0))\n                \n                if rating < 4.5:\n                    return {'passed': False, 'message': f'Hotel entry {i} rating must be \u22654.5'}\n                \n                if rate <= 0:\n                    return {'passed': False, 'message': f'Hotel entry {i} rate must be positive'}\n            except (ValueError, TypeError):\n                return {'passed': False, 'message': f'Hotel entry {i} has invalid numeric values'}\n        \n        # Use a data tool to cross-check\n        hotel_data = tools['search_hotel_pricing']('')\n        if isinstance(hotel_data, list) and hotel_data:\n            # Verify that at least some hotels exist in the data\n            hotel_count = 0\n            for item in hotel_data:\n                if isinstance(item, dict) and 'hotel_name' in item:\n                    hotel_count += 1\n            \n            if hotel_count == 0:\n                return {'passed': False, 'message': 'No hotel data found to verify against'}\n        \n        # Check data quality notes values are meaningful\n        high_rated = notes.get('high_rated_hotels_analyzed', '0')\n        low_rated = notes.get('low_rated_hotels_analyzed', '0')\n        \n        try:\n            high_int = int(high_rated)\n            low_int = int(low_rated)\n            \n            # ADR premium calculation requires both groups\n            if high_int == 0 or low_int == 0:\n                return {'passed': False, 'message': 'ADR premium requires both high and low rated hotels'}\n        except ValueError:\n            return {'passed': False, 'message': 'Invalid numeric values in data_quality_notes'}\n        \n        return {'passed': True, 'message': 'All verification checks passed'}\n        \n    except Exception as e:\n        return {'passed': False, 'message': f'Verification error: {str(e)}'}\n"
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