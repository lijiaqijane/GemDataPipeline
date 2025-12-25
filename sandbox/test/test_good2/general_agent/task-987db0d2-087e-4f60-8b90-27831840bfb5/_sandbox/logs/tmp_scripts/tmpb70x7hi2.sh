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
    solution_src = "def solve(tools):\n    # Get hotel pricing data\n    hotel_data = tools['search_hotel_pricing']('')\n    \n    # Get travel recommendations data\n    travel_data = tools['search_travel_recommendations']('Paris')\n    \n    # Process hotel data for ADR premium and correlation\n    high_rated_rates = []\n    low_rated_rates = []\n    all_rates = []\n    all_ratings = []\n    premium_hotels = []\n    \n    if isinstance(hotel_data, list):\n        for item in hotel_data:\n            if isinstance(item, dict):\n                try:\n                    hotel_name = str(item.get('hotel_name', '')).strip()\n                    rate_str = str(item.get('rate', '0')).replace('$', '').replace(',', '').strip()\n                    rating_str = str(item.get('guest_rating', '0')).strip()\n                    special_offer = str(item.get('special_offer', '')).strip()\n                    \n                    if rate_str and rating_str:\n                        rate = float(rate_str)\n                        rating = float(rating_str)\n                        \n                        # Collect for correlation\n                        all_rates.append(rate)\n                        all_ratings.append(rating)\n                        \n                        # Categorize for ADR premium\n                        if rating >= 4.5:\n                            high_rated_rates.append(rate)\n                        else:\n                            low_rated_rates.append(rate)\n                        \n                        # Check for premium hotels criteria\n                        if rating >= 4.5 and special_offer and special_offer.lower() != 'none' and special_offer != '':\n                            premium_hotels.append({\n                                'hotel_name': hotel_name,\n                                'guest_rating': rating,\n                                'rate': rate,\n                                'special_offer': special_offer\n                            })\n                except (ValueError, TypeError):\n                    continue\n    \n    # Calculate ADR premium\n    adr_premium = 0.0\n    if high_rated_rates and low_rated_rates:\n        avg_high = sum(high_rated_rates) / len(high_rated_rates)\n        avg_low = sum(low_rated_rates) / len(low_rated_rates)\n        if avg_low > 0:\n            adr_premium = ((avg_high - avg_low) / avg_low) * 100\n    \n    # Calculate 75th percentile rate\n    if all_rates:\n        sorted_rates = sorted(all_rates)\n        idx = int(0.75 * len(sorted_rates))\n        percentile_75 = sorted_rates[idx] if idx < len(sorted_rates) else sorted_rates[-1]\n        \n        # Filter premium hotels below 75th percentile\n        filtered_premium = [h for h in premium_hotels if h['rate'] < percentile_75]\n        \n        # Sort by rating descending, then rate ascending\n        filtered_premium.sort(key=lambda x: (-x['guest_rating'], x['rate']))\n    else:\n        filtered_premium = []\n        percentile_75 = 0\n    \n    # Process travel data for most expensive accommodation type\n    accommodation_costs = {}\n    accommodation_counts = {}\n    american_trips = 0\n    \n    if isinstance(travel_data, list):\n        for item in travel_data:\n            if isinstance(item, dict):\n                try:\n                    dest = str(item.get('destination', '')).lower()\n                    nationality = str(item.get('traveler_nationality', '')).lower()\n                    gender = str(item.get('traveler_gender', '')).lower()\n                    acc_type = str(item.get('accommodation_type', '')).strip()\n                    cost_str = str(item.get('accommodation_cost', '0')).replace('$', '').replace(',', '').strip()\n                    days_str = str(item.get('duration_days', '0')).strip()\n                    \n                    if ('paris' in dest and 'american' in nationality and \n                        cost_str and days_str and cost_str != '0' and days_str != '0'):\n                        cost = float(cost_str)\n                        days = float(days_str)\n                        \n                        if cost > 0 and days > 0:\n                            american_trips += 1\n                            cost_per_night = cost / days\n                            \n                            if acc_type:\n                                if acc_type not in accommodation_costs:\n                                    accommodation_costs[acc_type] = 0.0\n                                    accommodation_counts[acc_type] = 0\n                                \n                                accommodation_costs[acc_type] += cost_per_night\n                                accommodation_counts[acc_type] += 1\n                except (ValueError, TypeError):\n                    continue\n    \n    # Find most expensive accommodation type\n    most_expensive_type = ''\n    if accommodation_costs:\n        avg_costs = {acc_type: accommodation_costs[acc_type] / accommodation_counts[acc_type] \n                     for acc_type in accommodation_costs}\n        if avg_costs:\n            most_expensive_type = max(avg_costs.items(), key=lambda x: x[1])[0]\n    \n    # Calculate correlation coefficient\n    correlation = 0.0\n    if len(all_rates) > 1 and len(all_ratings) > 1:\n        n = len(all_rates)\n        sum_xy = sum(all_rates[i] * all_ratings[i] for i in range(n))\n        sum_x = sum(all_rates)\n        sum_y = sum(all_ratings)\n        sum_x2 = sum(r * r for r in all_rates)\n        sum_y2 = sum(r * r for r in all_ratings)\n        \n        numerator = n * sum_xy - sum_x * sum_y\n        denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)) ** 0.5\n        \n        if denominator != 0:\n            correlation = numerator / denominator\n    \n    # Prepare answer\n    answer = {\n        'adr_premium_percentage': round(adr_premium, 1),\n        'most_expensive_accommodation_type': most_expensive_type if most_expensive_type else 'Hotel',\n        'premium_hotels_below_75th_percentile': filtered_premium,\n        'rating_rate_correlation': round(correlation, 3),\n        'data_quality_notes': {\n            'high_rated_hotels_analyzed': len(high_rated_rates),\n            'low_rated_hotels_analyzed': len(low_rated_rates),\n            'american_traveler_trips_analyzed': american_trips,\n            'total_hotels_in_correlation': len(all_rates)\n        }\n    }\n    \n    return tools['submit_result'](answer)\n"
    verification_src = "def verify(tools, answer):\n    import json\n    \n    try:\n        # Handle wrapped answer\n        if isinstance(answer, dict):\n            if 'status' in answer and 'data' in answer:\n                answer_data = answer.get('data')\n            elif 'submitted_data' in answer:\n                answer_data = answer.get('submitted_data')\n            else:\n                answer_data = answer\n        else:\n            return {'passed': False, 'message': 'Answer is not a dictionary'}\n        \n        if not isinstance(answer_data, dict):\n            return {'passed': False, 'message': 'Answer data is not a dictionary'}\n        \n        # Check required keys\n        required_keys = [\n            'adr_premium_percentage',\n            'most_expensive_accommodation_type',\n            'premium_hotels_below_75th_percentile',\n            'rating_rate_correlation',\n            'data_quality_notes'\n        ]\n        \n        for key in required_keys:\n            if key not in answer_data:\n                return {'passed': False, 'message': f'Missing required key: {key}'}\n        \n        # Check data quality notes structure\n        notes = answer_data.get('data_quality_notes', {})\n        if not isinstance(notes, dict):\n            return {'passed': False, 'message': 'data_quality_notes is not a dictionary'}\n        \n        required_notes = [\n            'high_rated_hotels_analyzed',\n            'low_rated_hotels_analyzed',\n            'american_traveler_trips_analyzed',\n            'total_hotels_in_correlation'\n        ]\n        \n        for note_key in required_notes:\n            if note_key not in notes:\n                return {'passed': False, 'message': f'Missing data quality note: {note_key}'}\n        \n        # Check non-empty values\n        adr_premium = answer_data.get('adr_premium_percentage')\n        if adr_premium is None:\n            return {'passed': False, 'message': 'adr_premium_percentage is None'}\n        \n        acc_type = answer_data.get('most_expensive_accommodation_type')\n        if not acc_type or not isinstance(acc_type, str) or acc_type.strip() == '':\n            return {'passed': False, 'message': 'most_expensive_accommodation_type is empty or invalid'}\n        \n        correlation = answer_data.get('rating_rate_correlation')\n        if correlation is None:\n            return {'passed': False, 'message': 'rating_rate_correlation is None'}\n        \n        # Check premium hotels list\n        premium_hotels = answer_data.get('premium_hotels_below_75th_percentile', [])\n        if not isinstance(premium_hotels, list):\n            return {'passed': False, 'message': 'premium_hotels_below_75th_percentile is not a list'}\n        \n        # Verify with actual data using a tool\n        hotel_data = tools['search_hotel_pricing']('')\n        \n        if isinstance(hotel_data, list) and len(hotel_data) > 0:\n            # Check that we have some data to verify against\n            hotel_count = 0\n            for item in hotel_data:\n                if isinstance(item, dict) and 'hotel_name' in item:\n                    hotel_count += 1\n            \n            if hotel_count == 0:\n                return {'passed': False, 'message': 'No valid hotel data found to verify against'}\n            \n            # Check that premium hotels list is not empty when there should be data\n            if hotel_count > 0 and len(premium_hotels) == 0:\n                # This might be okay if no hotels meet criteria, but check notes\n                high_rated = notes.get('high_rated_hotels_analyzed', 0)\n                if high_rated > 0:\n                    return {'passed': False, 'message': 'Premium hotels list is empty but high-rated hotels exist'}\n        \n        # Check data quality notes values\n        high_rated = notes.get('high_rated_hotels_analyzed', 0)\n        low_rated = notes.get('low_rated_hotels_analyzed', 0)\n        american_trips = notes.get('american_traveler_trips_analyzed', 0)\n        total_hotels = notes.get('total_hotels_in_correlation', 0)\n        \n        if not isinstance(high_rated, (int, float)) or high_rated < 0:\n            return {'passed': False, 'message': 'Invalid high_rated_hotels_analyzed value'}\n        \n        if not isinstance(low_rated, (int, float)) or low_rated < 0:\n            return {'passed': False, 'message': 'Invalid low_rated_hotels_analyzed value'}\n        \n        if not isinstance(american_trips, (int, float)) or american_trips < 0:\n            return {'passed': False, 'message': 'Invalid american_traveler_trips_analyzed value'}\n        \n        if not isinstance(total_hotels, (int, float)) or total_hotels < 0:\n            return {'passed': False, 'message': 'Invalid total_hotels_in_correlation value'}\n        \n        # Check correlation value range\n        if not isinstance(correlation, (int, float)):\n            return {'passed': False, 'message': 'rating_rate_correlation is not numeric'}\n        \n        if correlation < -1.0 or correlation > 1.0:\n            return {'passed': False, 'message': 'rating_rate_correlation out of valid range [-1, 1]'}\n        \n        # Check premium hotels structure\n        for hotel in premium_hotels:\n            if not isinstance(hotel, dict):\n                return {'passed': False, 'message': 'Premium hotel entry is not a dictionary'}\n            \n            required_hotel_keys = ['hotel_name', 'guest_rating', 'rate', 'special_offer']\n            for key in required_hotel_keys:\n                if key not in hotel:\n                    return {'passed': False, 'message': f'Premium hotel missing key: {key}'}\n            \n            # Check non-empty values\n            if not hotel.get('hotel_name') or not isinstance(hotel['hotel_name'], str):\n                return {'passed': False, 'message': 'Hotel name is empty or invalid'}\n            \n            if not isinstance(hotel.get('guest_rating'), (int, float)):\n                return {'passed': False, 'message': 'Guest rating is not numeric'}\n            \n            if not isinstance(hotel.get('rate'), (int, float)):\n                return {'passed': False, 'message': 'Rate is not numeric'}\n            \n            if not hotel.get('special_offer') or not isinstance(hotel['special_offer'], str):\n                return {'passed': False, 'message': 'Special offer is empty or invalid'}\n        \n        return {'passed': True, 'message': 'All verification checks passed'}\n        \n    except Exception as e:\n        return {'passed': False, 'message': f'Verification error: {str(e)}'}\n"
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