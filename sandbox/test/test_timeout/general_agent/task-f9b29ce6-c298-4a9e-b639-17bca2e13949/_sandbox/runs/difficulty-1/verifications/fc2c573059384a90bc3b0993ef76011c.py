def verify(tools, answer):
    try:
        import json
        if answer is None:
            return {'passed': False, 'message': 'Answer is None'}
        if isinstance(answer, dict) and 'status' in answer and 'data' in answer:
            answer = answer['data']
        elif isinstance(answer, dict) and 'submitted_data' in answer:
            answer = answer['submitted_data']
        if not isinstance(answer, dict):
            return {'passed': False, 'message': 'Answer is not a dict'}
        if 'hotels' not in answer or 'count' not in answer:
            return {'passed': False, 'message': 'Missing required keys'}
        hotels = answer.get('hotels')
        count = answer.get('count')
        if not isinstance(hotels, list):
            return {'passed': False, 'message': 'Hotels is not a list'}
        if not isinstance(count, int):
            return {'passed': False, 'message': 'Count is not int'}
        if len(hotels) == 0:
            return {'passed': False, 'message': 'No hotels found'}
        if count <= 0:
            return {'passed': False, 'message': 'Count is zero or negative'}
        if count != len(hotels):
            return {'passed': False, 'message': 'Count does not match hotels length'}
        for h in hotels:
            if not isinstance(h, dict):
                return {'passed': False, 'message': 'Hotel entry is not dict'}
            required = ['name', 'stars', 'amenities', 'price', 'city']
            for key in required:
                if key not in h:
                    return {'passed': False, 'message': f'Missing key {key}'}
                val = h[key]
                if not isinstance(val, str):
                    return {'passed': False, 'message': f'{key} is not string'}
                if not val.strip():
                    return {'passed': False, 'message': f'{key} is empty'}
            if h['stars'] != '3':
                return {'passed': False, 'message': 'Stars not 3'}
            if h['city'].lower() != 'paris':
                return {'passed': False, 'message': 'City not Paris'}
        raw = tools['search_hotels']('Paris', 100)
        if not isinstance(raw, list):
            return {'passed': False, 'message': 'Tool returned non-list'}
        paris_3star = [h for h in raw if isinstance(h, dict) and str(h.get('stars', '')).strip() == '3' and str(h.get('city', '')).strip().lower() == 'paris']
        if len(paris_3star) == 0:
            return {'passed': False, 'message': 'No 3-star Paris hotels in raw data'}
        if len(hotels) > len(paris_3star):
            return {'passed': False, 'message': 'More hotels than possible in raw data'}
        return True
    except Exception as e:
        return {'passed': False, 'message': f'Exception: {str(e)}'}
