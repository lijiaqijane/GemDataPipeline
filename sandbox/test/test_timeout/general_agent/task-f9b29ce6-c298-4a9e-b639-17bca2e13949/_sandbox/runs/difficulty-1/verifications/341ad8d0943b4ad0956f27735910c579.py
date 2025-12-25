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
        if not isinstance(count, str):
            return {'passed': False, 'message': 'Count is not a string'}
        if len(hotels) == 0:
            return {'passed': False, 'message': 'No hotels found'}
        if not count.isdigit() or int(count) <= 0:
            return {'passed': False, 'message': 'Count is zero or negative'}
        if int(count) != len(hotels):
            return {'passed': False, 'message': 'Count does not match hotels length'}
        for hotel in hotels:
            if not isinstance(hotel, dict):
                return {'passed': False, 'message': 'Hotel entry is not a dict'}
            required = ['name', 'stars', 'amenities', 'price', 'city']
            for key in required:
                if key not in hotel:
                    return {'passed': False, 'message': f'Missing hotel key: {key}'}
                if not isinstance(hotel[key], str):
                    return {'passed': False, 'message': f'Hotel {key} is not string'}
                if not hotel[key].strip():
                    return {'passed': False, 'message': f'Hotel {key} is empty'}
            if hotel['stars'] != '3':
                return {'passed': False, 'message': 'Hotel stars not 3'}
            if hotel['city'].lower() != 'paris':
                return {'passed': False, 'message': 'Hotel city not Paris'}
        check_hotels = tools['search_hotels']('Paris', 10)
        if not isinstance(check_hotels, list):
            return {'passed': False, 'message': 'Tool check failed'}
        return {'passed': True, 'message': 'Verification passed'}
    except Exception as e:
        return {'passed': False, 'message': f'Exception: {str(e)}'}
