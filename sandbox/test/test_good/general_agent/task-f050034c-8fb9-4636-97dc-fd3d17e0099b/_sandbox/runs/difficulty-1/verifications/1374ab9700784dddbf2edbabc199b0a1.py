def verify(tools, answer):
    try:
        import json
        if answer is None:
            return {'passed': False, 'message': 'Answer is None'}
        if isinstance(answer, dict) and 'status' in answer:
            if 'submitted_data' in answer:
                data = answer['submitted_data']
            elif 'data' in answer:
                data = answer['data']
            else:
                data = answer
        else:
            data = answer
        if not isinstance(data, dict):
            return {'passed': False, 'message': 'Answer data is not a dict'}
        if 'hotels_found' not in data or 'count' not in data:
            return {'passed': False, 'message': 'Missing required keys'}
        hotels = data['hotels_found']
        count = data['count']
        if not isinstance(hotels, list):
            return {'passed': False, 'message': 'hotels_found is not a list'}
        if not isinstance(count, str):
            return {'passed': False, 'message': 'count is not a string'}
        try:
            count_int = int(count)
        except ValueError:
            return {'passed': False, 'message': 'count cannot be converted to int'}
        if len(hotels) != count_int:
            return {'passed': False, 'message': 'Count mismatch'}
        if count_int == 0:
            return {'passed': True, 'message': 'No hotels found (valid)'}
        for hotel in hotels:
            if not isinstance(hotel, dict):
                return {'passed': False, 'message': 'Hotel entry is not a dict'}
            required = ['name', 'stars', 'price', 'amenities']
            for key in required:
                if key not in hotel:
                    return {'passed': False, 'message': f'Missing hotel key: {key}'}
                if not isinstance(hotel[key], str):
                    return {'passed': False, 'message': f'Hotel {key} is not a string'}
            if 'pool' not in hotel['amenities'].lower():
                return {'passed': False, 'message': 'Hotel missing pool amenity'}
        dir_hotels = tools['search_hotel_directory']('Paris')
        paris_pool_count = 0
        for h in dir_hotels:
            if isinstance(h, dict) and 'pool' in h.get('amenities', '').lower():
                paris_pool_count += 1
        if count_int > paris_pool_count:
            return {'passed': False, 'message': 'More hotels reported than exist in directory'}
        return {'passed': True, 'message': 'Verification passed'}
    except Exception as e:
        return {'passed': False, 'message': f'Exception during verification: {str(e)}'}
