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
        if not isinstance(count, int):
            return {'passed': False, 'message': 'count is not an integer'}
        if len(hotels) != count:
            return {'passed': False, 'message': 'Count mismatch'}
        if count == 0:
            return {'passed': True, 'message': 'No hotels found (empty result)'}
        for hotel in hotels:
            if not isinstance(hotel, dict):
                return {'passed': False, 'message': 'Hotel entry is not a dict'}
            required = ['name', 'stars', 'price', 'amenities']
            for key in required:
                if key not in hotel:
                    return {'passed': False, 'message': f'Missing key {key} in hotel'}
                if not isinstance(hotel[key], str):
                    return {'passed': False, 'message': f'Hotel {key} is not a string'}
            if 'pool' not in hotel['amenities'].lower():
                return {'passed': False, 'message': 'Hotel missing pool in amenities'}
        dir_results = tools['search_hotel_directory']('Paris')
        if not isinstance(dir_results, list):
            return {'passed': False, 'message': 'Directory search failed'}
        paris_hotels = []
        for rec in dir_results:
            if isinstance(rec, dict):
                city = rec.get('city', '')
                if city and 'paris' in city.lower():
                    paris_hotels.append(rec)
        pool_in_dir = 0
        for rec in paris_hotels:
            amenities = rec.get('amenities', '')
            if 'pool' in amenities.lower():
                pool_in_dir += 1
        if pool_in_dir == 0 and count > 0:
            return {'passed': False, 'message': 'Directory shows no pool hotels but answer has some'}
        if pool_in_dir > 0 and count == 0:
            return {'passed': False, 'message': 'Directory shows pool hotels but answer has none'}
        return {'passed': True, 'message': 'Verification passed', 'details': {'found': count, 'directory_pool_count': pool_in_dir}}
    except Exception as e:
        return {'passed': False, 'message': f'Exception during verification: {str(e)}'}
