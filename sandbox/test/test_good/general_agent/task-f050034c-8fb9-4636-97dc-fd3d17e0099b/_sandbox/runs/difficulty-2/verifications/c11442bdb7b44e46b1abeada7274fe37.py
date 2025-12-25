def verify(tools, answer):
    try:
        import json
        if answer is None:
            return {'passed': False, 'message': 'Answer is None'}
        data = answer
        if isinstance(answer, dict):
            if 'status' in answer and 'submitted_data' in answer:
                data = answer['submitted_data']
            elif 'status' in answer and 'data' in answer:
                data = answer['data']
        if not isinstance(data, dict):
            return {'passed': False, 'message': 'Answer data is not a dict'}
        required_keys = ['package_name', 'hotels', 'attractions', 'cost_summary']
        for key in required_keys:
            if key not in data:
                return {'passed': False, 'message': f'Missing key: {key}'}
        if not isinstance(data['hotels'], list) or not isinstance(data['attractions'], list):
            return {'passed': False, 'message': 'Hotels or attractions not a list'}
        cost_keys = ['total_hotel_3_nights', 'total_attractions_2_people', 'grand_total']
        for key in cost_keys:
            if key not in data['cost_summary']:
                return {'passed': False, 'message': f'Missing cost key: {key}'}
        hotels = tools['search_hotel_directory']('Paris')
        paris_hotel_count = len([h for h in hotels if isinstance(h, dict) and h.get('city', '').lower() == 'paris'])
        if paris_hotel_count == 0:
            return {'passed': False, 'message': 'No Paris hotels found in directory'}
        if len(data['hotels']) == 0:
            return {'passed': False, 'message': 'No hotels in package'}
        for hotel in data['hotels']:
            if not isinstance(hotel, dict):
                return {'passed': False, 'message': 'Hotel entry not a dict'}
            hotel_keys = ['name', 'stars', 'amenities', 'seasonal_rate', '3_night_cost']
            for key in hotel_keys:
                if key not in hotel:
                    return {'passed': False, 'message': f'Hotel missing key: {key}'}
            if hotel['stars'] != '5':
                return {'passed': False, 'message': f'Hotel {hotel["name"]} not 5-star'}
        for attr in data['attractions']:
            if not isinstance(attr, dict):
                return {'passed': False, 'message': 'Attraction entry not a dict'}
            attr_keys = ['name', 'description', 'rating', 'entry_fee', 'cost_for_2']
            for key in attr_keys:
                if key not in attr:
                    return {'passed': False, 'message': f'Attraction missing key: {key}'}
            try:
                rating = float(attr['rating'])
                if rating < 4.5:
                    return {'passed': False, 'message': f'Attraction rating {rating} < 4.5'}
            except:
                return {'passed': False, 'message': f'Invalid rating: {attr["rating"]}'}
        total_hotel = data['cost_summary']['total_hotel_3_nights']
        total_attr = data['cost_summary']['total_attractions_2_people']
        grand = data['cost_summary']['grand_total']
        if not total_hotel.startswith('$') or not total_attr.startswith('$') or not grand.startswith('$'):
            return {'passed': False, 'message': 'Costs not formatted as currency'}
        return {'passed': True, 'message': 'Verification passed', 'details': {'hotels_count': len(data['hotels']), 'attractions_count': len(data['attractions'])}}
    except Exception as e:
        return {'passed': False, 'message': f'Verification exception: {str(e)}'}
