def verify(tools, answer):
    try:
        import json
        import re
        
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
        
        # Check required top-level keys
        required_keys = ['analysis_summary', 'available_luxury_hotels', 'top_attractions', 
                        'historical_data', 'recommended_package']
        for key in required_keys:
            if key not in data:
                return {'passed': False, 'message': f'Missing key: {key}'}
        
        # Verify hotels structure
        if not isinstance(data['available_luxury_hotels'], list):
            return {'passed': False, 'message': 'Hotels not a list'}
        
        for hotel in data['available_luxury_hotels']:
            if not isinstance(hotel, dict):
                return {'passed': False, 'message': 'Hotel entry not a dict'}
            
            hotel_keys = ['name', 'stars', 'amenities', 'seasonal_rate', 
                         'special_offer', '3_night_cost', 'booked_in_past_year']
            for key in hotel_keys:
                if key not in hotel:
                    return {'passed': False, 'message': f'Hotel missing key: {key}'}
            
            # Verify 5-star rating
            if hotel['stars'] != '5':
                return {'passed': False, 'message': f'Hotel {hotel["name"]} not 5-star'}
            
            # Verify amenities contain pool and jacuzzi
            amenities = hotel.get('amenities', '').lower()
            if 'pool' not in amenities or 'jacuzzi' not in amenities:
                return {'passed': False, 'message': f'Hotel {hotel["name"]} missing required amenities'}
        
        # Verify attractions structure
        if not isinstance(data['top_attractions'], list):
            return {'passed': False, 'message': 'Attractions not a list'}
        
        for attr in data['top_attractions']:
            if not isinstance(attr, dict):
                return {'passed': False, 'message': 'Attraction entry not a dict'}
            
            attr_keys = ['name', 'description', 'rating', 'entry_fee', 'cost_for_2']
            for key in attr_keys:
                if key not in attr:
                    return {'passed': False, 'message': f'Attraction missing key: {key}'}
            
            # Verify rating >= 4.5
            try:
                rating = float(attr['rating'])
                if rating < 4.5:
                    return {'passed': False, 'message': f'Attraction rating {rating} < 4.5'}
            except:
                return {'passed': False, 'message': f'Invalid rating: {attr["rating"]}'}
            
            # Verify entry fee exists and not free
            entry_fee = attr['entry_fee'].lower()
            if not entry_fee or 'free' in entry_fee:
                return {'passed': False, 'message': f'Attraction has free or no entry fee'}
        
        # Verify historical data structure
        hist_keys = ['average_hotel_nightly_cost', 'total_paris_hotel_trips', 'matching_hotels_booked']
        for key in hist_keys:
            if key not in data['historical_data']:
                return {'passed': False, 'message': f'Historical data missing key: {key}'}
        
        # Verify package structure
        pkg_keys = ['package_name', 'selected_hotel', 'selected_attractions', 'cost_breakdown']
        for key in pkg_keys:
            if key not in data['recommended_package']:
                return {'passed': False, 'message': f'Package missing key: {key}'}
        
        cost_keys = ['hotel_3_nights', 'attractions_2_people', 'total_estimated']
        for key in cost_keys:
            if key not in data['recommended_package']['cost_breakdown']:
                return {'passed': False, 'message': f'Cost breakdown missing key: {key}'}
        
        # Verify cost formatting
        for cost_key in cost_keys:
            cost_val = data['recommended_package']['cost_breakdown'][cost_key]
            if cost_val and not (cost_val.startswith('$') or cost_val == 'Not available'):
                return {'passed': False, 'message': f'Cost {cost_key} not properly formatted: {cost_val}'}
        
        # Verify data integration by checking tools were used
        hotels_dir = tools['search_hotel_directory']('Paris')
        if not hotels_dir:
            return {'passed': False, 'message': 'No hotel directory data found'}
        
        pricing = tools['search_hotel_pricing']('Paris')
        attractions = tools['search_tripadvisor_attractions']('Paris')
        trips = tools['search_travel_recommendations']('Paris')
        
        # Basic verification that multiple data sources were accessed
        if not (hotels_dir and attractions):
            return {'passed': False, 'message': 'Insufficient data sources accessed'}
        
        return {
            'passed': True, 
            'message': 'Verification passed', 
            'details': {
                'hotels_count': len(data['available_luxury_hotels']),
                'attractions_count': len(data['top_attractions']),
                'historical_trips': data['historical_data']['total_paris_hotel_trips']
            }
        }
        
    except Exception as e:
        return {'passed': False, 'message': f'Verification exception: {str(e)}'}
