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
        
        hotel_required = ['name', 'stars', 'amenities', 'seasonal_rate', 'special_offer', '3_night_cost', 'booked_in_past_year']
        for hotel in data['available_luxury_hotels']:
            if not isinstance(hotel, dict):
                return {'passed': False, 'message': 'Hotel entry not a dict'}
            for key in hotel_required:
                if key not in hotel:
                    return {'passed': False, 'message': f'Hotel missing key: {key}'}
            if hotel['stars'] != '5':
                return {'passed': False, 'message': 'Hotel not 5-star'}
            amenities = hotel.get('amenities', '').lower()
            if 'pool' not in amenities or 'jacuzzi' not in amenities:
                return {'passed': False, 'message': 'Hotel missing required amenities'}
        
        # Verify attractions structure
        if not isinstance(data['top_attractions'], list):
            return {'passed': False, 'message': 'Attractions not a list'}
        
        attr_required = ['name', 'description', 'rating', 'entry_fee', 'cost_for_2']
        for attr in data['top_attractions']:
            if not isinstance(attr, dict):
                return {'passed': False, 'message': 'Attraction entry not a dict'}
            for key in attr_required:
                if key not in attr:
                    return {'passed': False, 'message': f'Attraction missing key: {key}'}
            try:
                rating = float(attr['rating'])
                if rating < 4.5:
                    return {'passed': False, 'message': 'Attraction rating below 4.5'}
            except:
                pass
            if not attr['entry_fee'] or 'free' in attr['entry_fee'].lower():
                return {'passed': False, 'message': 'Attraction has no entry fee or is free'}
        
        # Verify historical data
        hist = data['historical_data']
        if not isinstance(hist, dict):
            return {'passed': False, 'message': 'Historical data not a dict'}
        hist_keys = ['average_hotel_nightly_cost', 'total_paris_hotel_trips', 'matching_hotels_booked']
        for key in hist_keys:
            if key not in hist:
                return {'passed': False, 'message': f'Historical missing key: {key}'}
        
        # Verify package
        pkg = data['recommended_package']
        if not isinstance(pkg, dict):
            return {'passed': False, 'message': 'Package not a dict'}
        pkg_keys = ['package_name', 'selected_hotel', 'selected_attractions', 'cost_breakdown']
        for key in pkg_keys:
            if key not in pkg:
                return {'passed': False, 'message': f'Package missing key: {key}'}
        
        cost = pkg['cost_breakdown']
        cost_keys = ['hotel_3_nights', 'attractions_2_people', 'total_estimated']
        for key in cost_keys:
            if key not in cost:
                return {'passed': False, 'message': f'Cost breakdown missing key: {key}'}
        
        # Cross-check with actual data sources
        # Check hotel directory data
        hotels_check = tools['search_hotel_directory']('sample')
        if hotels_check and isinstance(hotels_check, list) and len(hotels_check) > 0:
            found_hotels = False
            for h in hotels_check:
                if isinstance(h, dict) and h.get('city', '').lower() == 'paris':
                    found_hotels = True
                    break
            if not found_hotels and len(data['available_luxury_hotels']) > 0:
                return {'passed': False, 'message': 'No Paris hotels found in directory data'}
        
        # Check attractions data
        attractions_check = tools['search_tripadvisor_attractions']('sample')
        if attractions_check and isinstance(attractions_check, list) and len(attractions_check) > 0:
            found_attractions = False
            for a in attractions_check:
                if isinstance(a, dict) and a.get('country', '').lower() == 'france':
                    found_attractions = True
                    break
            if not found_attractions and len(data['top_attractions']) > 0:
                return {'passed': False, 'message': 'No France attractions found in TripAdvisor data'}
        
        # Check travel recommendations
        travel_check = tools['search_travel_recommendations']('sample')
        if travel_check and isinstance(travel_check, list) and len(travel_check) > 0:
            found_travel = False
            for t in travel_check:
                if isinstance(t, dict) and 'hotel' in t.get('accommodation_type', '').lower():
                    found_travel = True
                    break
            if not found_travel and data['historical_data']['total_paris_hotel_trips'] > 0:
                return {'passed': False, 'message': 'No hotel trips found in travel data'}
        
        # Final validation
        if len(data['available_luxury_hotels']) == 0:
            return {'passed': False, 'message': 'No luxury hotels found'}
        
        return {'passed': True, 'message': 'All verification checks passed'}
    
    except Exception as e:
        return {'passed': False, 'message': f'Verification error: {str(e)}'}
