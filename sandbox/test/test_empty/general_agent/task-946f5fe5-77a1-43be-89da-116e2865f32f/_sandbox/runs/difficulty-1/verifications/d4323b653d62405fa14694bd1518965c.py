def verify(tools, answer):
    import json
    try:
        # Handle wrapped answer
        if isinstance(answer, dict) and 'submitted_data' in answer:
            data = answer.get('submitted_data')
        elif isinstance(answer, dict) and 'data' in answer:
            data = answer.get('data')
        else:
            data = answer
        
        if not isinstance(data, dict):
            return {'passed': False, 'message': 'Answer is not a dict'}
        
        # Check required structure
        if 'hotels' not in data or 'total_count' not in data:
            return {'passed': False, 'message': 'Missing required keys'}
        
        hotels = data.get('hotels')
        total_count = data.get('total_count')
        
        if not isinstance(hotels, list):
            return {'passed': False, 'message': 'Hotels must be a list'}
        
        if not isinstance(total_count, int):
            return {'passed': False, 'message': 'total_count must be integer'}
        
        if len(hotels) != total_count:
            return {'passed': False, 'message': 'Count mismatch'}
        
        # Verify each hotel has required fields
        for hotel in hotels:
            if not isinstance(hotel, dict):
                return {'passed': False, 'message': 'Hotel entry not dict'}
            required = ['name', 'stars', 'price', 'amenities']
            for field in required:
                if field not in hotel or not hotel[field]:
                    return {'passed': False, 'message': f'Missing {field}'}
            # Check amenities contain 'pool'
            amenities = hotel.get('amenities', '').lower()
            if 'pool' not in amenities:
                return {'passed': False, 'message': 'Pool not in amenities'}
        
        # Use a data tool to cross-check
        # Search for Paris hotels to verify at least some exist
        listings = tools['search_hotel_listings']('Paris')
        if isinstance(listings, list) and len(listings) > 0:
            # If we found hotels in data but answer has none, that's suspicious
            if total_count == 0 and len(listings) > 0:
                # Check if any have pool
                pool_found = False
                for hotel in listings:
                    if isinstance(hotel, dict):
                        amenities = hotel.get('amenities', '')
                        if amenities and 'pool' in amenities.lower():
                            pool_found = True
                            break
                if pool_found:
                    return {'passed': False, 'message': 'Should have found pool hotels'}
        
        # If we got here, verification passed
        return {'passed': True, 'message': 'Verification successful', 'details': {'hotels_count': total_count}}
    except Exception as e:
        return {'passed': False, 'message': f'Exception: {str(e)}'}
