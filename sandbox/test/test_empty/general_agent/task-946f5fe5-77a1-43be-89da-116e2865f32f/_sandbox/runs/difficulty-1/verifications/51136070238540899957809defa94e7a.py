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
        
        if not isinstance(total_count, str):
            return {'passed': False, 'message': 'total_count must be string'}
        
        # Verify total_count matches actual count
        if not total_count.isdigit():
            return {'passed': False, 'message': 'total_count must be numeric string'}
        
        if int(total_count) != len(hotels):
            return {'passed': False, 'message': 'Count mismatch'}
        
        # Verify each hotel has required string fields
        for hotel in hotels:
            if not isinstance(hotel, dict):
                return {'passed': False, 'message': 'Hotel entry not a dict'}
            required = ['name', 'stars', 'price', 'amenities']
            for key in required:
                if key not in hotel:
                    return {'passed': False, 'message': f'Missing hotel field: {key}'}
                if not isinstance(hotel[key], str):
                    return {'passed': False, 'message': f'Hotel {key} must be string'}
        
        # Cross-check with tool data
        listings = tools['search_hotel_listings']('Paris')
        if isinstance(listings, list) and hotels:
            # Verify at least one hotel from answer exists in tool data
            found = False
            for hotel in hotels:
                for listing in listings:
                    if isinstance(listing, dict):
                        if listing.get('name') == hotel['name']:
                            found = True
                            break
                if found:
                    break
            if not found and len(hotels) > 0:
                return {'passed': False, 'message': 'Hotels not found in tool data'}
        
        return {'passed': True, 'message': 'Format and data validation passed'}
    except Exception as e:
        return {'passed': False, 'message': f'Verification error: {str(e)}'}
