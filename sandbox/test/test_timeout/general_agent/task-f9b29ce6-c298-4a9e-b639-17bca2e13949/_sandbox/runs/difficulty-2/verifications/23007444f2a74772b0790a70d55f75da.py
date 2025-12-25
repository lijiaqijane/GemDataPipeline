def verify(tools, answer):
    try:
        # Handle wrapped answer
        if answer is None:
            return {'passed': False, 'message': 'Answer is None'}
        
        if isinstance(answer, dict):
            if 'status' in answer and 'data' in answer:
                answer = answer['data']
            elif 'submitted_data' in answer:
                answer = answer['submitted_data']
            elif 'data' in answer:
                answer = answer['data']
        
        if not isinstance(answer, dict):
            return {'passed': False, 'message': 'Answer is not a dict'}
        
        # Check required keys
        required_keys = ['package_name', 'hotel', 'airbnb_experience', 'attraction', 'historical_travelers', 'verification_summary']
        for key in required_keys:
            if key not in answer:
                return {'passed': False, 'message': f'Missing required key: {key}'}
        
        # Verify hotel criteria
        hotel = answer['hotel']
        if not isinstance(hotel, dict):
            return {'passed': False, 'message': 'Hotel is not a dict'}
        
        if hotel.get('stars') != '4':
            return {'passed': False, 'message': f'Hotel stars is not 4: {hotel.get("stars")}'}
        
        if 'paris' not in str(hotel.get('city', '')).lower():
            return {'passed': False, 'message': f'Hotel city is not Paris: {hotel.get("city")}'}
        
        amenities = str(hotel.get('amenities', '')).lower()
        if 'pool' not in amenities or 'wifi' not in amenities:
            return {'passed': False, 'message': f'Hotel missing required amenities (pool and wifi): {amenities}'}
        
        # Verify Airbnb experience criteria
        airbnb = answer['airbnb_experience']
        if not isinstance(airbnb, dict):
            return {'passed': False, 'message': 'Airbnb experience is not a dict'}
        
        if 'paris' not in str(airbnb.get('location', '')).lower():
            return {'passed': False, 'message': f'Airbnb location is not Paris: {airbnb.get("location")}'}
        
        try:
            rating = float(str(airbnb.get('rating', '0')))
            if rating < 4.5:
                return {'passed': False, 'message': f'Airbnb rating {rating} is less than 4.5'}
        except:
            return {'passed': False, 'message': f'Invalid Airbnb rating: {airbnb.get("rating")}'}
        
        # Verify attraction criteria
        attraction = answer['attraction']
        if not isinstance(attraction, dict):
            return {'passed': False, 'message': 'Attraction is not a dict'}
        
        if 'paris' not in str(attraction.get('location', '')).lower():
            return {'passed': False, 'message': f'Attraction location is not Paris: {attraction.get("location")}'}
        
        entry_fee = str(attraction.get('entry_fee', '$100'))
        import re
        fee_match = re.search(r'\$?([0-9]+\.?[0-9]*)', entry_fee)
        if fee_match:
            fee_value = float(fee_match.group(1))
            if fee_value >= 30:
                return {'passed': False, 'message': f'Attraction entry fee ${fee_value} is not under $30'}
        else:
            return {'passed': False, 'message': f'Could not parse attraction entry fee: {entry_fee}'}
        
        # Verify historical travelers criteria
        travelers = answer['historical_travelers']
        if not isinstance(travelers, list):
            return {'passed': False, 'message': 'Historical travelers is not a list'}
        
        if len(travelers) < 3:
            return {'passed': False, 'message': f'Only {len(travelers)} historical travelers found (need 3)'}
        
        for i, traveler in enumerate(travelers):
            if not isinstance(traveler, dict):
                return {'passed': False, 'message': f'Traveler {i} is not a dict'}
            
            if 'american' not in str(traveler.get('nationality', '')).lower():
                return {'passed': False, 'message': f'Traveler {i} nationality is not American: {traveler.get("nationality")}'}
            
            if 'paris' not in str(traveler.get('destination', '')).lower():
                return {'passed': False, 'message': f'Traveler {i} destination is not Paris: {traveler.get("destination")}'}
            
            if 'hotel' not in str(traveler.get('accommodation_type', '')).lower():
                return {'passed': False, 'message': f'Traveler {i} accommodation is not hotel: {traveler.get("accommodation_type")}'}
        
        # Verify summary exists
        summary = answer['verification_summary']
        if not isinstance(summary, str) or not summary.strip():
            return {'passed': False, 'message': 'Verification summary is missing or empty'}
        
        # Cross-check with actual data sources
        # Verify hotel exists in dataset
        hotels_data = tools['search_hotels']('Paris', 20)
        hotel_found = False
        for h in hotels_data:
            if isinstance(h, dict) and h.get('name') == hotel.get('name'):
                hotel_found = True
                break
        
        if not hotel_found and hotel.get('name') != 'No suitable hotel found':
            return {'passed': False, 'message': f'Hotel {hotel.get("name")} not found in dataset'}
        
        return {
            'passed': True, 
            'message': 'All package criteria verified successfully', 
            'details': {
                'hotel_verified': hotel_found or hotel.get('name') == 'No suitable hotel found',
                'airbnb_rating_verified': rating >= 4.5,
                'attraction_fee_verified': fee_value < 30,
                'historical_travelers_count': len(travelers)
            }
        }
        
    except Exception as e:
        return {'passed': False, 'message': f'Verification error: {str(e)}'}
