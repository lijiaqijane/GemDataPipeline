def verify(tools, answer):
    try:
        import json
        
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
            return {'passed': False, 'message': f'Hotel stars must be \'4\', got {hotel.get("stars")}'}
        
        amenities = str(hotel.get('amenities', '')).lower()
        if 'pool' not in amenities or 'wifi' not in amenities:
            return {'passed': False, 'message': 'Hotel must have pool and wifi amenities'}
        
        city = str(hotel.get('city', '')).lower()
        if 'paris' not in city:
            return {'passed': False, 'message': 'Hotel must be in Paris'}
        
        # Verify Airbnb experience criteria
        airbnb = answer['airbnb_experience']
        if not isinstance(airbnb, dict):
            return {'passed': False, 'message': 'Airbnb experience is not a dict'}
        
        try:
            rating = float(str(airbnb.get('rating', '0')).strip())
            if rating < 4.5:
                return {'passed': False, 'message': f'Airbnb rating must be >= 4.5, got {rating}'}
        except:
            return {'passed': False, 'message': 'Invalid Airbnb rating format'}
        
        location = str(airbnb.get('location', '')).lower()
        if 'paris' not in location:
            return {'passed': False, 'message': 'Airbnb experience must be in Paris'}
        
        # Verify attraction criteria
        attraction = answer['attraction']
        if not isinstance(attraction, dict):
            return {'passed': False, 'message': 'Attraction is not a dict'}
        
        entry_fee = str(attraction.get('entry_fee', '')).lower()
        import re
        fee_match = re.search(r'\$?\s*(\d+\.?\d*)', entry_fee)
        if fee_match:
            try:
                fee = float(fee_match.group(1))
                if fee >= 30:
                    return {'passed': False, 'message': f'Attraction entry fee must be < $30, got ${fee}'}
            except:
                pass
        
        location = str(attraction.get('location', '')).lower()
        if 'paris' not in location:
            return {'passed': False, 'message': 'Attraction must be in Paris'}
        
        # Verify historical travelers criteria
        travelers = answer['historical_travelers']
        if not isinstance(travelers, list):
            return {'passed': False, 'message': 'Historical travelers must be a list'}
        
        if len(travelers) < 3:
            return {'passed': False, 'message': f'Must have at least 3 historical travelers, got {len(travelers)}'}
        
        for i, traveler in enumerate(travelers):
            if not isinstance(traveler, dict):
                return {'passed': False, 'message': f'Traveler {i} is not a dict'}
            
            if str(traveler.get('nationality', '')).lower() != 'american':
                return {'passed': False, 'message': f'Traveler {i} must be American'}
            
            destination = str(traveler.get('destination', '')).lower()
            if 'paris' not in destination:
                return {'passed': False, 'message': f'Traveler {i} destination must include Paris'}
            
            if str(traveler.get('accommodation_type', '')).lower() != 'hotel':
                return {'passed': False, 'message': f'Traveler {i} accommodation must be Hotel'}
        
        # Verify verification summary is meaningful
        summary = answer.get('verification_summary', '')
        if not summary or len(summary.strip()) < 50:
            return {'passed': False, 'message': 'Verification summary is too short or empty'}
        
        # Cross-check with tool call
        hotels_check = tools['search_hotels']('Paris 4 star', 10)
        if not isinstance(hotels_check, list):
            return {'passed': False, 'message': 'Hotel search tool returned invalid data'}
        
        # Check that answer contains meaningful content
        if not hotel.get('name') or not hotel.get('price'):
            return {'passed': False, 'message': 'Hotel name or price is empty'}
        
        if not airbnb.get('title') or not airbnb.get('price'):
            return {'passed': False, 'message': 'Airbnb title or price is empty'}
        
        if not attraction.get('name') or not attraction.get('entry_fee'):
            return {'passed': False, 'message': 'Attraction name or entry fee is empty'}
        
        for traveler in travelers:
            if not traveler.get('name'):
                return {'passed': False, 'message': 'Traveler name is empty'}
        
        return {'passed': True, 'message': 'All criteria verified successfully'}
        
    except Exception as e:
        return {'passed': False, 'message': f'Verification error: {str(e)}'}
