def verify(tools, answer):
    import json
    import ast
    
    try:
        # Parse the answer string back to dictionary
        if isinstance(answer, str):
            try:
                answer_dict = ast.literal_eval(answer)
            except:
                # Try JSON parsing
                answer_dict = json.loads(answer)
        else:
            answer_dict = answer
        
        if not isinstance(answer_dict, dict):
            return {'passed': False, 'message': 'Answer is not a dictionary'}
        
        # Check required top-level keys
        required_keys = ['average_accommodation_cost_per_night', 
                        'top_rated_hotels_with_offers', 
                        'most_common_accommodation_type',
                        'data_summary']
        
        for key in required_keys:
            if key not in answer_dict:
                return {'passed': False, 'message': f'Missing required key: {key}'}
        
        # Verify average cost format
        avg_cost = answer_dict.get('average_accommodation_cost_per_night')
        if not isinstance(avg_cost, str):
            return {'passed': False, 'message': 'average_accommodation_cost_per_night must be string'}
        
        try:
            avg_cost_float = float(avg_cost)
            # Check if rounded to 2 decimals
            if len(avg_cost.split('.')[-1]) > 2:
                return {'passed': False, 'message': 'Average cost should be rounded to 2 decimals'}
        except ValueError:
            return {'passed': False, 'message': 'Average cost is not a valid float string'}
        
        # Verify hotels list
        hotels_list = answer_dict.get('top_rated_hotels_with_offers', [])
        if not isinstance(hotels_list, list):
            return {'passed': False, 'message': 'top_rated_hotels_with_offers must be a list'}
        
        # Verify each hotel entry
        required_hotel_keys = ['hotel_name', 'room_type', 'rate', 'guest_rating', 'special_offer']
        for i, hotel in enumerate(hotels_list):
            if not isinstance(hotel, dict):
                return {'passed': False, 'message': f'Hotel entry {i} is not a dictionary'}
            
            for key in required_hotel_keys:
                if key not in hotel:
                    return {'passed': False, 'message': f'Hotel entry {i} missing key: {key}'}
                
                value = hotel.get(key)
                if key == 'guest_rating':
                    if not isinstance(value, (int, float)):
                        return {'passed': False, 'message': f'Hotel entry {i} guest_rating must be numeric'}
                    if value < 4.5:
                        return {'passed': False, 'message': f'Hotel entry {i} guest_rating must be ≥4.5'}
                elif key == 'special_offer':
                    if not isinstance(value, str):
                        return {'passed': False, 'message': f'Hotel entry {i} special_offer must be string'}
                    if value.lower() == 'none' or not value.strip():
                        return {'passed': False, 'message': f'Hotel entry {i} has invalid special_offer'}
                else:
                    if not isinstance(value, str):
                        return {'passed': False, 'message': f'Hotel entry {i} key {key} must be string'}
                    if not value.strip():
                        return {'passed': False, 'message': f'Hotel entry {i} key {key} is empty'}
        
        # Verify most common accommodation type
        acc_type = answer_dict.get('most_common_accommodation_type')
        if not isinstance(acc_type, str):
            return {'passed': False, 'message': 'most_common_accommodation_type must be string'}
        
        # Verify data summary
        data_summary = answer_dict.get('data_summary', {})
        if not isinstance(data_summary, dict):
            return {'passed': False, 'message': 'data_summary must be a dictionary'}
        
        summary_keys = ['total_american_female_travelers_analyzed', 'hotels_with_high_ratings_found']
        for key in summary_keys:
            if key not in data_summary:
                return {'passed': False, 'message': f'data_summary missing key: {key}'}
            
            value = data_summary.get(key)
            if not isinstance(value, str):
                return {'passed': False, 'message': f'data_summary key {key} must be string'}
            
            try:
                int_value = int(value)
                if int_value < 0:
                    return {'passed': False, 'message': f'data_summary key {key} must be non-negative'}
            except ValueError:
                return {'passed': False, 'message': f'data_summary key {key} must be integer string'}
        
        # Cross-verify with actual data
        # Check travel recommendations for American female travelers
        travel_check = tools['search_travel_recommendations']('American female Paris')
        if isinstance(travel_check, list) and len(travel_check) > 0:
            # At least some data should exist
            pass
        
        # Check hotel pricing for high-rated hotels
        hotel_check = tools['search_hotel_pricing']('Available')
        if isinstance(hotel_check, list) and len(hotel_check) > 0:
            # Verify at least one hotel meets criteria
            high_rated_found = False
            for hotel in hotel_check:
                if isinstance(hotel, dict) and 'error' not in hotel:
                    rating = hotel.get('guest_rating', 0)
                    offer = hotel.get('special_offer', '')
                    avail = hotel.get('availability', '')
                    
                    if (isinstance(rating, (int, float)) and 
                        rating >= 4.5 and 
                        offer and offer.lower() != 'none' and 
                        avail == 'Available'):
                        high_rated_found = True
                        break
            
            if not high_rated_found and len(hotels_list) > 0:
                return {'passed': False, 'message': 'Hotels listed but none found in verification'}
        
        return {'passed': True, 'message': 'All verification checks passed'}
        
    except Exception as e:
        return {'passed': False, 'message': f'Verification exception: {str(e)}'}
