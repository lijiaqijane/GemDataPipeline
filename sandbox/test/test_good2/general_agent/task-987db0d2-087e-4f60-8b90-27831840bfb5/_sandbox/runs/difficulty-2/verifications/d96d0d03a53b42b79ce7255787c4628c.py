def verify(tools, answer):
    import json
    
    try:
        # Handle answer that might be wrapped by submit_result
        if isinstance(answer, dict):
            # Check if answer is wrapped in submit_result format
            if 'status' in answer and 'data' in answer:
                answer_data = answer.get('data')
            elif 'submitted_data' in answer:
                answer_data = answer.get('submitted_data')
            else:
                answer_data = answer
        else:
            return {'passed': False, 'message': 'Answer is not a dictionary'}
        
        if not isinstance(answer_data, dict):
            return {'passed': False, 'message': 'Answer data is not a dictionary'}
        
        # Check required top-level keys
        required_keys = ['average_accommodation_cost_per_night', 
                        'top_rated_hotels_with_offers', 
                        'most_common_accommodation_type',
                        'data_summary']
        
        for key in required_keys:
            if key not in answer_data:
                return {'passed': False, 'message': f'Missing required key: {key}'}
        
        # Verify average cost format
        avg_cost_str = answer_data.get('average_accommodation_cost_per_night')
        if not isinstance(avg_cost_str, str):
            return {'passed': False, 'message': 'average_accommodation_cost_per_night must be a string'}
        
        try:
            avg_cost = float(avg_cost_str)
            if avg_cost <= 0:
                return {'passed': False, 'message': 'Average cost must be positive'}
        except ValueError:
            return {'passed': False, 'message': 'Average cost must be a valid float string'}
        
        # Verify top_rated_hotels_with_offers
        hotels = answer_data.get('top_rated_hotels_with_offers')
        if not isinstance(hotels, list):
            return {'passed': False, 'message': 'top_rated_hotels_with_offers must be a list'}
        
        if len(hotels) == 0:
            return {'passed': False, 'message': 'top_rated_hotels_with_offers list cannot be empty'}
        
        required_hotel_keys = ['hotel_name', 'room_type', 'rate', 'guest_rating', 'special_offer']
        for hotel in hotels:
            if not isinstance(hotel, dict):
                return {'passed': False, 'message': 'Each hotel must be a dictionary'}
            
            for key in required_hotel_keys:
                if key not in hotel:
                    return {'passed': False, 'message': f'Hotel missing required key: {key}'}
            
            # Check guest rating
            rating = hotel.get('guest_rating')
            if not isinstance(rating, (int, float)):
                return {'passed': False, 'message': 'guest_rating must be numeric'}
            
            if rating < 4.5:
                return {'passed': False, 'message': 'guest_rating must be ≥ 4.5'}
            
            # Check special offer is not empty
            special_offer = hotel.get('special_offer')
            if not special_offer or str(special_offer).strip() == '':
                return {'passed': False, 'message': 'special_offer cannot be empty'}
            
            # Check hotel name is not empty
            hotel_name = hotel.get('hotel_name')
            if not hotel_name or str(hotel_name).strip() == '':
                return {'passed': False, 'message': 'hotel_name cannot be empty'}
        
        # Verify most_common_accommodation_type
        acc_type = answer_data.get('most_common_accommodation_type')
        if not isinstance(acc_type, str) or not acc_type.strip():
            return {'passed': False, 'message': 'most_common_accommodation_type must be a non-empty string'}
        
        # Verify data_summary
        data_summary = answer_data.get('data_summary')
        if not isinstance(data_summary, dict):
            return {'passed': False, 'message': 'data_summary must be a dictionary'}
        
        required_summary_keys = ['total_american_female_travelers_analyzed', 'hotels_with_high_ratings_found']
        for key in required_summary_keys:
            if key not in data_summary:
                return {'passed': False, 'message': f'data_summary missing required key: {key}'}
            
            value = data_summary.get(key)
            if not isinstance(value, int):
                return {'passed': False, 'message': f'{key} must be an integer'}
            
            if value <= 0:
                return {'passed': False, 'message': f'{key} must be positive'}
        
        # Cross-check with a data tool call
        # Use search_travel_recommendations to verify data exists
        check_data = tools['search_travel_recommendations']('sample')
        
        # The check is about calling the tool, not about the specific data returned
        # Just verify the tool was called successfully
        if check_data is None:
            return {'passed': False, 'message': 'Tool call verification failed'}
        
        # All checks passed
        return {'passed': True, 'message': 'All verification checks passed'}
        
    except Exception as e:
        # Return failure with error message
        return {'passed': False, 'message': f'Verification error: {str(e)}'}
