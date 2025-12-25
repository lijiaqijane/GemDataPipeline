def verify(tools, answer):
    import json
    
    try:
        # Handle wrapped answer
        if isinstance(answer, dict):
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
        
        # Check required keys
        required_keys = [
            'adr_premium_percentage',
            'most_expensive_accommodation_type',
            'premium_hotels_below_75th_percentile',
            'rating_rate_correlation',
            'data_quality_notes'
        ]
        
        for key in required_keys:
            if key not in answer_data:
                return {'passed': False, 'message': f'Missing required key: {key}'}
        
        # Check data quality notes structure
        notes = answer_data.get('data_quality_notes', {})
        if not isinstance(notes, dict):
            return {'passed': False, 'message': 'data_quality_notes is not a dictionary'}
        
        required_notes = [
            'high_rated_hotels_analyzed',
            'low_rated_hotels_analyzed',
            'american_traveler_trips_analyzed',
            'total_hotels_in_correlation'
        ]
        
        for note_key in required_notes:
            if note_key not in notes:
                return {'passed': False, 'message': f'Missing data quality note: {note_key}'}
        
        # Check non-empty values
        adr_premium = answer_data.get('adr_premium_percentage')
        if adr_premium is None:
            return {'passed': False, 'message': 'adr_premium_percentage is None'}
        
        acc_type = answer_data.get('most_expensive_accommodation_type')
        if not acc_type or not isinstance(acc_type, str) or acc_type.strip() == '':
            return {'passed': False, 'message': 'most_expensive_accommodation_type is empty or invalid'}
        
        correlation = answer_data.get('rating_rate_correlation')
        if correlation is None:
            return {'passed': False, 'message': 'rating_rate_correlation is None'}
        
        # Check premium hotels list
        premium_hotels = answer_data.get('premium_hotels_below_75th_percentile', [])
        if not isinstance(premium_hotels, list):
            return {'passed': False, 'message': 'premium_hotels_below_75th_percentile is not a list'}
        
        # Verify with actual data using a tool
        hotel_data = tools['search_hotel_pricing']('')
        
        if isinstance(hotel_data, list) and len(hotel_data) > 0:
            # Check that we have some data to verify against
            hotel_count = 0
            for item in hotel_data:
                if isinstance(item, dict) and 'hotel_name' in item:
                    hotel_count += 1
            
            if hotel_count == 0:
                return {'passed': False, 'message': 'No valid hotel data found to verify against'}
            
            # Check that premium hotels list is not empty when there should be data
            if hotel_count > 0 and len(premium_hotels) == 0:
                # This might be okay if no hotels meet criteria, but check notes
                high_rated = notes.get('high_rated_hotels_analyzed', 0)
                if high_rated > 0:
                    return {'passed': False, 'message': 'Premium hotels list is empty but high-rated hotels exist'}
        
        # Check data quality notes values
        high_rated = notes.get('high_rated_hotels_analyzed', 0)
        low_rated = notes.get('low_rated_hotels_analyzed', 0)
        american_trips = notes.get('american_traveler_trips_analyzed', 0)
        total_hotels = notes.get('total_hotels_in_correlation', 0)
        
        if not isinstance(high_rated, (int, float)) or high_rated < 0:
            return {'passed': False, 'message': 'Invalid high_rated_hotels_analyzed value'}
        
        if not isinstance(low_rated, (int, float)) or low_rated < 0:
            return {'passed': False, 'message': 'Invalid low_rated_hotels_analyzed value'}
        
        if not isinstance(american_trips, (int, float)) or american_trips < 0:
            return {'passed': False, 'message': 'Invalid american_traveler_trips_analyzed value'}
        
        if not isinstance(total_hotels, (int, float)) or total_hotels < 0:
            return {'passed': False, 'message': 'Invalid total_hotels_in_correlation value'}
        
        # Check correlation value range
        if not isinstance(correlation, (int, float)):
            return {'passed': False, 'message': 'rating_rate_correlation is not numeric'}
        
        if correlation < -1.0 or correlation > 1.0:
            return {'passed': False, 'message': 'rating_rate_correlation out of valid range [-1, 1]'}
        
        # Check premium hotels structure
        for hotel in premium_hotels:
            if not isinstance(hotel, dict):
                return {'passed': False, 'message': 'Premium hotel entry is not a dictionary'}
            
            required_hotel_keys = ['hotel_name', 'guest_rating', 'rate', 'special_offer']
            for key in required_hotel_keys:
                if key not in hotel:
                    return {'passed': False, 'message': f'Premium hotel missing key: {key}'}
            
            # Check non-empty values
            if not hotel.get('hotel_name') or not isinstance(hotel['hotel_name'], str):
                return {'passed': False, 'message': 'Hotel name is empty or invalid'}
            
            if not isinstance(hotel.get('guest_rating'), (int, float)):
                return {'passed': False, 'message': 'Guest rating is not numeric'}
            
            if not isinstance(hotel.get('rate'), (int, float)):
                return {'passed': False, 'message': 'Rate is not numeric'}
            
            if not hotel.get('special_offer') or not isinstance(hotel['special_offer'], str):
                return {'passed': False, 'message': 'Special offer is empty or invalid'}
        
        return {'passed': True, 'message': 'All verification checks passed'}
        
    except Exception as e:
        return {'passed': False, 'message': f'Verification error: {str(e)}'}
