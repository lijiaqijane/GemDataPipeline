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
            return {'passed': False, 'message': 'data_quality_notes must be a dictionary'}
        
        required_note_keys = [
            'high_rated_hotels_analyzed',
            'low_rated_hotels_analyzed',
            'american_traveler_trips_analyzed',
            'total_hotels_in_correlation'
        ]
        
        for key in required_note_keys:
            if key not in notes:
                return {'passed': False, 'message': f'Missing data_quality_notes key: {key}'}
            
            # Check that values are strings
            if not isinstance(notes[key], str):
                return {'passed': False, 'message': f'data_quality_notes[{key}] must be a string'}
        
        # Check field types as per format specification
        adr_premium = answer_data.get('adr_premium_percentage')
        if not isinstance(adr_premium, str):
            return {'passed': False, 'message': 'adr_premium_percentage must be a string'}
        
        # Try to parse as float to validate format
        try:
            float_val = float(adr_premium)
            # Check if it's properly formatted with 1 decimal
            if '.' in adr_premium:
                decimal_part = adr_premium.split('.')[1]
                if len(decimal_part) != 1:
                    return {'passed': False, 'message': 'adr_premium_percentage must have exactly 1 decimal place'}
        except ValueError:
            return {'passed': False, 'message': 'adr_premium_percentage must be a valid numeric string'}
        
        accommodation_type = answer_data.get('most_expensive_accommodation_type')
        if not isinstance(accommodation_type, str):
            return {'passed': False, 'message': 'most_expensive_accommodation_type must be a string'}
        
        if not accommodation_type:
            return {'passed': False, 'message': 'most_expensive_accommodation_type cannot be empty'}
        
        correlation = answer_data.get('rating_rate_correlation')
        if not isinstance(correlation, str):
            return {'passed': False, 'message': 'rating_rate_correlation must be a string'}
        
        # Validate correlation format
        try:
            corr_val = float(correlation)
            if not (-1.0 <= corr_val <= 1.0):
                return {'passed': False, 'message': 'rating_rate_correlation must be between -1.0 and 1.0'}
            
            # Check decimal places
            if '.' in correlation:
                decimal_part = correlation.split('.')[1]
                if len(decimal_part) != 3:
                    return {'passed': False, 'message': 'rating_rate_correlation must have exactly 3 decimal places'}
        except ValueError:
            return {'passed': False, 'message': 'rating_rate_correlation must be a valid numeric string'}
        
        premium_hotels = answer_data.get('premium_hotels_below_75th_percentile')
        if not isinstance(premium_hotels, list):
            return {'passed': False, 'message': 'premium_hotels_below_75th_percentile must be a list'}
        
        # Check that list is not empty (meaningful content requirement)
        if not premium_hotels:
            return {'passed': False, 'message': 'premium_hotels_below_75th_percentile list cannot be empty'}
        
        # Validate each hotel entry
        for i, hotel in enumerate(premium_hotels):
            if not isinstance(hotel, dict):
                return {'passed': False, 'message': f'Hotel entry {i} must be a dictionary'}
            
            required_hotel_keys = ['hotel_name', 'guest_rating', 'rate', 'special_offer']
            for key in required_hotel_keys:
                if key not in hotel:
                    return {'passed': False, 'message': f'Hotel entry {i} missing key: {key}'}
            
            # Check hotel name is non-empty string
            hotel_name = hotel.get('hotel_name')
            if not isinstance(hotel_name, str) or not hotel_name.strip():
                return {'passed': False, 'message': f'Hotel entry {i} must have non-empty hotel_name'}
            
            # Check special offer is non-empty string
            special_offer = hotel.get('special_offer')
            if not isinstance(special_offer, str) or not special_offer.strip():
                return {'passed': False, 'message': f'Hotel entry {i} must have non-empty special_offer'}
            
            # Check rating and rate are numeric
            try:
                rating = float(hotel.get('guest_rating', 0))
                rate = float(hotel.get('rate', 0))
                
                if rating < 4.5:
                    return {'passed': False, 'message': f'Hotel entry {i} rating must be ≥4.5'}
                
                if rate <= 0:
                    return {'passed': False, 'message': f'Hotel entry {i} rate must be positive'}
            except (ValueError, TypeError):
                return {'passed': False, 'message': f'Hotel entry {i} has invalid numeric values'}
        
        # Use a data tool to cross-check
        hotel_data = tools['search_hotel_pricing']('')
        if isinstance(hotel_data, list) and hotel_data:
            # Verify that at least some hotels exist in the data
            hotel_count = 0
            for item in hotel_data:
                if isinstance(item, dict) and 'hotel_name' in item:
                    hotel_count += 1
            
            if hotel_count == 0:
                return {'passed': False, 'message': 'No hotel data found to verify against'}
        
        # Check data quality notes values are meaningful
        high_rated = notes.get('high_rated_hotels_analyzed', '0')
        low_rated = notes.get('low_rated_hotels_analyzed', '0')
        
        try:
            high_int = int(high_rated)
            low_int = int(low_rated)
            
            # ADR premium calculation requires both groups
            if high_int == 0 or low_int == 0:
                return {'passed': False, 'message': 'ADR premium requires both high and low rated hotels'}
        except ValueError:
            return {'passed': False, 'message': 'Invalid numeric values in data_quality_notes'}
        
        return {'passed': True, 'message': 'All verification checks passed'}
        
    except Exception as e:
        return {'passed': False, 'message': f'Verification error: {str(e)}'}
