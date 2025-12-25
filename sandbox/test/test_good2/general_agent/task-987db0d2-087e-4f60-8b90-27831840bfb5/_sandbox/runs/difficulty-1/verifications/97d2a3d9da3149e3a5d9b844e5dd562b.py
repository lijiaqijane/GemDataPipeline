def verify(tools, answer):
    import json
    
    try:
        # Handle wrapped answer from submit_result
        if isinstance(answer, dict):
            # Check if answer is wrapped with status/message/data
            if 'status' in answer and 'data' in answer:
                actual_data = answer.get('data')
            # Or if it's wrapped with status/message/submitted_data
            elif 'status' in answer and 'submitted_data' in answer:
                actual_data = answer.get('submitted_data')
            else:
                actual_data = answer
        else:
            actual_data = answer
        
        # Check if answer is None
        if actual_data is None:
            return {'passed': False, 'message': 'Answer is None'}
        
        # Check required keys
        if not isinstance(actual_data, dict):
            return {'passed': False, 'message': 'Answer is not a dictionary'}
        
        required_keys = ['available_hotels_with_offers', 'total_count']
        for key in required_keys:
            if key not in actual_data:
                return {'passed': False, 'message': f'Missing required key: {key}'}
        
        hotels_list = actual_data.get('available_hotels_with_offers')
        total_count = actual_data.get('total_count')
        
        # Check types
        if not isinstance(hotels_list, list):
            return {'passed': False, 'message': 'available_hotels_with_offers is not a list'}
        
        if not isinstance(total_count, int):
            return {'passed': False, 'message': 'total_count is not an integer'}
        
        # Check non-empty requirement
        if len(hotels_list) == 0:
            return {'passed': False, 'message': 'No hotels found - list is empty'}
        
        if total_count == 0:
            return {'passed': False, 'message': 'total_count is zero but should have hotels'}
        
        if total_count != len(hotels_list):
            return {'passed': False, 'message': f'total_count ({total_count}) does not match list length ({len(hotels_list)})'}
        
        # Check each hotel entry
        required_hotel_keys = ['hotel_name', 'room_type', 'rate', 'special_offer']
        for i, hotel in enumerate(hotels_list):
            if not isinstance(hotel, dict):
                return {'passed': False, 'message': f'Hotel entry {i} is not a dictionary'}
            
            for key in required_hotel_keys:
                if key not in hotel:
                    return {'passed': False, 'message': f'Hotel {i} missing key: {key}'}
                
                value = hotel.get(key)
                if not isinstance(value, str):
                    return {'passed': False, 'message': f'Hotel {i} {key} is not a string'}
                
                if value.strip() == '':
                    return {'passed': False, 'message': f'Hotel {i} {key} is empty string'}
            
            # Special check: special_offer should not be 'None'
            special_offer = hotel.get('special_offer', '')
            if special_offer.lower() == 'none':
                return {'passed': False, 'message': f'Hotel {i} special_offer is \'None\''}
        
        # Use a tool to cross-check data
        # Search for Paris hotels to verify consistency
        tool_result = tools['search_hotel_pricing']('Paris')
        
        # Even if tool returns error, we can still verify our answer structure
        # Count how many hotels in tool result match our answer
        if isinstance(tool_result, list):
            tool_hotel_names = set()
            for item in tool_result:
                if isinstance(item, dict) and 'hotel_name' in item:
                    tool_hotel_names.add(item['hotel_name'].strip().lower())
            
            # Check if any hotels in answer exist in tool results
            answer_hotel_names = [h['hotel_name'].strip().lower() for h in hotels_list]
            matching_count = sum(1 for name in answer_hotel_names if name in tool_hotel_names)
            
            if matching_count == 0 and len(tool_hotel_names) > 0:
                return {'passed': False, 'message': 'No hotels in answer match tool search results', 'details': {'answer_hotels': answer_hotel_names[:3], 'tool_hotels': list(tool_hotel_names)[:3]}}
        
        # All checks passed
        return {'passed': True, 'message': 'Verification successful', 'details': {'hotels_count': total_count}}
        
    except Exception as e:
        # Exception-safe: return False with error message
        return {'passed': False, 'message': f'Verification error: {str(e)}'}
