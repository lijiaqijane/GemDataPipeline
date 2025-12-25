def verify(tools, answer):
    import json
    
    try:
        # Unwrap answer if needed
        if isinstance(answer, dict):
            if 'status' in answer and 'data' in answer:
                actual_data = answer.get('data')
            elif 'status' in answer and 'submitted_data' in answer:
                actual_data = answer.get('submitted_data')
            else:
                actual_data = answer
        else:
            actual_data = answer
        
        if actual_data is None:
            return {'passed': False, 'message': 'Answer is None'}
        
        if not isinstance(actual_data, dict):
            return {'passed': False, 'message': 'Answer is not a dictionary'}
        
        # Check required keys
        required_keys = ['available_hotels_with_offers', 'total_count']
        for key in required_keys:
            if key not in actual_data:
                return {'passed': False, 'message': f'Missing required key: {key}'}
        
        hotels_list = actual_data.get('available_hotels_with_offers', [])
        total_count = actual_data.get('total_count')
        
        # total_count must be integer
        if not isinstance(total_count, int):
            return {'passed': False, 'message': 'total_count must be integer, not string'}
        
        # Consistency check: total_count should match list length
        if total_count != len(hotels_list):
            return {'passed': False, 'message': 'total_count does not match list length'}
        
        # Non‑emptiness check: if there are hotels, they must have content
        if total_count > 0:
            if not hotels_list:
                return {'passed': False, 'message': 'total_count > 0 but list is empty'}
            
            required_fields = ['hotel_name', 'room_type', 'rate', 'special_offer']
            for hotel in hotels_list:
                if not isinstance(hotel, dict):
                    return {'passed': False, 'message': 'Hotel entry is not a dict'}
                
                for field in required_fields:
                    if field not in hotel:
                        return {'passed': False, 'message': f'Missing field {field} in hotel'}
                    
                    value = hotel.get(field)
                    if not isinstance(value, str) or not value.strip():
                        return {'passed': False, 'message': f'Field {field} is empty or not a string'}
                
                # Special offer must not be 'None'
                if hotel.get('special_offer', '').lower() == 'none':
                    return {'passed': False, 'message': 'special_offer contains "None"'}
        
        # Cross‑check with a data tool
        check_data = tools['search_hotel_pricing']('Available')
        if isinstance(check_data, list) and len(check_data) > 0:
            # Verify that at least some hotels in the dataset have availability 'Available'
            available_in_dataset = any(
                isinstance(h, dict) and h.get('availability') == 'Available'
                for h in check_data if isinstance(h, dict)
            )
            
            # If dataset has Available hotels but answer has none, that's suspicious
            if available_in_dataset and total_count == 0:
                return {'passed': False, 'message': 'Dataset has Available hotels but answer shows none'}
        
        return {'passed': True, 'message': 'Verification passed'}
    
    except Exception as e:
        return {'passed': False, 'message': f'Exception during verification: {str(e)}'}
