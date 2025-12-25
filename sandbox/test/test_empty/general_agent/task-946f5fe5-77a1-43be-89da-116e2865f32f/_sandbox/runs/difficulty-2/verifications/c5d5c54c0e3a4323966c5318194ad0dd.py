def verify(tools, answer):
    import json
    try:
        # Handle wrapped answer
        if isinstance(answer, dict):
            if 'submitted_data' in answer:
                data = answer.get('submitted_data')
            elif 'data' in answer:
                data = answer.get('data')
            else:
                data = answer
        else:
            return {'passed': False, 'message': 'Answer is not a dict'}
        
        if data is None:
            return {'passed': False, 'message': 'Answer data is None'}
        
        # Check required structure
        required_keys = ['analysis', 'hotel_with_highest_positive_discrepancy', 'total_hotels_analyzed']
        for key in required_keys:
            if key not in data:
                return {'passed': False, 'message': f'Missing required key: {key}'}
        
        analysis = data.get('analysis')
        if not isinstance(analysis, list):
            return {'passed': False, 'message': 'Analysis must be a list'}
        
        total_hotels = data.get('total_hotels_analyzed')
        if not isinstance(total_hotels, str):
            return {'passed': False, 'message': 'total_hotels_analyzed must be string'}
        
        # Verify total_hotels matches analysis length
        try:
            total_int = int(total_hotels)
            if total_int != len(analysis):
                return {'passed': False, 'message': f'total_hotels_analyzed {total_hotels} does not match analysis length {len(analysis)}'}
        except:
            return {'passed': False, 'message': 'total_hotels_analyzed must be convertible to integer'}
        
        # Verify each analysis item has correct structure
        for idx, item in enumerate(analysis):
            if not isinstance(item, dict):
                return {'passed': False, 'message': f'Analysis item {idx} is not a dict'}
            
            required_item_keys = ['hotel_name', 'stars', 'amenities', 'most_recent_review', 
                                 'current_deluxe_rate', 'average_review_rating', 'rating_discrepancy']
            for key in required_item_keys:
                if key not in item:
                    return {'passed': False, 'message': f'Analysis item {idx} missing key: {key}'}
            
            # Check types
            if not isinstance(item['hotel_name'], str):
                return {'passed': False, 'message': f'Hotel name at index {idx} must be string'}
            if not isinstance(item['stars'], str):
                return {'passed': False, 'message': f'Stars at index {idx} must be string'}
            if not isinstance(item['amenities'], str):
                return {'passed': False, 'message': f'Amenities at index {idx} must be string'}
            if not isinstance(item['current_deluxe_rate'], str):
                return {'passed': False, 'message': f'Current deluxe rate at index {idx} must be string'}
            if not isinstance(item['average_review_rating'], float):
                return {'passed': False, 'message': f'Average review rating at index {idx} must be float'}
            if not isinstance(item['rating_discrepancy'], float):
                return {'passed': False, 'message': f'Rating discrepancy at index {idx} must be float'}
            
            # Check most_recent_review structure
            review = item['most_recent_review']
            if not isinstance(review, dict):
                return {'passed': False, 'message': f'Most recent review at index {idx} must be dict'}
            
            review_keys = ['review_date', 'review_title', 'rating', 'provider_name']
            for rkey in review_keys:
                if rkey not in review:
                    return {'passed': False, 'message': f'Review at index {idx} missing key: {rkey}'}
                if not isinstance(review[rkey], str):
                    return {'passed': False, 'message': f'Review {rkey} at index {idx} must be string'}
        
        # Verify hotel_with_highest_positive_discrepancy is string
        if not isinstance(data['hotel_with_highest_positive_discrepancy'], str):
            return {'passed': False, 'message': 'hotel_with_highest_positive_discrepancy must be string'}
        
        # Cross-check with hotel listings tool
        listings = tools['search_hotel_listings']('Paris')
        if isinstance(listings, list):
            # Count hotels with pool and wifi
            pool_wifi_count = 0
            for hotel in listings:
                if isinstance(hotel, dict):
                    amenities = hotel.get('amenities', '')
                    if amenities and 'pool' in amenities.lower() and 'wifi' in amenities.lower():
                        pool_wifi_count += 1
            
            # If we found hotels with pool+wifi, verify analysis is not empty
            if pool_wifi_count > 0 and len(analysis) == 0:
                return {'passed': False, 'message': f'Found {pool_wifi_count} hotels with pool+wifi but analysis is empty'}
        
        return {'passed': True, 'message': 'Verification passed'}
    except Exception as e:
        return {'passed': False, 'message': f'Verification error: {str(e)}'}
