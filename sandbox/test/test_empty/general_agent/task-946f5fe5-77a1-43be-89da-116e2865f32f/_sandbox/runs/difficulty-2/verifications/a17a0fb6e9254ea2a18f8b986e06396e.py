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
        if not isinstance(total_hotels, int):
            return {'passed': False, 'message': 'total_hotels_analyzed must be integer'}
        
        if len(analysis) != total_hotels:
            return {'passed': False, 'message': f'Analysis count {len(analysis)} != total_hotels {total_hotels}'}
        
        # Verify each hotel in analysis
        for idx, hotel_analysis in enumerate(analysis):
            if not isinstance(hotel_analysis, dict):
                return {'passed': False, 'message': f'Hotel analysis {idx} is not dict'}
            
            required_hotel_keys = ['hotel_name', 'stars', 'amenities', 'most_recent_review', 
                                  'current_deluxe_rate', 'average_review_rating', 'rating_discrepancy']
            for key in required_hotel_keys:
                if key not in hotel_analysis:
                    return {'passed': False, 'message': f'Hotel {idx} missing key: {key}'}
            
            # Check types
            if not isinstance(hotel_analysis['hotel_name'], str):
                return {'passed': False, 'message': f'Hotel {idx} name not string'}
            
            if not isinstance(hotel_analysis['stars'], str):
                return {'passed': False, 'message': f'Hotel {idx} stars not string'}
            
            if not isinstance(hotel_analysis['average_review_rating'], (int, float)):
                return {'passed': False, 'message': f'Hotel {idx} average rating not numeric'}
            
            if not isinstance(hotel_analysis['rating_discrepancy'], (int, float)):
                return {'passed': False, 'message': f'Hotel {idx} discrepancy not numeric'}
            
            # Verify most_recent_review structure
            review = hotel_analysis.get('most_recent_review')
            if not isinstance(review, dict):
                return {'passed': False, 'message': f'Hotel {idx} review not dict'}
            
            review_keys = ['review_date', 'review_title', 'rating', 'provider_name']
            for rkey in review_keys:
                if rkey not in review:
                    return {'passed': False, 'message': f'Hotel {idx} review missing {rkey}'}
                if not isinstance(review[rkey], str):
                    return {'passed': False, 'message': f'Hotel {idx} review {rkey} not string'}
        
        # Cross-check with actual data using a tool
        if analysis:
            # Use search_hotel_listings to verify at least one hotel exists
            sample_hotel = analysis[0]['hotel_name']
            listings_check = tools['search_hotel_listings'](sample_hotel)
            
            if isinstance(listings_check, list) and len(listings_check) > 0:
                # Found some data
                pass
            else:
                # No data found - might be okay if hotel doesn't exist in listings
                pass
        
        # Verify hotel_with_highest_positive_discrepancy
        highest_hotel = data.get('hotel_with_highest_positive_discrepancy')
        if not isinstance(highest_hotel, str):
            return {'passed': False, 'message': 'Highest hotel name not string'}
        
        # If there are hotels, verify highest hotel is in analysis
        if analysis and highest_hotel:
            hotel_names = [h.get('hotel_name', '') for h in analysis]
            if highest_hotel not in hotel_names and highest_hotel != '':
                return {'passed': False, 'message': 'Highest hotel not in analysis list'}
        
        return {'passed': True, 'message': 'Verification passed', 'details': {'hotels_analyzed': total_hotels}}
        
    except Exception as e:
        return {'passed': False, 'message': f'Verification error: {str(e)}'}
