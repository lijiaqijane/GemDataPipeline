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
        required_keys = ['analysis', 'traveler_with_highest_overall_ratio', 'total_travelers_analyzed']
        for key in required_keys:
            if key not in data:
                return {'passed': False, 'message': f'Missing required key: {key}'}
        
        analysis = data.get('analysis')
        if not isinstance(analysis, list):
            return {'passed': False, 'message': 'Analysis must be a list'}
        
        total_travelers = data.get('total_travelers_analyzed')
        if not isinstance(total_travelers, str):
            return {'passed': False, 'message': 'total_travelers_analyzed must be string'}
        
        # Verify total_travelers matches analysis length
        if int(total_travelers) != len(analysis):
            return {'passed': False, 'message': f'total_travelers_analyzed ({total_travelers}) does not match analysis length ({len(analysis)})'}
        
        # Use tools to cross-check data
        trips = tools['search_trip_details']('Paris 2023')
        if isinstance(trips, list):
            paris_travelers_count = 0
            for trip in trips:
                if isinstance(trip, dict):
                    try:
                        age = int(trip.get('traveler_age', '0'))
                        destination = trip.get('destination', '')
                        year = trip.get('start_date', '').split('/')[-1] if '/' in trip.get('start_date', '') else ''
                        
                        if 30 <= age <= 45 and 'paris' in destination.lower() and '2023' in year:
                            paris_travelers_count += 1
                    except:
                        continue
            
            # Check if analysis is empty when travelers exist
            if paris_travelers_count > 0 and len(analysis) == 0:
                return {'passed': False, 'message': f'Found {paris_travelers_count} travelers but analysis is empty'}
            
            # Check if analysis has entries when no travelers found
            if paris_travelers_count == 0 and len(analysis) > 0:
                return {'passed': False, 'message': 'Analysis has entries but no travelers found matching criteria'}
        
        # Validate each analysis entry structure
        for i, entry in enumerate(analysis):
            if not isinstance(entry, dict):
                return {'passed': False, 'message': f'Analysis entry {i} is not a dict'}
            
            entry_keys = ['traveler_name', 'traveler_age', 'traveler_nationality', 'accommodation_type', 
                         'total_trip_cost', 'matching_hotels', 'best_hotel_pair']
            for key in entry_keys:
                if key not in entry:
                    return {'passed': False, 'message': f'Analysis entry {i} missing key: {key}'}
            
            # Check matching_hotels structure
            hotels = entry.get('matching_hotels')
            if not isinstance(hotels, list):
                return {'passed': False, 'message': f'Analysis entry {i} matching_hotels must be a list'}
            
            for j, hotel in enumerate(hotels):
                if not isinstance(hotel, dict):
                    return {'passed': False, 'message': f'Analysis entry {i} hotel {j} is not a dict'}
                
                hotel_keys = ['hotel_name', 'stars', 'most_expensive_deluxe_rate', 'highest_review_rating']
                for key in hotel_keys:
                    if key not in hotel:
                        return {'passed': False, 'message': f'Analysis entry {i} hotel {j} missing key: {key}'}
                
                # Check data types
                if not isinstance(hotel.get('highest_review_rating'), (int, float)):
                    return {'passed': False, 'message': f'Analysis entry {i} hotel {j} highest_review_rating must be numeric'}
            
            # Check best_hotel_pair structure
            best_pair = entry.get('best_hotel_pair')
            if not isinstance(best_pair, dict):
                return {'passed': False, 'message': f'Analysis entry {i} best_hotel_pair must be a dict'}
            
            pair_keys = ['hotel_name', 'cost_to_quality_ratio']
            for key in pair_keys:
                if key not in best_pair:
                    return {'passed': False, 'message': f'Analysis entry {i} best_hotel_pair missing key: {key}'}
            
            if not isinstance(best_pair.get('cost_to_quality_ratio'), (int, float)):
                return {'passed': False, 'message': f'Analysis entry {i} best_hotel_pair cost_to_quality_ratio must be numeric'}
            
            # Check total_trip_cost type
            if not isinstance(entry.get('total_trip_cost'), (int, float)):
                return {'passed': False, 'message': f'Analysis entry {i} total_trip_cost must be numeric'}
        
        # Check traveler_with_highest_overall_ratio
        highest_traveler = data.get('traveler_with_highest_overall_ratio', '')
        if not isinstance(highest_traveler, str):
            return {'passed': False, 'message': 'traveler_with_highest_overall_ratio must be string'}
        
        # If highest traveler is specified, verify it exists in analysis
        if highest_traveler:
            found = False
            for entry in analysis:
                if entry.get('traveler_name') == highest_traveler:
                    found = True
                    # Verify best_hotel_pair has positive ratio
                    if entry.get('best_hotel_pair', {}).get('cost_to_quality_ratio', 0) <= 0:
                        return {'passed': False, 'message': f'Highest traveler {highest_traveler} does not have positive cost-to-quality ratio'}
                    break
            if not found:
                return {'passed': False, 'message': f'Highest traveler {highest_traveler} not found in analysis'}
        
        return {'passed': True, 'message': 'Verification passed'}
    
    except Exception as e:
        return {'passed': False, 'message': f'Verification exception: {str(e)}'}
