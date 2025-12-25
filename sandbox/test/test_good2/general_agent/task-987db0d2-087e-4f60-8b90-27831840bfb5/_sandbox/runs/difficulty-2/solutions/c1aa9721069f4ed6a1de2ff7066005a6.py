def solve(tools):
    # Step 1: Search for American female travelers to Paris
    travel_data = tools['search_travel_recommendations']('Paris')
    
    # Filter for American female travelers aged 25-35
    american_female_travelers = []
    total_accommodation_cost = 0
    total_nights = 0
    accommodation_types = {}
    
    if isinstance(travel_data, list):
        for trip in travel_data:
            if isinstance(trip, dict) and 'error' not in trip:
                destination = trip.get('destination', '').lower()
                nationality = trip.get('traveler_nationality', '').lower()
                gender = trip.get('traveler_gender', '').lower()
                age_str = trip.get('traveler_age', '')
                
                # Check if trip is to Paris and traveler is American female
                if ('paris' in destination and 
                    'american' in nationality and 
                    'female' in gender):
                    
                    try:
                        age = int(age_str)
                        if 25 <= age <= 35:
                            american_female_travelers.append(trip)
                            
                            # Process accommodation cost
                            acc_cost_str = trip.get('accommodation_cost', '0')
                            # Clean cost string
                            acc_cost_clean = acc_cost_str.replace('$', '').replace(',', '').replace(' ', '')
                            try:
                                acc_cost = float(acc_cost_clean)
                                total_accommodation_cost += acc_cost
                            except (ValueError, TypeError):
                                pass
                            
                            # Process duration
                            duration_str = trip.get('duration_days', '0')
                            try:
                                duration = int(duration_str)
                                total_nights += duration
                            except (ValueError, TypeError):
                                pass
                            
                            # Track accommodation types
                            acc_type = trip.get('accommodation_type', '')
                            if acc_type:
                                accommodation_types[acc_type] = accommodation_types.get(acc_type, 0) + 1
                    except (ValueError, TypeError):
                        continue
    
    # Calculate average cost per night
    avg_cost_per_night = 0.0
    if total_nights > 0 and total_accommodation_cost > 0:
        avg_cost_per_night = round(total_accommodation_cost / total_nights, 2)
    
    # Find most common accommodation type
    most_common_type = 'Not available'
    if accommodation_types:
        most_common_type = max(accommodation_types.items(), key=lambda x: x[1])[0]
    
    # Step 2: Search for Paris hotels with high ratings and special offers
    hotel_data = tools['search_hotel_pricing']('Paris')
    
    top_hotels = []
    if isinstance(hotel_data, list):
        for hotel in hotel_data:
            if isinstance(hotel, dict) and 'error' not in hotel:
                availability = hotel.get('availability', '')
                special_offer = hotel.get('special_offer', '')
                guest_rating = hotel.get('guest_rating', 0)
                
                # Check criteria: Available, has special offer, high rating
                if (availability == 'Available' and 
                    special_offer and 
                    special_offer.lower() != 'none' and 
                    isinstance(guest_rating, (int, float)) and 
                    guest_rating >= 4.5):
                    
                    # Convert rate to string
                    rate = hotel.get('rate', '')
                    if not isinstance(rate, str):
                        rate = str(rate)
                    
                    top_hotels.append({
                        'hotel_name': hotel.get('hotel_name', ''),
                        'room_type': hotel.get('room_type', ''),
                        'rate': rate,
                        'guest_rating': float(guest_rating),
                        'special_offer': special_offer
                    })
    
    # Sort hotels by rating (highest first)
    top_hotels.sort(key=lambda x: x['guest_rating'], reverse=True)
    
    # Prepare final answer
    answer = {
        'average_accommodation_cost_per_night': str(avg_cost_per_night),
        'top_rated_hotels_with_offers': top_hotels,
        'most_common_accommodation_type': most_common_type,
        'data_summary': {
            'total_american_female_travelers_analyzed': str(len(american_female_travelers)),
            'hotels_with_high_ratings_found': str(len(top_hotels))
        }
    }
    
    return tools['submit_result'](str(answer))
