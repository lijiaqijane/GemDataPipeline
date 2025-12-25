def solve(tools):
    # Step 1: Find 5-star Paris hotels with pool and jacuzzi
    hotels = tools['search_hotel_directory']('Paris')
    luxury_hotels = []
    for hotel in hotels:
        if isinstance(hotel, dict):
            amenities = hotel.get('amenities', '').lower()
            stars = hotel.get('stars', '')
            if 'pool' in amenities and 'jacuzzi' in amenities and stars == '5':
                luxury_hotels.append({
                    'name': hotel.get('name', ''),
                    'stars': stars,
                    'amenities': hotel.get('amenities', ''),
                    'directory_price': hotel.get('price', '')
                })
    
    # Step 2: Get seasonal pricing for these hotels
    pricing_data = tools['search_hotel_pricing']('Paris')
    for hotel in luxury_hotels:
        seasonal_rate = 'Not available'
        special_offer = 'None'
        for price_rec in pricing_data:
            if isinstance(price_rec, dict):
                if price_rec.get('hotel_name', '') == hotel['name']:
                    rate = price_rec.get('rate', '')
                    if rate:
                        seasonal_rate = str(rate)
                    offer = price_rec.get('special_offer', '')
                    if offer and offer.lower() != 'none':
                        special_offer = offer
                    break
        hotel['seasonal_rate'] = seasonal_rate
        hotel['special_offer'] = special_offer
        
        # Calculate 3-night cost
        try:
            rate_val = float(seasonal_rate.replace('$', '').replace(',', '')) if seasonal_rate != 'Not available' else 0
            hotel['3_night_cost'] = f'${rate_val * 3:.2f}'
        except:
            hotel['3_night_cost'] = 'Not available'
    
    # Step 3: Find top attractions in Paris
    attractions = tools['search_tripadvisor_attractions']('Paris')
    top_attractions = []
    for attr in attractions:
        if isinstance(attr, dict):
            rating = attr.get('rating', '')
            entry_fee = attr.get('entry_fee', '')
            try:
                rating_val = float(rating)
                if rating_val >= 4.5 and entry_fee and 'free' not in entry_fee.lower():
                    top_attractions.append({
                        'name': attr.get('sub_category', ''),
                        'description': attr.get('description', ''),
                        'rating': rating,
                        'entry_fee': entry_fee
                    })
            except:
                continue
    
    # Calculate attraction costs for 2 people
    for attr in top_attractions:
        fee_str = attr['entry_fee']
        try:
            # Extract first number from fee string
            import re
            numbers = re.findall(r'\d+\.?\d*', fee_str)
            if numbers:
                fee_val = float(numbers[0])
                attr['cost_for_2'] = f'${fee_val * 2:.2f}'
            else:
                attr['cost_for_2'] = 'Not available'
        except:
            attr['cost_for_2'] = 'Not available'
    
    # Step 4: Analyze travel recommendations for Paris hotel trips
    trips = tools['search_travel_recommendations']('Paris')
    paris_hotel_trips = []
    total_accommodation_cost = 0
    total_nights = 0
    
    for trip in trips:
        if isinstance(trip, dict):
            accommodation = trip.get('accommodation_type', '')
            if accommodation.lower() == 'hotel':
                cost_str = trip.get('accommodation_cost', '')
                duration_str = trip.get('duration_days', '')
                try:
                    # Clean cost string
                    cost_clean = ''.join(c for c in cost_str if c.isdigit() or c == '.')
                    if cost_clean:
                        cost = float(cost_clean)
                        duration = int(duration_str) if duration_str.isdigit() else 1
                        total_accommodation_cost += cost
                        total_nights += duration
                        paris_hotel_trips.append(trip)
                except:
                    continue
    
    # Calculate average nightly cost
    avg_nightly_cost = total_accommodation_cost / total_nights if total_nights > 0 else 0
    
    # Step 5: Cross-reference hotels with travel data
    booked_hotel_names = set()
    for trip in paris_hotel_trips:
        # Extract hotel name from trip data (simplified - in real scenario would need hotel name field)
        # For this exercise, we'll check if any hotel names appear in destination or other fields
        destination = trip.get('destination', '').lower()
        for hotel in luxury_hotels:
            if hotel['name'].lower() in destination:
                booked_hotel_names.add(hotel['name'])
    
    # Mark which hotels have been booked
    for hotel in luxury_hotels:
        hotel['booked_in_past_year'] = hotel['name'] in booked_hotel_names
    
    # Step 6: Create recommended package
    # Select first available hotel with seasonal rate
    selected_hotel = None
    for hotel in luxury_hotels:
        if hotel['seasonal_rate'] != 'Not available':
            selected_hotel = hotel
            break
    
    if not selected_hotel and luxury_hotels:
        selected_hotel = luxury_hotels[0]
    
    # Select top 3 attractions
    selected_attractions = top_attractions[:3] if len(top_attractions) >= 3 else top_attractions
    
    # Calculate total costs
    hotel_cost = 0
    try:
        if selected_hotel and selected_hotel['3_night_cost'] != 'Not available':
            hotel_cost = float(selected_hotel['3_night_cost'].replace('$', '').replace(',', ''))
    except:
        hotel_cost = 0
    
    attractions_cost = 0
    for attr in selected_attractions:
        try:
            if attr['cost_for_2'] != 'Not available':
                attractions_cost += float(attr['cost_for_2'].replace('$', '').replace(',', ''))
        except:
            continue
    
    total_cost = hotel_cost + attractions_cost
    
    # Prepare final answer
    answer = {
        'analysis_summary': f'Analysis of {len(luxury_hotels)} luxury hotels and {len(top_attractions)} top attractions in Paris',
        'available_luxury_hotels': [{
            'name': h['name'],
            'stars': h['stars'],
            'amenities': h['amenities'],
            'seasonal_rate': h['seasonal_rate'],
            'special_offer': h['special_offer'],
            '3_night_cost': h['3_night_cost'],
            'booked_in_past_year': h['booked_in_past_year']
        } for h in luxury_hotels],
        'top_attractions': [{
            'name': a['name'],
            'description': a['description'],
            'rating': a['rating'],
            'entry_fee': a['entry_fee'],
            'cost_for_2': a['cost_for_2']
        } for a in top_attractions],
        'historical_data': {
            'average_hotel_nightly_cost': f'${avg_nightly_cost:.2f}',
            'total_paris_hotel_trips': len(paris_hotel_trips),
            'matching_hotels_booked': len(booked_hotel_names)
        },
        'recommended_package': {
            'package_name': 'Executive Paris Luxury Retreat',
            'selected_hotel': selected_hotel['name'] if selected_hotel else 'None available',
            'selected_attractions': [a['name'] for a in selected_attractions],
            'cost_breakdown': {
                'hotel_3_nights': selected_hotel['3_night_cost'] if selected_hotel else '$0',
                'attractions_2_people': f'${attractions_cost:.2f}',
                'total_estimated': f'${total_cost:.2f}'
            }
        }
    }
    
    return tools['submit_result'](answer)
