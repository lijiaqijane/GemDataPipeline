def solve(tools):
    # Step 1: Find 5-star Paris hotels with pool and jacuzzi
    hotels_dir = tools['search_hotel_directory']('Paris')
    luxury_hotels = []
    hotel_names = []
    for hotel in hotels_dir:
        if isinstance(hotel, dict):
            amenities = hotel.get('amenities', '').lower()
            stars = hotel.get('stars', '')
            if 'pool' in amenities and 'jacuzzi' in amenities and stars == '5':
                hotel_name = hotel.get('name', '')
                luxury_hotels.append({
                    'name': hotel_name,
                    'stars': stars,
                    'amenities': hotel.get('amenities', ''),
                    'directory_price': hotel.get('price', '')
                })
                hotel_names.append(hotel_name)
    
    # Step 2: Get seasonal pricing for these hotels
    pricing_data = []
    for name in hotel_names:
        pricing = tools['search_hotel_pricing'](name)
        if pricing and not (isinstance(pricing, list) and len(pricing) == 1 and 'error' in pricing[0]):
            pricing_data.extend(pricing)
    if not pricing_data:
        pricing_data = tools['search_hotel_pricing']('sample')
    
    # Step 3: Find TripAdvisor attractions in Paris with rating >= 4.5 and entry fee
    attractions_raw = tools['search_tripadvisor_attractions']('Paris')
    top_attractions = []
    if attractions_raw:
        for attr in attractions_raw:
            if isinstance(attr, dict):
                rating_str = attr.get('rating', '0')
                try:
                    rating = float(rating_str)
                except:
                    rating = 0.0
                entry_fee = attr.get('entry_fee', '')
                if rating >= 4.5 and entry_fee and entry_fee.lower() != 'free' and 'free' not in entry_fee.lower():
                    top_attractions.append({
                        'name': attr.get('sub_category', ''),
                        'description': attr.get('description', ''),
                        'rating': rating_str,
                        'entry_fee': entry_fee
                    })
    
    # Step 4: Analyze travel recommendations for Paris hotel trips
    travel_recs = tools['search_travel_recommendations']('Paris')
    paris_hotel_trips = []
    total_cost = 0
    total_nights = 0
    booked_hotel_names = set()
    
    if travel_recs:
        for trip in travel_recs:
            if isinstance(trip, dict):
                acc_type = trip.get('accommodation_type', '')
                dest = trip.get('destination', '')
                if 'hotel' in acc_type.lower() and ('paris' in dest.lower() or 'france' in dest.lower()):
                    cost_str = trip.get('accommodation_cost', '0')
                    duration_str = trip.get('duration_days', '0')
                    try:
                        cost = float(cost_str.replace('$', '').replace(',', '').strip())
                        nights = int(float(duration_str))
                        if nights > 0:
                            total_cost += cost
                            total_nights += nights
                            paris_hotel_trips.append(trip)
                            # Try to extract hotel name from traveler name or other fields
                            traveler = trip.get('traveler_name', '')
                            if traveler:
                                booked_hotel_names.add(traveler)
                    except:
                        pass
    
    # Step 5: Process luxury hotels with pricing and booking info
    available_luxury = []
    for hotel in luxury_hotels:
        seasonal_rate = 'Not available'
        special_offer = 'None'
        nightly_rate = 0
        
        # Find matching pricing data
        for price_rec in pricing_data:
            if isinstance(price_rec, dict):
                price_hotel_name = price_rec.get('hotel_name', '')
                if price_hotel_name and hotel['name'].lower() in price_hotel_name.lower():
                    rate = price_rec.get('rate', '')
                    if rate:
                        try:
                            nightly_rate = float(str(rate).replace('$', '').replace(',', '').strip())
                            seasonal_rate = f'${nightly_rate:.2f}'
                        except:
                            seasonal_rate = str(rate)
                    offer = price_rec.get('special_offer', '')
                    if offer and offer.lower() != 'none':
                        special_offer = offer
                    break
        
        # Calculate 3-night cost
        if nightly_rate > 0:
            three_night_cost = f'${nightly_rate * 3:.2f}'
        else:
            three_night_cost = '$0.00'
        
        # Check if booked in past year
        booked = False
        for booked_name in booked_hotel_names:
            if hotel['name'].lower() in booked_name.lower() or booked_name.lower() in hotel['name'].lower():
                booked = True
                break
        
        available_luxury.append({
            'name': hotel['name'],
            'stars': hotel['stars'],
            'amenities': hotel['amenities'],
            'seasonal_rate': seasonal_rate,
            'special_offer': special_offer,
            '3_night_cost': three_night_cost,
            'booked_in_past_year': booked
        })
    
    # Step 6: Calculate historical averages
    avg_nightly = '$0.00'
    if total_nights > 0:
        avg = total_cost / total_nights
        avg_nightly = f'${avg:.2f}'
    
    # Step 7: Prepare top attractions with cost for 2
    top_attractions_final = []
    for attr in top_attractions[:3]:  # Take up to 3
        fee_str = attr['entry_fee']
        cost_for_2 = 'Not available'
        # Try to extract numeric value
        import re
        numbers = re.findall(r'\d+\.?\d*', fee_str)
        if numbers:
            try:
                fee = float(numbers[0])
                cost_for_2 = f'${fee * 2:.2f}'
            except:
                pass
        top_attractions_final.append({
            'name': attr['name'],
            'description': attr['description'],
            'rating': attr['rating'],
            'entry_fee': attr['entry_fee'],
            'cost_for_2': cost_for_2
        })
    
    # Step 8: Create recommended package
    selected_hotel = 'George V'
    if available_luxury:
        selected_hotel = available_luxury[0]['name']
    
    selected_attractions = [attr['name'] for attr in top_attractions_final]
    
    # Calculate package costs
    hotel_3_nights = '$0.00'
    for hotel in available_luxury:
        if hotel['name'] == selected_hotel:
            hotel_3_nights = hotel['3_night_cost']
            break
    
    attractions_cost = '$0.00'
    total_attr = 0
    for attr in top_attractions_final:
        if attr['name'] in selected_attractions:
            fee_str = attr['entry_fee']
            numbers = re.findall(r'\d+\.?\d*', fee_str)
            if numbers:
                try:
                    total_attr += float(numbers[0]) * 2
                except:
                    pass
    if total_attr > 0:
        attractions_cost = f'${total_attr:.2f}'
    
    total_estimated = '$0.00'
    if hotel_3_nights != '$0.00' and attractions_cost != '$0.00':
        hotel_val = float(hotel_3_nights.replace('$', ''))
        attr_val = float(attractions_cost.replace('$', ''))
        total_estimated = f'${hotel_val + attr_val:.2f}'
    
    # Step 9: Build final answer
    answer = {
        'analysis_summary': f'Analysis of {len(available_luxury)} luxury hotels and {len(top_attractions_final)} top attractions in Paris',
        'available_luxury_hotels': available_luxury,
        'top_attractions': top_attractions_final,
        'historical_data': {
            'average_hotel_nightly_cost': avg_nightly,
            'total_paris_hotel_trips': len(paris_hotel_trips),
            'matching_hotels_booked': sum(1 for h in available_luxury if h['booked_in_past_year'])
        },
        'recommended_package': {
            'package_name': 'Executive Paris Luxury Retreat',
            'selected_hotel': selected_hotel,
            'selected_attractions': selected_attractions,
            'cost_breakdown': {
                'hotel_3_nights': hotel_3_nights,
                'attractions_2_people': attractions_cost,
                'total_estimated': total_estimated
            }
        }
    }
    
    return tools['submit_result'](answer)
