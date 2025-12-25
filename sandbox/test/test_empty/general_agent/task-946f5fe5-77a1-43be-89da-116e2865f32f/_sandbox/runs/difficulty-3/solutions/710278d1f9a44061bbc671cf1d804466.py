def solve(tools):
    # Step 1: Find travelers aged 30-45 who visited Paris in 2023
    trip_query = 'Paris 2023'
    trips = tools['search_trip_details'](trip_query)
    
    travelers = []
    if isinstance(trips, list):
        for trip in trips:
            if isinstance(trip, dict):
                try:
                    age = int(trip.get('traveler_age', '0'))
                    destination = trip.get('destination', '')
                    year = trip.get('start_date', '').split('/')[-1] if '/' in trip.get('start_date', '') else ''
                    
                    if 30 <= age <= 45 and 'paris' in destination.lower() and '2023' in year:
                        # Calculate total trip cost
                        acc_cost_str = trip.get('accommodation_cost', '0').replace('$', '').replace(',', '').strip()
                        trans_cost_str = trip.get('transportation_cost', '0').replace('$', '').replace(',', '').strip()
                        
                        try:
                            acc_cost = float(acc_cost_str) if acc_cost_str else 0.0
                        except:
                            acc_cost = 0.0
                        
                        try:
                            trans_cost = float(trans_cost_str) if trans_cost_str else 0.0
                        except:
                            trans_cost = 0.0
                        
                        total_cost = acc_cost + trans_cost
                        
                        travelers.append({
                            'name': trip.get('traveler_name', ''),
                            'age': str(age),
                            'nationality': trip.get('traveler_nationality', ''),
                            'accommodation_type': trip.get('accommodation_type', ''),
                            'total_cost': total_cost
                        })
                except:
                    continue
    
    analysis_list = []
    highest_overall_ratio = -float('inf')
    highest_traveler = ''
    
    for traveler in travelers:
        traveler_name = traveler['name']
        acc_type = traveler['accommodation_type']
        total_cost = traveler['total_cost']
        
        # Step 2: Find matching hotels in Paris based on accommodation type
        hotels_query = f'Paris {acc_type}'
        hotels = tools['search_hotel_listings'](hotels_query)
        
        matching_hotels = []
        if isinstance(hotels, list):
            for hotel in hotels:
                if isinstance(hotel, dict):
                    city = hotel.get('city', '')
                    country = hotel.get('country', '')
                    if 'paris' in city.lower() and 'france' in country.lower():
                        hotel_name = hotel.get('name', '')
                        stars = hotel.get('stars', '')
                        
                        # Step 3: Find most expensive deluxe/executive rate
                        pricing_query = f'{hotel_name} Deluxe Executive'
                        pricing = tools['search_hotel_pricing'](pricing_query)
                        
                        max_rate = 'N/A'
                        max_rate_num = 0.0
                        if isinstance(pricing, list):
                            for price in pricing:
                                if isinstance(price, dict):
                                    room_type = price.get('room_type', '')
                                    rate_str = price.get('rate', '')
                                    hotel_name_price = price.get('hotel_name', '')
                                    
                                    if hotel_name.lower() in hotel_name_price.lower():
                                        if 'deluxe' in room_type.lower() or 'executive' in room_type.lower():
                                            try:
                                                rate_num = float(rate_str.replace('$', '').replace(',', '').strip())
                                                if rate_num > max_rate_num:
                                                    max_rate_num = rate_num
                                                    max_rate = rate_str
                                            except:
                                                pass
                        
                        # Step 4: Find highest review rating for this hotel
                        reviews_query = f'{hotel_name} Paris'
                        reviews = tools['search_travel_reviews'](reviews_query)
                        
                        highest_rating = 0.0
                        if isinstance(reviews, list):
                            for review in reviews:
                                if isinstance(review, dict):
                                    provider = review.get('provider_name', '')
                                    dest = review.get('destination_location', '')
                                    rating_str = review.get('rating', '')
                                    
                                    if hotel_name.lower() in provider.lower() or hotel_name.lower() in dest.lower():
                                        try:
                                            rating = float(rating_str)
                                            if rating > highest_rating:
                                                highest_rating = rating
                                        except:
                                            pass
                        
                        matching_hotels.append({
                            'hotel_name': hotel_name,
                            'stars': stars,
                            'most_expensive_deluxe_rate': max_rate,
                            'highest_review_rating': round(highest_rating, 2)
                        })
        
        # Step 5: Calculate cost-to-quality ratio for each hotel and find best pair
        best_pair = {'hotel_name': '', 'cost_to_quality_ratio': 0.0}
        
        for hotel in matching_hotels:
            stars_str = hotel['stars']
            highest_rating = hotel['highest_review_rating']
            
            try:
                stars_float = float(stars_str)
                if total_cost > 0 and highest_rating > 0:
                    ratio = (stars_float * highest_rating) / total_cost
                    if ratio > best_pair['cost_to_quality_ratio']:
                        best_pair = {
                            'hotel_name': hotel['hotel_name'],
                            'cost_to_quality_ratio': round(ratio, 4)
                        }
            except:
                continue
        
        # Track highest overall ratio
        if best_pair['cost_to_quality_ratio'] > highest_overall_ratio:
            highest_overall_ratio = best_pair['cost_to_quality_ratio']
            highest_traveler = traveler_name
        
        analysis_entry = {
            'traveler_name': traveler_name,
            'traveler_age': traveler['age'],
            'traveler_nationality': traveler['nationality'],
            'accommodation_type': acc_type,
            'total_trip_cost': round(total_cost, 2),
            'matching_hotels': matching_hotels,
            'best_hotel_pair': best_pair
        }
        
        analysis_list.append(analysis_entry)
    
    # Prepare final answer
    answer = {
        'analysis': analysis_list,
        'traveler_with_highest_overall_ratio': highest_traveler if highest_overall_ratio > 0 else '',
        'total_travelers_analyzed': str(len(analysis_list))
    }
    
    return tools['submit_result'](answer)
