def solve(tools):
    # Get hotel pricing data
    hotel_data = tools['search_hotel_pricing']('')
    
    # Get travel recommendations data for Paris trips
    travel_data = tools['search_travel_recommendations']('Paris')
    
    # Initialize data structures
    high_rated_rates = []
    low_rated_rates = []
    all_rates = []
    all_ratings = []
    premium_hotels = []
    
    # Process hotel data
    if isinstance(hotel_data, list):
        for item in hotel_data:
            if isinstance(item, dict):
                try:
                    hotel_name = str(item.get('hotel_name', '')).strip()
                    rate_str = str(item.get('rate', '0')).replace('$', '').replace(',', '').strip()
                    rating_str = str(item.get('guest_rating', '0')).strip()
                    special_offer = str(item.get('special_offer', '')).strip()
                    
                    if rate_str and rating_str:
                        rate = float(rate_str)
                        rating = float(rating_str)
                        
                        # Collect for correlation
                        all_rates.append(rate)
                        all_ratings.append(rating)
                        
                        # Categorize by rating for ADR premium
                        if rating >= 4.5:
                            high_rated_rates.append(rate)
                        else:
                            low_rated_rates.append(rate)
                        
                        # Check for premium hotel criteria
                        if rating >= 4.5 and special_offer and special_offer.lower() != 'none' and special_offer.strip() != '':
                            premium_hotels.append({
                                'hotel_name': hotel_name,
                                'guest_rating': rating,
                                'rate': rate,
                                'special_offer': special_offer
                            })
                except (ValueError, TypeError):
                    continue
    
    # Calculate ADR premium percentage
    adr_premium = 0.0
    if high_rated_rates and low_rated_rates:
        avg_high = sum(high_rated_rates) / len(high_rated_rates)
        avg_low = sum(low_rated_rates) / len(low_rated_rates)
        if avg_low > 0:
            adr_premium = ((avg_high - avg_low) / avg_low) * 100
    
    # Round and convert to string
    adr_premium_str = str(round(adr_premium, 1))
    
    # Calculate 75th percentile rate for premium hotels
    if all_rates:
        sorted_rates = sorted(all_rates)
        idx = int(0.75 * len(sorted_rates))
        percentile_75 = sorted_rates[idx] if idx < len(sorted_rates) else sorted_rates[-1]
    else:
        percentile_75 = float('inf')
    
    # Filter premium hotels below 75th percentile
    filtered_premium_hotels = []
    for hotel in premium_hotels:
        if hotel['rate'] < percentile_75:
            filtered_premium_hotels.append(hotel)
    
    # Sort premium hotels by rating descending, then rate ascending
    filtered_premium_hotels.sort(key=lambda x: (-x['guest_rating'], x['rate']))
    
    # Calculate correlation coefficient
    correlation = 0.0
    if len(all_rates) >= 2 and len(all_ratings) >= 2:
        n = len(all_rates)
        sum_x = sum(all_rates)
        sum_y = sum(all_ratings)
        sum_xy = sum(all_rates[i] * all_ratings[i] for i in range(n))
        sum_x2 = sum(r * r for r in all_rates)
        sum_y2 = sum(r * r for r in all_ratings)
        
        numerator = n * sum_xy - sum_x * sum_y
        denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)) ** 0.5
        
        if denominator != 0:
            correlation = numerator / denominator
    
    # Round correlation to 3 decimals and convert to string
    correlation_str = str(round(correlation, 3))
    
    # Process travel data for most expensive accommodation type
    accommodation_costs = {}
    american_trips = 0
    
    if isinstance(travel_data, list):
        for item in travel_data:
            if isinstance(item, dict):
                try:
                    destination = str(item.get('destination', '')).strip()
                    nationality = str(item.get('traveler_nationality', '')).strip()
                    acc_type = str(item.get('accommodation_type', '')).strip()
                    cost_str = str(item.get('accommodation_cost', '0')).replace('$', '').replace(',', '').strip()
                    duration_str = str(item.get('duration_days', '0')).strip()
                    
                    # Check if it's an American traveler to Paris
                    if 'paris' in destination.lower() and 'american' in nationality.lower():
                        cost = float(cost_str) if cost_str else 0
                        duration = float(duration_str) if duration_str else 0
                        
                        if cost > 0 and duration > 0:
                            american_trips += 1
                            cost_per_night = cost / duration
                            
                            if acc_type:
                                if acc_type not in accommodation_costs:
                                    accommodation_costs[acc_type] = []
                                accommodation_costs[acc_type].append(cost_per_night)
                except (ValueError, TypeError):
                    continue
    
    # Find most expensive accommodation type
    most_expensive_type = ''
    max_avg_cost = 0
    
    for acc_type, costs in accommodation_costs.items():
        if costs:
            avg_cost = sum(costs) / len(costs)
            if avg_cost > max_avg_cost:
                max_avg_cost = avg_cost
                most_expensive_type = acc_type
    
    # If no accommodation type found, set default
    if not most_expensive_type:
        most_expensive_type = 'Hotel'
    
    # Prepare data quality notes
    data_quality_notes = {
        'high_rated_hotels_analyzed': str(len(high_rated_rates)),
        'low_rated_hotels_analyzed': str(len(low_rated_rates)),
        'american_traveler_trips_analyzed': str(american_trips),
        'total_hotels_in_correlation': str(len(all_rates))
    }
    
    # Prepare final answer with proper string conversions
    answer = {
        'adr_premium_percentage': adr_premium_str,
        'most_expensive_accommodation_type': most_expensive_type,
        'premium_hotels_below_75th_percentile': filtered_premium_hotels,
        'rating_rate_correlation': correlation_str,
        'data_quality_notes': data_quality_notes
    }
    
    return tools['submit_result'](answer)
