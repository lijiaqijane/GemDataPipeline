def solve(tools):
    # Get hotel pricing data
    hotel_data = tools['search_hotel_pricing']('')
    
    # Get travel recommendations data
    travel_data = tools['search_travel_recommendations']('Paris')
    
    # Process hotel data for ADR premium and correlation
    high_rated_rates = []
    low_rated_rates = []
    all_rates = []
    all_ratings = []
    premium_hotels = []
    
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
                        
                        # Categorize for ADR premium
                        if rating >= 4.5:
                            high_rated_rates.append(rate)
                        else:
                            low_rated_rates.append(rate)
                        
                        # Check for premium hotels criteria
                        if rating >= 4.5 and special_offer and special_offer.lower() != 'none' and special_offer != '':
                            premium_hotels.append({
                                'hotel_name': hotel_name,
                                'guest_rating': rating,
                                'rate': rate,
                                'special_offer': special_offer
                            })
                except (ValueError, TypeError):
                    continue
    
    # Calculate ADR premium
    adr_premium = 0.0
    if high_rated_rates and low_rated_rates:
        avg_high = sum(high_rated_rates) / len(high_rated_rates)
        avg_low = sum(low_rated_rates) / len(low_rated_rates)
        if avg_low > 0:
            adr_premium = ((avg_high - avg_low) / avg_low) * 100
    
    # Calculate 75th percentile rate
    if all_rates:
        sorted_rates = sorted(all_rates)
        idx = int(0.75 * len(sorted_rates))
        percentile_75 = sorted_rates[idx] if idx < len(sorted_rates) else sorted_rates[-1]
        
        # Filter premium hotels below 75th percentile
        filtered_premium = [h for h in premium_hotels if h['rate'] < percentile_75]
        
        # Sort by rating descending, then rate ascending
        filtered_premium.sort(key=lambda x: (-x['guest_rating'], x['rate']))
    else:
        filtered_premium = []
        percentile_75 = 0
    
    # Process travel data for most expensive accommodation type
    accommodation_costs = {}
    accommodation_counts = {}
    american_trips = 0
    
    if isinstance(travel_data, list):
        for item in travel_data:
            if isinstance(item, dict):
                try:
                    dest = str(item.get('destination', '')).lower()
                    nationality = str(item.get('traveler_nationality', '')).lower()
                    gender = str(item.get('traveler_gender', '')).lower()
                    acc_type = str(item.get('accommodation_type', '')).strip()
                    cost_str = str(item.get('accommodation_cost', '0')).replace('$', '').replace(',', '').strip()
                    days_str = str(item.get('duration_days', '0')).strip()
                    
                    if ('paris' in dest and 'american' in nationality and 
                        cost_str and days_str and cost_str != '0' and days_str != '0'):
                        cost = float(cost_str)
                        days = float(days_str)
                        
                        if cost > 0 and days > 0:
                            american_trips += 1
                            cost_per_night = cost / days
                            
                            if acc_type:
                                if acc_type not in accommodation_costs:
                                    accommodation_costs[acc_type] = 0.0
                                    accommodation_counts[acc_type] = 0
                                
                                accommodation_costs[acc_type] += cost_per_night
                                accommodation_counts[acc_type] += 1
                except (ValueError, TypeError):
                    continue
    
    # Find most expensive accommodation type
    most_expensive_type = ''
    if accommodation_costs:
        avg_costs = {acc_type: accommodation_costs[acc_type] / accommodation_counts[acc_type] 
                     for acc_type in accommodation_costs}
        if avg_costs:
            most_expensive_type = max(avg_costs.items(), key=lambda x: x[1])[0]
    
    # Calculate correlation coefficient
    correlation = 0.0
    if len(all_rates) > 1 and len(all_ratings) > 1:
        n = len(all_rates)
        sum_xy = sum(all_rates[i] * all_ratings[i] for i in range(n))
        sum_x = sum(all_rates)
        sum_y = sum(all_ratings)
        sum_x2 = sum(r * r for r in all_rates)
        sum_y2 = sum(r * r for r in all_ratings)
        
        numerator = n * sum_xy - sum_x * sum_y
        denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)) ** 0.5
        
        if denominator != 0:
            correlation = numerator / denominator
    
    # Prepare answer
    answer = {
        'adr_premium_percentage': round(adr_premium, 1),
        'most_expensive_accommodation_type': most_expensive_type if most_expensive_type else 'Hotel',
        'premium_hotels_below_75th_percentile': filtered_premium,
        'rating_rate_correlation': round(correlation, 3),
        'data_quality_notes': {
            'high_rated_hotels_analyzed': len(high_rated_rates),
            'low_rated_hotels_analyzed': len(low_rated_rates),
            'american_traveler_trips_analyzed': american_trips,
            'total_hotels_in_correlation': len(all_rates)
        }
    }
    
    return tools['submit_result'](answer)
