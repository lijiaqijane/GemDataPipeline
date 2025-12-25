def solve(tools):
    # Step 1: Find Paris hotels with both 'pool' and 'wifi' amenities
    listings = tools['search_hotel_listings']('Paris, France')
    paris_hotels = []
    if isinstance(listings, list):
        for hotel in listings:
            if isinstance(hotel, dict):
                amenities = hotel.get('amenities', '')
                city = hotel.get('city', '')
                country = hotel.get('country', '')
                if amenities and city and country:
                    amenities_lower = amenities.lower()
                    if 'pool' in amenities_lower and 'wifi' in amenities_lower:
                        if 'paris' in city.lower() and 'france' in country.lower():
                            paris_hotels.append({
                                'name': str(hotel.get('name', '')),
                                'stars': str(hotel.get('stars', '')),
                                'amenities': amenities,
                                'city': city,
                                'country': country
                            })
    
    # Sort hotels alphabetically
    paris_hotels.sort(key=lambda x: x['name'].lower())
    
    analysis_list = []
    
    for hotel in paris_hotels:
        hotel_name = hotel['name']
        stars = hotel['stars']
        amenities = hotel['amenities']
        
        # Step 2: Search travel reviews for this hotel
        reviews = tools['search_travel_reviews'](hotel_name)
        all_reviews = []
        most_recent_review = None
        
        if isinstance(reviews, list):
            for review in reviews:
                if isinstance(review, dict):
                    # Check if review mentions hotel name or provider_name
                    provider = review.get('provider_name', '')
                    if hotel_name.lower() in provider.lower() or hotel_name.lower() in str(review.get('review_text', '')).lower():
                        all_reviews.append(review)
        
        # Filter reviews: require at least 3 reviews before calculating average
        if len(all_reviews) >= 3:
            # Find most recent review by review_date
            valid_reviews = []
            for rev in all_reviews:
                if rev.get('review_date'):
                    valid_reviews.append(rev)
            
            if valid_reviews:
                valid_reviews.sort(key=lambda x: x.get('review_date', ''), reverse=True)
                most_recent = valid_reviews[0]
                most_recent_review = {
                    'review_date': str(most_recent.get('review_date', '')),
                    'review_title': str(most_recent.get('review_title', '')),
                    'rating': str(most_recent.get('rating', '')),
                    'provider_name': str(most_recent.get('provider_name', ''))
                }
            
            # Calculate average rating from all reviews
            total_rating = 0
            count = 0
            for rev in all_reviews:
                rating_str = rev.get('rating', '')
                if rating_str and rating_str.replace('.', '').isdigit():
                    try:
                        total_rating += float(rating_str)
                        count += 1
                    except:
                        pass
            
            average_rating = round(total_rating / count, 2) if count > 0 else 0.0
            
            # Step 3: Search hotel pricing for Deluxe or Executive room
            pricing = tools['search_hotel_pricing'](hotel_name)
            current_rate = ''
            
            if isinstance(pricing, list):
                deluxe_rates = []
                executive_rates = []
                
                for price in pricing:
                    if isinstance(price, dict):
                        room_type = str(price.get('room_type', '')).lower()
                        rate_val = price.get('rate', '')
                        if 'deluxe' in room_type and rate_val:
                            deluxe_rates.append(rate_val)
                        elif 'executive' in room_type and rate_val:
                            executive_rates.append(rate_val)
                
                # Prioritize Deluxe over Executive
                if deluxe_rates:
                    current_rate = str(deluxe_rates[0])
                elif executive_rates:
                    current_rate = str(executive_rates[0])
            
            # Calculate rating discrepancy
            try:
                star_float = float(stars) if stars else 0.0
                discrepancy = round(average_rating - star_float, 2)
            except:
                discrepancy = 0.0
            
            analysis_list.append({
                'hotel_name': hotel_name,
                'stars': stars,
                'amenities': amenities,
                'most_recent_review': most_recent_review if most_recent_review else {
                    'review_date': '',
                    'review_title': '',
                    'rating': '',
                    'provider_name': ''
                },
                'current_deluxe_rate': current_rate,
                'average_review_rating': average_rating,
                'rating_discrepancy': discrepancy
            })
    
    # Find hotel with highest positive discrepancy
    highest_hotel = ''
    highest_discrepancy = -999.0
    
    for item in analysis_list:
        disc = item.get('rating_discrepancy', 0.0)
        if disc > highest_discrepancy:
            highest_discrepancy = disc
            highest_hotel = item.get('hotel_name', '')
    
    # Prepare final answer with proper string types
    answer = {
        'analysis': analysis_list,
        'hotel_with_highest_positive_discrepancy': str(highest_hotel),
        'total_hotels_analyzed': str(len(analysis_list))
    }
    
    return tools['submit_result'](answer)
