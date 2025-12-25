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
    
    analysis_list = []
    
    for hotel in paris_hotels:
        hotel_name = hotel['name']
        stars = hotel['stars']
        amenities = hotel['amenities']
        
        # Step 2: Find most recent review for this hotel
        reviews = tools['search_travel_reviews'](hotel_name)
        most_recent_review = None
        all_ratings = []
        
        if isinstance(reviews, list):
            for review in reviews:
                if isinstance(review, dict):
                    provider = review.get('provider_name', '')
                    rating_str = review.get('rating', '')
                    review_date = review.get('review_date', '')
                    
                    # Check if review mentions hotel name or provider matches
                    if hotel_name.lower() in provider.lower() or hotel_name.lower() in str(review.get('review_text', '')).lower():
                        try:
                            rating = float(rating_str)
                            all_ratings.append(rating)
                        except:
                            pass
                        
                        # Find most recent by date
                        if review_date:
                            if most_recent_review is None:
                                most_recent_review = review
                            else:
                                # Simple date comparison (assuming format like '1/10/2025')
                                try:
                                    new_date_parts = review_date.split('/')
                                    old_date_parts = most_recent_review.get('review_date', '').split('/')
                                    if len(new_date_parts) == 3 and len(old_date_parts) == 3:
                                        new_year = int(new_date_parts[2])
                                        new_month = int(new_date_parts[0])
                                        old_year = int(old_date_parts[2])
                                        old_month = int(old_date_parts[0])
                                        if new_year > old_year or (new_year == old_year and new_month > old_month):
                                            most_recent_review = review
                                except:
                                    pass
        
        # Step 3: Find current deluxe/executive room rate
        pricing = tools['search_hotel_pricing'](hotel_name)
        current_rate = 'Not found'
        
        if isinstance(pricing, list):
            for price_entry in pricing:
                if isinstance(price_entry, dict):
                    room_type = price_entry.get('room_type', '')
                    rate = price_entry.get('rate', '')
                    date = price_entry.get('date', '')
                    
                    if room_type and ('deluxe' in room_type.lower() or 'executive' in room_type.lower()):
                        if rate and rate != 'Sold Out' and rate != 'Not available':
                            current_rate = str(rate)
                            break
        
        # Step 4: Calculate average rating and discrepancy
        avg_rating = 0.0
        discrepancy = 0.0
        
        if all_ratings:
            avg_rating = sum(all_ratings) / len(all_ratings)
            try:
                star_float = float(stars)
                discrepancy = avg_rating - star_float
            except:
                discrepancy = avg_rating
        
        # Prepare most recent review details
        review_details = {
            'review_date': '',
            'review_title': '',
            'rating': '',
            'provider_name': ''
        }
        
        if most_recent_review:
            review_details = {
                'review_date': str(most_recent_review.get('review_date', '')),
                'review_title': str(most_recent_review.get('review_title', '')),
                'rating': str(most_recent_review.get('rating', '')),
                'provider_name': str(most_recent_review.get('provider_name', ''))
            }
        
        analysis_list.append({
            'hotel_name': hotel_name,
            'stars': stars,
            'amenities': amenities,
            'most_recent_review': review_details,
            'current_deluxe_rate': current_rate,
            'average_review_rating': round(avg_rating, 2),
            'rating_discrepancy': round(discrepancy, 2)
        })
    
    # Step 5: Find hotel with highest positive discrepancy
    highest_hotel = ''
    highest_discrepancy = -999.0
    
    for item in analysis_list:
        disc = item.get('rating_discrepancy', 0.0)
        if disc > highest_discrepancy:
            highest_discrepancy = disc
            highest_hotel = item.get('hotel_name', '')
    
    answer = {
        'analysis': analysis_list,
        'hotel_with_highest_positive_discrepancy': highest_hotel,
        'total_hotels_analyzed': len(analysis_list)
    }
    
    return tools['submit_result'](answer)
