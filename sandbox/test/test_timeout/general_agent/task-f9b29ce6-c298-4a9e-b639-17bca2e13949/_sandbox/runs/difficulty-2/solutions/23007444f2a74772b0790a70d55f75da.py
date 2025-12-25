def solve(tools):
    # Step 1: Find 4-star Paris hotel with pool and wifi
    hotels = tools['search_hotels']('Paris', 50)
    target_hotel = None
    
    for hotel in hotels:
        if isinstance(hotel, dict):
            stars = str(hotel.get('stars', '')).strip()
            city = str(hotel.get('city', '')).lower()
            amenities = str(hotel.get('amenities', '')).lower()
            
            if ('4' in stars or 'four' in stars.lower()) and 'paris' in city:
                if 'pool' in amenities and 'wifi' in amenities:
                    target_hotel = {
                        'name': hotel.get('name', ''),
                        'stars': '4',
                        'amenities': hotel.get('amenities', ''),
                        'price': hotel.get('price', ''),
                        'city': hotel.get('city', '')
                    }
                    break
    
    # Step 2: Find Airbnb experience in Paris with rating >= 4.5
    attractions = tools['search_attractions']('Paris', 50)
    airbnb_exp = None
    
    for attr in attractions:
        if isinstance(attr, dict):
            location = str(attr.get('location', '')).lower()
            rating_str = str(attr.get('rating', '0'))
            
            # Check if it's an Airbnb experience (has experience_title)
            if 'paris' in location and 'experience_title' in attr:
                try:
                    rating = float(rating_str)
                    if rating >= 4.5:
                        airbnb_exp = {
                            'title': attr.get('experience_title', ''),
                            'rating': rating_str,
                            'price': attr.get('price', ''),
                            'location': attr.get('location', '')
                        }
                        break
                except:
                    continue
    
    # Step 3: Find Paris attraction with entry fee < $30
    attraction_target = None
    for attr in attractions:
        if isinstance(attr, dict):
            location = str(attr.get('location', '')).lower()
            entry_fee = str(attr.get('entry_fee', '$100'))
            
            if 'paris' in location and 'entry_fee' in attr:
                # Extract numeric value from entry fee
                import re
                fee_match = re.search(r'\$?([0-9]+\.?[0-9]*)', entry_fee)
                if fee_match:
                    fee_value = float(fee_match.group(1))
                    if fee_value < 30:
                        attraction_target = {
                            'name': attr.get('sub_category', attr.get('experience_title', '')),
                            'entry_fee': entry_fee,
                            'rating': attr.get('rating', ''),
                            'location': attr.get('location', '')
                        }
                        break
    
    # Step 4: Find historical American travelers to Paris who stayed in hotels
    travel_data = tools['search_travel_recommendations']('', 100)
    american_travelers = []
    
    for trip in travel_data:
        if isinstance(trip, dict):
            nationality = str(trip.get('traveler_nationality', '')).lower()
            destination = str(trip.get('destination', '')).lower()
            accommodation = str(trip.get('accommodation_type', '')).lower()
            
            if 'american' in nationality and 'paris' in destination and 'hotel' in accommodation:
                american_travelers.append({
                    'name': trip.get('traveler_name', ''),
                    'nationality': trip.get('traveler_nationality', ''),
                    'destination': trip.get('destination', ''),
                    'accommodation_type': trip.get('accommodation_type', '')
                })
                
                if len(american_travelers) >= 3:
                    break
    
    # Step 5: Verify all criteria are met
    verification = []
    
    if target_hotel:
        verification.append("✓ Found 4-star Paris hotel with pool and wifi")
    else:
        verification.append("✗ No suitable hotel found")
        
    if airbnb_exp:
        verification.append(f"✓ Found Airbnb experience with rating {airbnb_exp['rating']} (>= 4.5)")
    else:
        verification.append("✗ No suitable Airbnb experience found")
        
    if attraction_target:
        verification.append(f"✓ Found attraction with entry fee {attraction_target['entry_fee']} (< $30)")
    else:
        verification.append("✗ No suitable attraction found")
        
    if len(american_travelers) >= 3:
        verification.append(f"✓ Found {len(american_travelers)} American travelers to Paris who stayed in hotels")
    else:
        verification.append(f"✗ Only found {len(american_travelers)} American travelers (need 3)")
    
    # Step 6: Create final package
    package_name = "Paris Premium Experience Package"
    
    answer = {
        'package_name': package_name,
        'hotel': target_hotel if target_hotel else {
            'name': 'No suitable hotel found',
            'stars': '',
            'amenities': '',
            'price': '',
            'city': ''
        },
        'airbnb_experience': airbnb_exp if airbnb_exp else {
            'title': 'No suitable experience found',
            'rating': '',
            'price': '',
            'location': ''
        },
        'attraction': attraction_target if attraction_target else {
            'name': 'No suitable attraction found',
            'entry_fee': '',
            'rating': '',
            'location': ''
        },
        'historical_travelers': american_travelers if american_travelers else [],
        'verification_summary': ' | '.join(verification)
    }
    
    return tools['submit_result'](answer)
