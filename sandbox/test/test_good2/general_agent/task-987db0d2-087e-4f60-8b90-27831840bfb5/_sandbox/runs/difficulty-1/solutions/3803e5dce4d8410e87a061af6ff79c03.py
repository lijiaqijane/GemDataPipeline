def solve(tools):
    # Search for hotel pricing data with special offers in Paris
    hotel_data = tools['search_hotel_pricing']('Paris special offer')
    
    # Search Paris hotel guides for additional context
    guide_data = tools['search_paris_hotel_guides']('hotel')
    
    # Filter hotels: must be Available and have a non-empty special offer
    available_hotels = []
    
    if isinstance(hotel_data, list):
        for hotel in hotel_data:
            if isinstance(hotel, dict) and 'error' not in hotel:
                availability = hotel.get('availability', '')
                special_offer = hotel.get('special_offer', '')
                hotel_name = hotel.get('hotel_name', '')
                room_type = hotel.get('room_type', '')
                rate = hotel.get('rate', '')
                
                # Check criteria: Available and has a real special offer
                if (availability == 'Available' and 
                    special_offer and 
                    special_offer.lower() != 'none' and 
                    hotel_name and room_type and rate):
                    
                    # Format rate as string with currency if needed
                    rate_str = str(rate)
                    if not rate_str.startswith('$') and rate_str.replace('.', '').isdigit():
                        rate_str = f'${rate_str}'
                    
                    available_hotels.append({
                        'hotel_name': hotel_name,
                        'room_type': room_type,
                        'rate': rate_str,
                        'special_offer': special_offer
                    })
    
    # Build answer matching required format
    answer = {
        'available_hotels_with_offers': available_hotels,
        'total_count': len(available_hotels)  # integer, not string
    }
    
    return tools['submit_result'](answer)
