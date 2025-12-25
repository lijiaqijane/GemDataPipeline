def solve(tools):
    # First tool: search hotel pricing data for Paris hotels with special offers
    hotel_data = tools['search_hotel_pricing']('Paris special offer')
    
    # Second tool: search Paris hotel guides for additional information
    guide_data = tools['search_paris_hotel_guides']('hotel')
    
    # Process hotel data to find available hotels with special offers
    available_hotels = []
    
    # Check if hotel_data is valid and not an error
    if isinstance(hotel_data, list) and len(hotel_data) > 0:
        for hotel in hotel_data:
            # Skip if hotel is an error record
            if isinstance(hotel, dict) and 'error' not in hotel:
                # Check if hotel is available and has a special offer
                availability = hotel.get('availability', '')
                special_offer = hotel.get('special_offer', '')
                hotel_name = hotel.get('hotel_name', '')
                room_type = hotel.get('room_type', '')
                rate = hotel.get('rate', '')
                
                # Filter: must be available and have a non-empty special offer (not 'None')
                if (availability == 'Available' and 
                    special_offer and 
                    special_offer.lower() != 'none' and 
                    special_offer.strip() != '' and
                    hotel_name and room_type and rate):
                    
                    available_hotels.append({
                        'hotel_name': hotel_name,
                        'room_type': room_type,
                        'rate': rate,
                        'special_offer': special_offer
                    })
    
    # Create answer according to required format
    answer = {
        'available_hotels_with_offers': available_hotels,
        'total_count': len(available_hotels)
    }
    
    return tools['submit_result'](answer)
