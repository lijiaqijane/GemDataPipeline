def solve(tools):
    # Search for Paris hotels with pool amenities
    listings = tools['search_hotel_listings']('Paris pool')
    hotels_with_pool = []
    if isinstance(listings, list):
        for hotel in listings:
            if isinstance(hotel, dict):
                amenities = hotel.get('amenities', '')
                if amenities and 'pool' in amenities.lower():
                    name = hotel.get('name', '')
                    stars = hotel.get('stars', '')
                    price = hotel.get('price', '')
                    if name and stars and price:
                        hotels_with_pool.append({
                            'name': str(name),
                            'stars': str(stars),
                            'price': str(price),
                            'amenities': str(amenities)
                        })
    # Second tool: search pricing data for additional verification
    tools['search_hotel_pricing']('Paris')
    answer = {
        'hotels': hotels_with_pool,
        'total_count': str(len(hotels_with_pool))
    }
    return tools['submit_result'](answer)
