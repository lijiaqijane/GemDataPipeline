def solve(tools):
    # First search for Paris hotels with pool amenities
    listings = tools['search_hotel_listings']('Paris, France pool')
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
                            'name': name,
                            'stars': stars,
                            'price': price,
                            'amenities': amenities
                        })
    # Second tool: search pricing data for additional verification
    pricing = tools['search_hotel_pricing']('Paris')
    # We could cross-reference but for now just use listings data
    answer = {
        'hotels': hotels_with_pool,
        'total_count': len(hotels_with_pool)
    }
    return tools['submit_result'](answer)
