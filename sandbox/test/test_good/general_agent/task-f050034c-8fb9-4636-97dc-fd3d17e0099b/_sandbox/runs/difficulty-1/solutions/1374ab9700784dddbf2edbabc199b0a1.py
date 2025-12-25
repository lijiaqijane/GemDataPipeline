def solve(tools):
    hotels = tools['search_hotel_directory']('Paris')
    pool_hotels = []
    for hotel in hotels:
        if isinstance(hotel, dict):
            amenities = hotel.get('amenities', '')
            if 'pool' in amenities.lower():
                name = hotel.get('name', '')
                stars = hotel.get('stars', '')
                price = hotel.get('price', '')
                amenities_str = hotel.get('amenities', '')
                if name and stars and price:
                    pool_hotels.append({
                        'name': name,
                        'stars': stars,
                        'price': price,
                        'amenities': amenities_str
                    })
    pricing = tools['search_hotel_pricing']('Paris')
    for price_rec in pricing:
        if isinstance(price_rec, dict):
            hotel_name = price_rec.get('hotel_name', '')
            if hotel_name:
                for hotel in pool_hotels:
                    if hotel['name'] == hotel_name:
                        rate = price_rec.get('rate', '')
                        if rate:
                            hotel['price'] = str(rate)
    for hotel in pool_hotels:
        hotel['price'] = str(hotel['price'])
        hotel['stars'] = str(hotel['stars'])
    answer = {
        'hotels_found': pool_hotels,
        'count': str(len(pool_hotels))
    }
    return tools['submit_result'](answer)
