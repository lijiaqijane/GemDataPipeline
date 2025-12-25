def solve(tools):
    hotels = tools['search_hotels']('Paris 3 star', 50)
    ratings = tools['get_hotel_star_ratings']('paris', 20)
    filtered = []
    for h in hotels:
        if isinstance(h, dict):
            stars = str(h.get('stars', '')).strip()
            city = str(h.get('city', '')).strip()
            if stars == '3' and city.lower() == 'paris':
                name = str(h.get('name', '')).strip()
                amenities = str(h.get('amenities', '')).strip()
                price = str(h.get('price', '')).strip()
                if name and amenities and price:
                    filtered.append({
                        'name': name,
                        'stars': '3',
                        'amenities': amenities,
                        'price': price,
                        'city': 'Paris'
                    })
    answer = {'hotels': filtered, 'count': str(len(filtered))}
    return tools['submit_result'](answer)
