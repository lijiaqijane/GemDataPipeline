def solve(tools):
    hotels = tools['search_hotel_directory']('Paris')
    luxury_hotels = []
    for hotel in hotels:
        if isinstance(hotel, dict):
            amenities = hotel.get('amenities', '').lower()
            stars = hotel.get('stars', '')
            if 'pool' in amenities and 'jacuzzi' in amenities and stars == '5':
                name = hotel.get('name', '')
                if name:
                    luxury_hotels.append({
                        'name': name,
                        'stars': stars,
                        'amenities': hotel.get('amenities', ''),
                        'price': hotel.get('price', '')
                    })
    pricing = tools['search_hotel_pricing']('Paris')
    for hotel in luxury_hotels:
        seasonal_rate = 'Not available'
        for price_rec in pricing:
            if isinstance(price_rec, dict):
                if price_rec.get('hotel_name', '') == hotel['name']:
                    rate = price_rec.get('rate', '')
                    if rate:
                        seasonal_rate = str(rate)
                        hotel['price'] = seasonal_rate
                    break
        hotel['seasonal_rate'] = seasonal_rate
        try:
            rate_val = float(seasonal_rate.replace('$', '').replace(',', '')) if seasonal_rate != 'Not available' else 0
            hotel['3_night_cost'] = f'${rate_val * 3:.2f}'
        except:
            hotel['3_night_cost'] = 'Not available'
    attractions = tools['search_tripadvisor_attractions']('Paris')
    top_attractions = []
    for attr in attractions:
        if isinstance(attr, dict):
            rating = attr.get('rating', '')
            entry_fee = attr.get('entry_fee', '')
            try:
                rating_val = float(rating)
                if rating_val >= 4.5 and entry_fee and 'free' not in entry_fee.lower():
                    top_attractions.append({
                        'name': attr.get('sub_category', ''),
                        'description': attr.get('description', ''),
                        'rating': rating,
                        'entry_fee': entry_fee
                    })
            except:
                continue
    for attr in top_attractions:
        fee_str = attr['entry_fee']
        try:
            fee_val = float(''.join(c for c in fee_str if c.isdigit() or c == '.'))
            attr['cost_for_2'] = f'${fee_val * 2:.2f}'
        except:
            attr['cost_for_2'] = 'Not available'
    total_hotel = 0
    for hotel in luxury_hotels:
        try:
            total_hotel += float(hotel['3_night_cost'].replace('$', '').replace(',', '')) if hotel['3_night_cost'] != 'Not available' else 0
        except:
            pass
    total_attractions = 0
    for attr in top_attractions:
        try:
            total_attractions += float(attr['cost_for_2'].replace('$', '').replace(',', '')) if attr['cost_for_2'] != 'Not available' else 0
        except:
            pass
    answer = {
        'package_name': 'Luxury Paris Experience',
        'hotels': [{
            'name': h['name'],
            'stars': h['stars'],
            'amenities': h['amenities'],
            'seasonal_rate': h['seasonal_rate'],
            '3_night_cost': h['3_night_cost']
        } for h in luxury_hotels],
        'attractions': [{
            'name': a['name'],
            'description': a['description'],
            'rating': a['rating'],
            'entry_fee': a['entry_fee'],
            'cost_for_2': a['cost_for_2']
        } for a in top_attractions],
        'cost_summary': {
            'total_hotel_3_nights': f'${total_hotel:.2f}',
            'total_attractions_2_people': f'${total_attractions:.2f}',
            'grand_total': f'${total_hotel + total_attractions:.2f}'
        }
    }
    return tools['submit_result'](answer)
