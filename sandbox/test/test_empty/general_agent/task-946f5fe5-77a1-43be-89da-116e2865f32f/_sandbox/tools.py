import json
import csv
import os
import re
from typing import List, Dict, Any
import random

# Fallback shim if mcp is not available
try:
    import mcp
except ImportError:
    # Create a minimal identity decorator
    class MockMCP:
        @staticmethod
        def tool(func):
            return func
    mcp = MockMCP()

# Set seed for deterministic behavior
random.seed(0)

@mcp.tool
def search_hotel_pricing(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Search hotel pricing data from seasonal pricing CSV.
    Query can match hotel names, room types, dates, or special offers.
    """
    results = []
    file_path = "data/scrape-london-paris-hotel-seasonal-prici.csv"
    
    if not os.path.exists(file_path):
        return [{"error": f"File not found: {file_path}"}]
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Normalize query for case-insensitive matching
                query_lower = query.lower()
                match_found = False
                
                # Check each field for matches
                for key, value in row.items():
                    if value and query_lower in value.lower():
                        match_found = True
                        break
                
                if match_found:
                    # Convert rate to numeric if possible
                    rate = row.get('rate', '')
                    try:
                        rate_num = float(rate) if rate else 0.0
                    except ValueError:
                        rate_num = 0.0
                    
                    # Convert rating to numeric if possible
                    rating = row.get('guest_rating', '')
                    try:
                        rating_num = float(rating) if rating else 0.0
                    except ValueError:
                        rating_num = 0.0
                    
                    results.append({
                        'hotel_name': row.get('hotel_name', ''),
                        'room_type': row.get('room_type', ''),
                        'date': row.get('date', ''),
                        'rate': rate,
                        'rate_numeric': rate_num,
                        'availability': row.get('availability', ''),
                        'guest_rating': rating,
                        'guest_rating_numeric': rating_num,
                        'special_offer': row.get('special_offer', '')
                    })
                    
                    if len(results) >= max_results:
                        break
    except Exception as e:
        return [{"error": f"Error reading CSV: {str(e)}"}]
    
    return results

@mcp.tool
def search_hotel_listings(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Search hotel listings data with amenities, stars, city, country, and pricing.
    Query can match hotel names, cities, countries, or amenities.
    """
    results = []
    file_path = "data/hotels-csv-bixbydevelopers-sample-templa.csv"
    
    if not os.path.exists(file_path):
        return [{"error": f"File not found: {file_path}"}]
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Normalize query for case-insensitive matching
                query_lower = query.lower()
                match_found = False
                
                # Check key fields for matches
                search_fields = ['name', 'city', 'country', 'amenities']
                for field in search_fields:
                    value = row.get(field, '')
                    if value and query_lower in value.lower():
                        match_found = True
                        break
                
                if match_found:
                    # Convert stars and price to numeric if possible
                    stars = row.get('stars', '')
                    try:
                        stars_num = int(stars) if stars else 0
                    except ValueError:
                        stars_num = 0
                    
                    price = row.get('price', '')
                    try:
                        price_num = float(price) if price else 0.0
                    except ValueError:
                        price_num = 0.0
                    
                    # Parse amenities into list
                    amenities_str = row.get('amenities', '')
                    amenities_list = [a.strip() for a in amenities_str.split(',')] if amenities_str else []
                    
                    results.append({
                        'name': row.get('name', ''),
                        'amenities': amenities_str,
                        'amenities_list': amenities_list,
                        'stars': stars,
                        'stars_numeric': stars_num,
                        'city': row.get('city', ''),
                        'country': row.get('country', ''),
                        'photo_url': row.get('photo', ''),
                        'price': price,
                        'price_numeric': price_num,
                        'website': row.get('website', '')
                    })
                    
                    if len(results) >= max_results:
                        break
    except Exception as e:
        return [{"error": f"Error reading CSV: {str(e)}"}]
    
    return results

@mcp.tool
def search_travel_reviews(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Search travel reviews from Trivago dataset.
    Query can match destination locations, provider names, review text, or ratings.
    """
    results = []
    file_path = "data/trivago-travel-reviews-dataset.csv"
    
    if not os.path.exists(file_path):
        return [{"error": f"File not found: {file_path}"}]
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Normalize query for case-insensitive matching
                query_lower = query.lower()
                match_found = False
                
                # Check key fields for matches
                search_fields = ['destination_location', 'provider_name', 'review_title', 
                               'review_text', 'trip_type', 'service_type']
                for field in search_fields:
                    value = row.get(field, '')
                    if value and query_lower in value.lower():
                        match_found = True
                        break
                
                # Also check rating if query is numeric
                try:
                    if query_lower and row.get('rating'):
                        rating_val = float(row['rating'])
                        query_num = float(query_lower)
                        if abs(rating_val - query_num) <= 0.5:  # Allow some tolerance
                            match_found = True
                except ValueError:
                    pass
                
                if match_found:
                    # Convert numeric fields
                    rating = row.get('rating', '')
                    try:
                        rating_num = float(rating) if rating else 0.0
                    except ValueError:
                        rating_num = 0.0
                    
                    price_paid = row.get('price_paid_usd', '')
                    try:
                        price_num = float(price_paid) if price_paid else 0.0
                    except ValueError:
                        price_num = 0.0
                    
                    word_count = row.get('word_count', '')
                    try:
                        word_count_num = int(word_count) if word_count else 0
                    except ValueError:
                        word_count_num = 0
                    
                    # Parse aspect scores if available
                    aspect_scores = row.get('aspect_scores', '')
                    aspect_dict = {}
                    if aspect_scores:
                        for part in aspect_scores.split(','):
                            if ':' in part:
                                key_val = part.split(':', 1)
                                if len(key_val) == 2:
                                    aspect_dict[key_val[0].strip()] = key_val[1].strip()
                    
                    results.append({
                        'entry_reference': row.get('entry_reference', ''),
                        'review_date': row.get('review_date', ''),
                        'review_title': row.get('review_title', ''),
                        'review_text': row.get('review_text', ''),
                        'rating': rating,
                        'rating_numeric': rating_num,
                        'trip_type': row.get('trip_type', ''),
                        'service_type': row.get('service_type', ''),
                        'provider_name': row.get('provider_name', ''),
                        'destination_location': row.get('destination_location', ''),
                        'price_paid_usd': price_paid,
                        'price_paid_numeric': price_num,
                        'booking_platform': row.get('booking_platform', ''),
                        'language': row.get('language', ''),
                        'word_count': word_count,
                        'word_count_numeric': word_count_num,
                        'provider_response': row.get('provider_response', ''),
                        'images_present': row.get('images_present', ''),
                        'response_date': row.get('response_date', ''),
                        'aspect_scores': aspect_scores,
                        'aspect_scores_dict': aspect_dict,
                        'review_source': row.get('review_source', ''),
                        'price_currency': row.get('price_currency', ''),
                        'travel_date': row.get('travel_date', '')
                    })
                    
                    if len(results) >= max_results:
                        break
    except Exception as e:
        return [{"error": f"Error reading CSV: {str(e)}"}]
    
    return results

@mcp.tool
def search_trip_details(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Search trip details dataset for travel itineraries.
    Query can match destinations, traveler names, accommodation types, etc.
    """
    results = []
    file_path = "data/travel-details-dataset-csv.csv"
    
    if not os.path.exists(file_path):
        return [{"error": f"File not found: {file_path}"}]
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Normalize query for case-insensitive matching
                query_lower = query.lower()
                match_found = False
                
                # Check key fields for matches
                search_fields = ['destination', 'traveler_name', 'traveler_nationality',
                               'accommodation_type', 'transportation_type']
                for field in search_fields:
                    value = row.get(field, '')
                    if value and query_lower in value.lower():
                        match_found = True
                        break
                
                if match_found:
                    # Convert numeric fields
                    duration = row.get('duration_days', '')
                    try:
                        duration_num = int(duration) if duration else 0
                    except ValueError:
                        duration_num = 0
                    
                    traveler_age = row.get('traveler_age', '')
                    try:
                        age_num = int(traveler_age) if traveler_age else 0
                    except ValueError:
                        age_num = 0
                    
                    # Clean cost fields (remove $, commas, etc.)
                    acc_cost = row.get('accommodation_cost', '')
                    trans_cost = row.get('transportation_cost', '')
                    
                    def clean_cost(cost_str):
                        if not cost_str:
                            return 0.0
                        # Remove currency symbols and commas
                        cleaned = re.sub(r'[^\d.]', '', str(cost_str))
                        try:
                            return float(cleaned) if cleaned else 0.0
                        except ValueError:
                            return 0.0
                    
                    acc_cost_num = clean_cost(acc_cost)
                    trans_cost_num = clean_cost(trans_cost)
                    total_cost_num = acc_cost_num + trans_cost_num
                    
                    results.append({
                        'trip_id': row.get('trip_id', ''),
                        'destination': row.get('destination', ''),
                        'start_date': row.get('start_date', ''),
                        'end_date': row.get('end_date', ''),
                        'duration_days': duration,
                        'duration_days_numeric': duration_num,
                        'traveler_name': row.get('traveler_name', ''),
                        'traveler_age': traveler_age,
                        'traveler_age_numeric': age_num,
                        'traveler_gender': row.get('traveler_gender', ''),
                        'traveler_nationality': row.get('traveler_nationality', ''),
                        'accommodation_type': row.get('accommodation_type', ''),
                        'accommodation_cost': acc_cost,
                        'accommodation_cost_numeric': acc_cost_num,
                        'transportation_type': row.get('transportation_type', ''),
                        'transportation_cost': trans_cost,
                        'transportation_cost_numeric': trans_cost_num,
                        'total_cost_numeric': total_cost_num
                    })
                    
                    if len(results) >= max_results:
                        break
    except Exception as e:
        return [{"error": f"Error reading CSV: {str(e)}"}]
    
    return results

@mcp.tool
def submit_result(result) -> Any:
    """
    Submit a result payload and persist it to submitted_result.json for inspection.
    """
    output_file = "submitted_result.json"
    
    try:
        # Write the result to JSON file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        # Return the result as confirmation
        return result
    except Exception as e:
        return {"error": f"Failed to write result to {output_file}: {str(e)}"}
