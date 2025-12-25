import json
import csv
import os
import re
from typing import List, Dict, Any
import random

# Set seed for deterministic behavior if randomness is used
random.seed(0)

# Fallback shim if mcp is not available
try:
    import mcp
except ImportError:
    # Create a minimal fallback decorator
    class MockMCP:
        @staticmethod
        def tool(func):
            return func
    mcp = MockMCP()

@mcp.tool
def search_hotel_pricing(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Search hotel pricing data from seasonal pricing CSV.
    Returns hotels matching query in name, room_type, or special_offer.
    """
    results = []
    csv_path = "data/scrape-london-paris-hotel-seasonal-prici.csv"
    
    if not os.path.exists(csv_path):
        return [{"error": f"File not found: {csv_path}"}]
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Clean up field names (strip whitespace)
                row = {k.strip(): v for k, v in row.items()}
                
                # Check if query matches any relevant field
                query_lower = query.lower()
                matches = False
                for field in ['hotel_name', 'room_type', 'special_offer']:
                    if field in row and query_lower in row[field].lower():
                        matches = True
                        break
                
                if matches:
                    # Convert rate to numeric if possible
                    rate = row.get('rate', '0')
                    try:
                        rate_num = float(rate)
                    except (ValueError, TypeError):
                        rate_num = 0
                    
                    results.append({
                        'hotel_name': row.get('hotel_name', ''),
                        'room_type': row.get('room_type', ''),
                        'date': row.get('date', ''),
                        'rate': rate,
                        'rate_numeric': rate_num,
                        'availability': row.get('availability', ''),
                        'guest_rating': row.get('guest_rating', ''),
                        'special_offer': row.get('special_offer', '')
                    })
                    
                    if len(results) >= max_results:
                        break
    except Exception as e:
        return [{"error": f"Error reading CSV: {str(e)}"}]
    
    return results

@mcp.tool
def search_hotel_directory(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Search hotel directory data with amenities, stars, and pricing.
    Returns hotels matching query in name, city, country, or amenities.
    """
    results = []
    csv_path = "data/hotels-csv-bixbydevelopers-sample-templa.csv"
    
    if not os.path.exists(csv_path):
        return [{"error": f"File not found: {csv_path}"}]
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Clean up field names
                row = {k.strip(): v for k, v in row.items()}
                
                # Check if query matches any relevant field
                query_lower = query.lower()
                matches = False
                for field in ['name', 'city', 'country', 'amenities']:
                    if field in row and query_lower in row[field].lower():
                        matches = True
                        break
                
                if matches:
                    # Convert stars and price to numeric if possible
                    stars = row.get('stars', '0')
                    price = row.get('price', '0')
                    try:
                        stars_num = int(stars)
                    except (ValueError, TypeError):
                        stars_num = 0
                    
                    try:
                        price_num = float(price)
                    except (ValueError, TypeError):
                        price_num = 0
                    
                    results.append({
                        'name': row.get('name', ''),
                        'amenities': row.get('amenities', ''),
                        'stars': stars,
                        'stars_numeric': stars_num,
                        'city': row.get('city', ''),
                        'country': row.get('country', ''),
                        'photo': row.get('photo', ''),
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
def search_travel_recommendations(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Search travel recommendation data with trip details.
    Returns trips matching query in destination, traveler_name, or accommodation_type.
    """
    results = []
    csv_path = "data/travel-recommendation-system-travel-deta.csv"
    
    if not os.path.exists(csv_path):
        return [{"error": f"File not found: {csv_path}"}]
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Clean up field names
                row = {k.strip(): v for k, v in row.items()}
                
                # Check if query matches any relevant field
                query_lower = query.lower()
                matches = False
                for field in ['destination', 'traveler_name', 'accommodation_type', 'traveler_nationality']:
                    if field in row and query_lower in row[field].lower():
                        matches = True
                        break
                
                if matches:
                    # Convert numeric fields
                    duration_days = row.get('duration_days', '0')
                    traveler_age = row.get('traveler_age', '0')
                    accommodation_cost = row.get('accommodation_cost', '0')
                    transportation_cost = row.get('transportation_cost', '0')
                    
                    # Clean cost fields (remove $ and commas)
                    for cost_field in ['accommodation_cost', 'transportation_cost']:
                        if cost_field in row:
                            row[cost_field] = re.sub(r'[^\d.]', '', row[cost_field])
                    
                    try:
                        duration_num = int(duration_days)
                    except (ValueError, TypeError):
                        duration_num = 0
                    
                    try:
                        age_num = int(traveler_age)
                    except (ValueError, TypeError):
                        age_num = 0
                    
                    try:
                        acc_cost_num = float(row.get('accommodation_cost', '0'))
                    except (ValueError, TypeError):
                        acc_cost_num = 0
                    
                    try:
                        trans_cost_num = float(row.get('transportation_cost', '0'))
                    except (ValueError, TypeError):
                        trans_cost_num = 0
                    
                    total_cost = acc_cost_num + trans_cost_num
                    
                    results.append({
                        'trip_id': row.get('trip_id', ''),
                        'destination': row.get('destination', ''),
                        'start_date': row.get('start_date', ''),
                        'end_date': row.get('end_date', ''),
                        'duration_days': duration_days,
                        'duration_numeric': duration_num,
                        'traveler_name': row.get('traveler_name', ''),
                        'traveler_age': traveler_age,
                        'traveler_age_numeric': age_num,
                        'traveler_gender': row.get('traveler_gender', ''),
                        'traveler_nationality': row.get('traveler_nationality', ''),
                        'accommodation_type': row.get('accommodation_type', ''),
                        'accommodation_cost': row.get('accommodation_cost', ''),
                        'accommodation_cost_numeric': acc_cost_num,
                        'transportation_type': row.get('transportation_type', ''),
                        'transportation_cost': row.get('transportation_cost', ''),
                        'transportation_cost_numeric': trans_cost_num,
                        'total_cost_numeric': total_cost
                    })
                    
                    if len(results) >= max_results:
                        break
    except Exception as e:
        return [{"error": f"Error reading CSV: {str(e)}"}]
    
    return results

@mcp.tool
def search_tripadvisor_attractions(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Search TripAdvisor attractions data.
    Returns attractions matching query in sub_category, description, location, or country.
    """
    results = []
    csv_path = "data/tripadvisor-travel-datasets-web-scraping.csv"
    
    if not os.path.exists(csv_path):
        return [{"error": f"File not found: {csv_path}"}]
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            # Try to detect delimiter and handle BOM
            sample = f.read(1024)
            f.seek(0)
            
            # Check for BOM
            if sample.startswith('\ufeff'):
                f = open(csv_path, 'r', encoding='utf-8-sig')
            
            reader = csv.DictReader(f)
            for row in reader:
                # Clean up field names
                row = {k.strip().replace('\ufeff', ''): v for k, v in row.items()}
                
                # Check if query matches any relevant field
                query_lower = query.lower()
                matches = False
                for field in ['sub_category', 'description', 'location', 'country', 'category']:
                    if field in row and query_lower in row[field].lower():
                        matches = True
                        break
                
                if matches:
                    # Convert rating to numeric if possible
                    rating = row.get('rating', '0')
                    try:
                        rating_num = float(rating)
                    except (ValueError, TypeError):
                        rating_num = 0
                    
                    # Clean number of reviews
                    num_reviews = row.get('number_of_reviews', '0')
                    num_reviews_clean = re.sub(r'[^\d]', '', num_reviews)
                    try:
                        num_reviews_num = int(num_reviews_clean) if num_reviews_clean else 0
                    except (ValueError, TypeError):
                        num_reviews_num = 0
                    
                    results.append({
                        'id': row.get('id', ''),
                        'platform_name': row.get('platform_name', ''),
                        'platform_url': row.get('platform_url', ''),
                        'item_id': row.get('item_id', ''),
                        'category': row.get('category', ''),
                        'sub_category': row.get('sub_category', ''),
                        'description': row.get('description', ''),
                        'location': row.get('location', ''),
                        'major_region': row.get('major_region', ''),
                        'country': row.get('country', ''),
                        'rating': rating,
                        'rating_numeric': rating_num,
                        'number_of_reviews': num_reviews,
                        'number_of_reviews_numeric': num_reviews_num,
                        'entry_fee': row.get('entry_fee', ''),
                        'url': row.get('url', '')
                    })
                    
                    if len(results) >= max_results:
                        break
    except Exception as e:
        return [{"error": f"Error reading CSV: {str(e)}"}]
    
    return results

@mcp.tool
def search_hotel_star_ratings(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Search hotel star ratings information from JSON records.
    Returns rating levels matching query in rating_level or description.
    """
    results = []
    json_path = "records.json"
    
    if not os.path.exists(json_path):
        return [{"error": f"File not found: {json_path}"}]
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Look for the Paris Hotel Star Ratings entry
        for item in data:
            if isinstance(item, dict) and 'title' in item:
                title = item.get('title', '')
                if 'Paris Hotel Star Ratings' in title:
                    # Extract real_data_samples if available
                    real_samples = item.get('real_data_samples', [])
                    query_lower = query.lower()
                    
                    for sample in real_samples:
                        if isinstance(sample, dict):
                            matches = False
                            for field in ['rating_level', 'description']:
                                if field in sample and query_lower in str(sample[field]).lower():
                                    matches = True
                                    break
                            
                            if matches:
                                results.append({
                                    'rating_level': sample.get('rating_level', ''),
                                    'description': sample.get('description', ''),
                                    'source': 'Paris Hotel Star Ratings Guide'
                                })
                                
                                if len(results) >= max_results:
                                    break
                    
                    # If no matches in real_samples, check data_schema samples
                    if not results and 'data_schema' in item:
                        data_schema = item['data_schema']
                        if 'samples' in data_schema:
                            samples = data_schema['samples']
                            # Samples appear to be interleaved: rating, description, rating, description...
                            for i in range(0, len(samples) - 1, 2):
                                if i + 1 < len(samples):
                                    rating = str(samples[i])
                                    desc = str(samples[i + 1])
                                    if query_lower in rating.lower() or query_lower in desc.lower():
                                        results.append({
                                            'rating_level': rating,
                                            'description': desc,
                                            'source': 'Paris Hotel Star Ratings Guide (schema samples)'
                                        })
                                        
                                        if len(results) >= max_results:
                                            break
                    
                    break  # Stop after finding the Paris Hotel Star Ratings entry
    except Exception as e:
        return [{"error": f"Error reading JSON: {str(e)}"}]
    
    return results

@mcp.tool
def submit_result(result) -> Any:
    """
    Persist the result payload to submitted_result.json in current working directory.
    Returns the result that was submitted.
    """
    output_path = "submitted_result.json"
    
    try:
        # Convert result to JSON-serializable format if needed
        if hasattr(result, '__dict__'):
            data = result.__dict__
        elif isinstance(result, (dict, list, str, int, float, bool, type(None))):
            data = result
        else:
            data = str(result)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return result
    except Exception as e:
        # If serialization fails, save error message
        error_data = {"error": f"Failed to save result: {str(e)}", "original_result": str(result)}
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(error_data, f, indent=2, ensure_ascii=False)
        return result
