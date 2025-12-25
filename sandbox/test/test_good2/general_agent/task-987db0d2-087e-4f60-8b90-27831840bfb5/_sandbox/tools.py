import json
import csv
import os
import re
import random
from typing import List, Dict, Any
from datetime import datetime

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
    Search hotel pricing data from London/Paris seasonal pricing CSV.
    Returns hotels matching query in name, room type, or special offer.
    """
    results = []
    csv_path = "data/scrape-london-paris-hotel-seasonal-prici.csv"
    
    if not os.path.exists(csv_path):
        return [{"error": f"File not found: {csv_path}"}]
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Normalize query for case-insensitive search
                query_lower = query.lower()
                hotel_name = row.get('hotel_name', '').lower()
                room_type = row.get('room_type', '').lower()
                special_offer = row.get('special_offer', '').lower()
                
                # Check if query matches any field
                if (query_lower in hotel_name or 
                    query_lower in room_type or 
                    query_lower in special_offer or
                    query == ""):  # Empty query returns all
                    
                    # Convert rate to numeric if possible
                    rate = row.get('rate', '0')
                    try:
                        rate_num = float(rate)
                    except (ValueError, TypeError):
                        rate_num = 0
                    
                    # Convert rating to numeric if possible
                    rating = row.get('guest_rating', '0')
                    try:
                        rating_num = float(rating)
                    except (ValueError, TypeError):
                        rating_num = 0
                    
                    results.append({
                        'hotel_name': row.get('hotel_name', ''),
                        'room_type': row.get('room_type', ''),
                        'date': row.get('date', ''),
                        'rate': rate_num,
                        'availability': row.get('availability', ''),
                        'guest_rating': rating_num,
                        'special_offer': row.get('special_offer', '')
                    })
                    
                    if len(results) >= max_results:
                        break
    except Exception as e:
        return [{"error": f"Error reading CSV: {str(e)}"}]
    
    return results

@mcp.tool
def search_travel_recommendations(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Search travel recommendation dataset for trips matching destination,
    traveler demographics, or accommodation type.
    """
    results = []
    csv_path = "data/travel-recommendation-system-travel-deta.csv"
    
    if not os.path.exists(csv_path):
        return [{"error": f"File not found: {csv_path}"}]
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Normalize query for case-insensitive search
                query_lower = query.lower()
                destination = row.get('destination', '').lower()
                traveler_name = row.get('traveler_name', '').lower()
                accommodation_type = row.get('accommodation_type', '').lower()
                traveler_nationality = row.get('traveler_nationality', '').lower()
                
                # Check if query matches any relevant field
                if (query_lower in destination or 
                    query_lower in traveler_name or 
                    query_lower in accommodation_type or
                    query_lower in traveler_nationality or
                    query == ""):  # Empty query returns all
                    
                    # Parse numeric fields
                    try:
                        duration_days = int(row.get('duration_days', '0'))
                    except (ValueError, TypeError):
                        duration_days = 0
                    
                    try:
                        traveler_age = int(row.get('traveler_age', '0'))
                    except (ValueError, TypeError):
                        traveler_age = 0
                    
                    # Clean cost fields (remove $, commas, etc.)
                    acc_cost_raw = row.get('accommodation_cost', '0')
                    trans_cost_raw = row.get('transportation_cost', '0')
                    
                    def clean_cost(cost_str):
                        if not cost_str:
                            return 0
                        # Remove currency symbols and commas
                        cleaned = re.sub(r'[^\d.]', '', str(cost_str))
                        try:
                            return float(cleaned) if cleaned else 0
                        except ValueError:
                            return 0
                    
                    acc_cost = clean_cost(acc_cost_raw)
                    trans_cost = clean_cost(trans_cost_raw)
                    total_cost = acc_cost + trans_cost
                    
                    results.append({
                        'trip_id': row.get('trip_id', ''),
                        'destination': row.get('destination', ''),
                        'start_date': row.get('start_date', ''),
                        'end_date': row.get('end_date', ''),
                        'duration_days': duration_days,
                        'traveler_name': row.get('traveler_name', ''),
                        'traveler_age': traveler_age,
                        'traveler_gender': row.get('traveler_gender', ''),
                        'traveler_nationality': row.get('traveler_nationality', ''),
                        'accommodation_type': row.get('accommodation_type', ''),
                        'accommodation_cost': acc_cost,
                        'transportation_type': row.get('transportation_type', ''),
                        'transportation_cost': trans_cost,
                        'total_cost': total_cost
                    })
                    
                    if len(results) >= max_results:
                        break
    except Exception as e:
        return [{"error": f"Error reading CSV: {str(e)}"}]
    
    return results

@mcp.tool
def search_tripadvisor_attractions(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Search TripAdvisor attractions dataset for matching locations,
    categories, or descriptions.
    """
    results = []
    csv_path = "data/tripadvisor-travel-datasets-web-scraping.csv"
    
    if not os.path.exists(csv_path):
        return [{"error": f"File not found: {csv_path}"}]
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            # Handle potential BOM
            sample = f.read(1024)
            f.seek(0)
            has_bom = sample.startswith('\ufeff')
            
            reader = csv.DictReader(f)
            for row in reader:
                # Clean field names (remove BOM if present)
                clean_row = {}
                for key, value in row.items():
                    clean_key = key.lstrip('\ufeff') if has_bom else key
                    clean_row[clean_key] = value
                
                # Normalize query for case-insensitive search
                query_lower = query.lower()
                location = clean_row.get('location', '').lower()
                category = clean_row.get('category', '').lower()
                sub_category = clean_row.get('sub_category', '').lower()
                description = clean_row.get('description', '').lower()
                country = clean_row.get('country', '').lower()
                
                # Check if query matches any field
                if (query_lower in location or 
                    query_lower in category or 
                    query_lower in sub_category or
                    query_lower in description or
                    query_lower in country or
                    query == ""):  # Empty query returns all
                    
                    # Parse numeric fields
                    rating = clean_row.get('rating', '0')
                    try:
                        rating_num = float(rating)
                    except (ValueError, TypeError):
                        rating_num = 0
                    
                    # Clean review count (remove commas)
                    reviews_raw = clean_row.get('number_of_reviews', '0')
                    reviews_clean = re.sub(r'[^\d]', '', str(reviews_raw))
                    try:
                        reviews_num = int(reviews_clean) if reviews_clean else 0
                    except ValueError:
                        reviews_num = 0
                    
                    results.append({
                        'id': clean_row.get('id', ''),
                        'platform_name': clean_row.get('platform_name', ''),
                        'item_id': clean_row.get('item_id', ''),
                        'category': clean_row.get('category', ''),
                        'sub_category': clean_row.get('sub_category', ''),
                        'description': clean_row.get('description', ''),
                        'location': clean_row.get('location', ''),
                        'major_region': clean_row.get('major_region', ''),
                        'country': clean_row.get('country', ''),
                        'rating': rating_num,
                        'number_of_reviews': reviews_num,
                        'entry_fee': clean_row.get('entry_fee', ''),
                        'url': clean_row.get('url', '')
                    })
                    
                    if len(results) >= max_results:
                        break
    except Exception as e:
        return [{"error": f"Error reading CSV: {str(e)}"}]
    
    return results

@mcp.tool
def search_paris_hotel_guides(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Search Paris hotel guide content from JSON records.
    Returns articles matching query in title, summary, or content.
    """
    results = []
    json_path = "records.json"
    
    if not os.path.exists(json_path):
        return [{"error": f"File not found: {json_path}"}]
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            if not isinstance(data, list):
                return [{"error": "JSON data is not a list"}]
            
            query_lower = query.lower()
            
            for item in data:
                if not isinstance(item, dict):
                    continue
                
                title = item.get('title', '').lower()
                summary = item.get('summary', '').lower()
                source = item.get('source', '').lower()
                
                # Check real_content_samples if available
                real_samples = item.get('real_data_samples', [])
                has_match_in_samples = False
                if real_samples and query:
                    for sample in real_samples:
                        if isinstance(sample, dict):
                            sample_str = ' '.join(str(v).lower() for v in sample.values() if v)
                            if query_lower in sample_str:
                                has_match_in_samples = True
                                break
                
                # Check if query matches
                if (query_lower in title or 
                    query_lower in summary or 
                    query_lower in source or
                    has_match_in_samples or
                    query == ""):  # Empty query returns all
                    
                    # Extract star rating info if available
                    star_ratings = []
                    if 'real_data_samples' in item and isinstance(item['real_data_samples'], list):
                        for sample in item['real_data_samples']:
                            if isinstance(sample, dict) and 'rating_level' in sample:
                                star_ratings.append({
                                    'rating_level': sample.get('rating_level'),
                                    'description': sample.get('description', ''),
                                    'typical_amenities': sample.get('typical_amenities', ''),
                                    'room_size': sample.get('room_size', '')
                                })
                    
                    results.append({
                        'title': item.get('title', ''),
                        'summary': item.get('summary', ''),
                        'url': item.get('url', ''),
                        'source': item.get('source', ''),
                        'star_ratings': star_ratings[:3],  # Limit to 3 ratings
                        'has_real_data': len(star_ratings) > 0
                    })
                    
                    if len(results) >= max_results:
                        break
    except Exception as e:
        return [{"error": f"Error reading JSON: {str(e)}"}]
    
    return results

@mcp.tool
def submit_result(result) -> Any:
    """
    Persist the provided result to submitted_result.json for inspection.
    Returns the result unchanged.
    """
    output_path = "submitted_result.json"
    
    try:
        # Convert result to JSON-serializable format if needed
        def make_serializable(obj):
            if isinstance(obj, (str, int, float, bool, type(None))):
                return obj
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(item) for item in obj]
            elif isinstance(obj, tuple):
                return [make_serializable(item) for item in obj]
            elif hasattr(obj, '__dict__'):
                return make_serializable(obj.__dict__)
            else:
                return str(obj)
        
        serializable_result = make_serializable(result)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'result': serializable_result
            }, f, indent=2, ensure_ascii=False)
        
        # Also print to stdout for immediate feedback
        print(f"Result submitted to {output_path}")
        
    except Exception as e:
        print(f"Error saving result: {str(e)}")
        # Still return the result even if saving fails
        pass
    
    return result
