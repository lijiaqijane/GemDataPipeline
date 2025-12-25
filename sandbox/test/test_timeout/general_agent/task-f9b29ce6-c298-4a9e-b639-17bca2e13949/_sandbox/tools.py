import json
import csv
import os
import re
from pathlib import Path
from typing import List, Dict, Any
import random

# Set seed for deterministic behavior
random.seed(0)

# Fallback shim if mcp is not available
try:
    import mcp
except ImportError:
    # Create a minimal shim that defines mcp.tool as identity decorator
    class MCPShim:
        @staticmethod
        def tool(func):
            return func
    mcp = MCPShim()

# Helper function to read CSV files
def read_csv_file(filepath: str) -> List[Dict[str, str]]:
    """Read a CSV file and return list of dictionaries."""
    data = []
    if not os.path.exists(filepath):
        return data
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            # Try to detect delimiter
            sample = f.read(1024)
            f.seek(0)
            
            # Check for tab delimiter
            if '\t' in sample and sample.count('\t') > sample.count(','):
                delimiter = '\t'
            else:
                delimiter = ','
            
            # Handle potential BOM
            if sample.startswith('\ufeff'):
                f = open(filepath, 'r', encoding='utf-8-sig')
            
            reader = csv.DictReader(f, delimiter=delimiter)
            for row in reader:
                # Clean up keys (remove BOM, whitespace)
                cleaned_row = {}
                for key, value in row.items():
                    clean_key = key.strip().replace('\ufeff', '')
                    cleaned_row[clean_key] = value.strip() if value else ''
                data.append(cleaned_row)
    except Exception as e:
        print(f"Error reading CSV {filepath}: {e}")
    
    return data

# Helper function to read JSON files
def read_json_file(filepath: str) -> Any:
    """Read a JSON file and return parsed data."""
    if not os.path.exists(filepath):
        return []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading JSON {filepath}: {e}")
        return []

# Helper function for text search
def search_in_text(text: str, query: str) -> bool:
    """Check if query matches text (case-insensitive)."""
    if not text or not query:
        return False
    return query.lower() in text.lower()

@mcp.tool
def search_hotels(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Search hotels from available hotel datasets.
    
    Args:
        query: Search term (hotel name, city, country, amenities, etc.)
        max_results: Maximum number of results to return
        
    Returns:
        List of hotel dictionaries matching the query
    """
    results = []
    
    # Read hotels CSV
    hotels_csv_path = "data/hotels-csv-bixbydevelopers-sample-templa.csv"
    hotels = read_csv_file(hotels_csv_path)
    
    # Also read from the travel-recommendation CSV for accommodation info
    travel_csv_path = "data/travel-recommendation-system-travel-deta.csv"
    travel_data = read_csv_file(travel_csv_path)
    
    # Search in hotels dataset
    for hotel in hotels:
        if not hotel:
            continue
            
        # Create searchable text from hotel fields
        search_text = " ".join([
            hotel.get('name', ''),
            hotel.get('city', ''),
            hotel.get('country', ''),
            hotel.get('amenities', ''),
            hotel.get('stars', '')
        ])
        
        if search_in_text(search_text, query):
            # Convert price to numeric if possible
            price_str = hotel.get('price', '')
            try:
                price = float(price_str.replace('$', '').replace(',', '').strip())
            except:
                price = price_str
            
            results.append({
                'name': hotel.get('name', ''),
                'city': hotel.get('city', ''),
                'country': hotel.get('country', ''),
                'stars': hotel.get('stars', ''),
                'amenities': hotel.get('amenities', ''),
                'price': price,
                'price_display': price_str,
                'photo_url': hotel.get('photo', ''),
                'website': hotel.get('website', ''),
                'source': 'hotels_dataset'
            })
    
    # Search in travel recommendation data for hotel accommodations
    for trip in travel_data:
        if not trip:
            continue
            
        accommodation = trip.get('accommodation_type', '')
        destination = trip.get('destination', '')
        
        if accommodation.lower() in ['hotel', 'resort'] and search_in_text(destination, query):
            # Try to extract cost
            cost_str = trip.get('accommodation_cost', '')
            try:
                cost = float(cost_str.replace('$', '').replace(',', '').strip())
            except:
                cost = cost_str
            
            results.append({
                'name': f"Accommodation in {destination}",
                'city': destination.split(',')[0] if ',' in destination else destination,
                'country': destination.split(',')[-1].strip() if ',' in destination else '',
                'stars': 'N/A',
                'amenities': 'Not specified',
                'price': cost,
                'price_display': cost_str,
                'photo_url': '',
                'website': '',
                'traveler_name': trip.get('traveler_name', ''),
                'duration_days': trip.get('duration_days', ''),
                'source': 'travel_recommendation'
            })
    
    # Sort by price if possible
    def get_price(item):
        price = item.get('price', 0)
        return float(price) if isinstance(price, (int, float)) else float('inf')
    
    results.sort(key=get_price)
    
    # Limit results
    return results[:max_results]

@mcp.tool
def search_attractions(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Search attractions and experiences from TripAdvisor and Airbnb datasets.
    
    Args:
        query: Search term (attraction name, location, category, etc.)
        max_results: Maximum number of results to return
        
    Returns:
        List of attraction dictionaries matching the query
    """
    results = []
    
    # Read TripAdvisor CSV
    tripadvisor_csv_path = "data/tripadvisor-travel-datasets-web-scraping.csv"
    tripadvisor_data = read_csv_file(tripadvisor_csv_path)
    
    # Read Airbnb CSV
    airbnb_csv_path = "data/airbnb-travel-datasets-web-scraping-hote.csv"
    airbnb_data = read_csv_file(airbnb_csv_path)
    
    # Search in TripAdvisor data
    for attraction in tripadvisor_data:
        if not attraction:
            continue
            
        search_text = " ".join([
            attraction.get('sub_category', ''),
            attraction.get('description', ''),
            attraction.get('location', ''),
            attraction.get('country', ''),
            attraction.get('category', '')
        ])
        
        if search_in_text(search_text, query):
            # Parse rating
            rating_str = attraction.get('rating', '')
            try:
                rating = float(rating_str)
            except:
                rating = rating_str
            
            # Parse review count
            reviews_str = attraction.get('number_of_reviews', '').replace(',', '')
            try:
                reviews = int(reviews_str)
            except:
                reviews = reviews_str
            
            results.append({
                'name': attraction.get('sub_category', ''),
                'description': attraction.get('description', ''),
                'location': attraction.get('location', ''),
                'country': attraction.get('country', ''),
                'category': attraction.get('category', ''),
                'rating': rating,
                'review_count': reviews,
                'entry_fee': attraction.get('entry_fee', ''),
                'url': attraction.get('url', ''),
                'platform': 'TripAdvisor',
                'source': 'tripadvisor_dataset'
            })
    
    # Search in Airbnb experiences data
    for experience in airbnb_data:
        if not experience:
            continue
            
        search_text = " ".join([
            experience.get('experience_title', ''),
            experience.get('location', ''),
            experience.get('country', ''),
            experience.get('category', ''),
            experience.get('highlights', '')
        ])
        
        if search_in_text(search_text, query):
            # Parse rating
            rating_str = experience.get('rating', '')
            try:
                rating = float(rating_str)
            except:
                rating = rating_str
            
            # Parse review count
            reviews_str = experience.get('review_count', '').replace(',', '')
            try:
                reviews = int(reviews_str)
            except:
                reviews = reviews_str
            
            # Parse price
            price_str = experience.get('price', '')
            try:
                price = float(re.sub(r'[^\d.]', '', price_str))
            except:
                price = price_str
            
            results.append({
                'name': experience.get('experience_title', ''),
                'description': experience.get('highlights', ''),
                'location': experience.get('location', ''),
                'country': experience.get('country', ''),
                'category': experience.get('category', ''),
                'rating': rating,
                'review_count': reviews,
                'price': price,
                'price_display': price_str,
                'duration': experience.get('duration', ''),
                'host_name': experience.get('host_name', ''),
                'url': experience.get('product_url', ''),
                'platform': 'Airbnb',
                'source': 'airbnb_dataset'
            })
    
    # Sort by rating (highest first)
    def get_rating(item):
        rating = item.get('rating', 0)
        return float(rating) if isinstance(rating, (int, float)) else 0
    
    results.sort(key=get_rating, reverse=True)
    
    return results[:max_results]

@mcp.tool
def search_travel_recommendations(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Search travel recommendations and trip data.
    
    Args:
        query: Search term (destination, traveler name, nationality, etc.)
        max_results: Maximum number of results to return
        
    Returns:
        List of travel recommendation dictionaries matching the query
    """
    results = []
    
    # Read travel recommendations CSV
    travel_csv_path = "data/travel-recommendation-system-travel-deta.csv"
    travel_data = read_csv_file(travel_csv_path)
    
    # Read records.json for additional context
    records_json_path = "records.json"
    records_data = read_json_file(records_json_path)
    
    # Search in travel data
    for trip in travel_data:
        if not trip:
            continue
            
        search_text = " ".join([
            trip.get('destination', ''),
            trip.get('traveler_name', ''),
            trip.get('traveler_nationality', ''),
            trip.get('accommodation_type', ''),
            trip.get('transportation_type', '')
        ])
        
        if search_in_text(search_text, query):
            # Parse costs
            acc_cost_str = trip.get('accommodation_cost', '')
            trans_cost_str = trip.get('transportation_cost', '')
            
            try:
                acc_cost = float(acc_cost_str.replace('$', '').replace(',', '').strip())
            except:
                acc_cost = acc_cost_str
            
            try:
                trans_cost = float(trans_cost_str.replace('$', '').replace(',', '').strip())
            except:
                trans_cost = trans_cost_str
            
            # Calculate total cost if both are numeric
            total_cost = None
            if isinstance(acc_cost, (int, float)) and isinstance(trans_cost, (int, float)):
                total_cost = acc_cost + trans_cost
            
            results.append({
                'trip_id': trip.get('trip_id', ''),
                'destination': trip.get('destination', ''),
                'start_date': trip.get('start_date', ''),
                'end_date': trip.get('end_date', ''),
                'duration_days': trip.get('duration_days', ''),
                'traveler_name': trip.get('traveler_name', ''),
                'traveler_age': trip.get('traveler_age', ''),
                'traveler_gender': trip.get('traveler_gender', ''),
                'traveler_nationality': trip.get('traveler_nationality', ''),
                'accommodation_type': trip.get('accommodation_type', ''),
                'accommodation_cost': acc_cost,
                'accommodation_cost_display': acc_cost_str,
                'transportation_type': trip.get('transportation_type', ''),
                'transportation_cost': trans_cost,
                'transportation_cost_display': trans_cost_str,
                'total_cost': total_cost,
                'source': 'travel_recommendation_dataset'
            })
    
    # Search in records.json for Paris hotel star ratings
    if isinstance(records_data, list):
        for record in records_data:
            if not isinstance(record, dict):
                continue
                
            title = record.get('title', '')
            summary = record.get('summary', '')
            
            if 'Paris Hotel Star Ratings' in title and search_in_text(summary, query):
                # Extract star rating data
                real_samples = record.get('real_data_samples', [])
                for sample in real_samples:
                    if isinstance(sample, dict):
                        results.append({
                            'star_rating': sample.get('star_rating', ''),
                            'description': sample.get('description', ''),
                            'typical_amenities': sample.get('typical_amenities', ''),
                            'source': 'paris_hotel_ratings_article',
                            'article_title': title,
                            'article_url': record.get('url', '')
                        })
    
    # Sort by duration (longest first)
    def get_duration(item):
        duration_str = item.get('duration_days', '0')
        try:
            return int(duration_str)
        except:
            return 0
    
    results.sort(key=get_duration, reverse=True)
    
    return results[:max_results]

@mcp.tool
def get_hotel_star_ratings(query: str = "", max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Get hotel star rating explanations, particularly for Paris hotels.
    
    Args:
        query: Optional filter (e.g., "paris", "5 star")
        max_results: Maximum number of results to return
        
    Returns:
        List of hotel star rating explanations
    """
    results = []
    
    # Read Paris hotel star ratings CSV
    paris_csv_path = "data/paris-hotel-star-ratings-explained-paris.csv"
    paris_data = read_csv_file(paris_csv_path)
    
    # Read records.json for additional context
    records_json_path = "records.json"
    records_data = read_json_file(records_json_path)
    
    # Get data from CSV
    for rating in paris_data:
        if not rating:
            continue
            
        search_text = " ".join([
            rating.get('star_rating', ''),
            rating.get('description', ''),
            rating.get('typical_amenities', '')
        ])
        
        if not query or search_in_text(search_text, query):
            results.append({
                'star_rating': rating.get('star_rating', ''),
                'description': rating.get('description', ''),
                'typical_amenities': rating.get('typical_amenities', ''),
                'source': 'paris_hotel_ratings_csv'
            })
    
    # Get data from records.json
    if isinstance(records_data, list):
        for record in records_data:
            if not isinstance(record, dict):
                continue
                
            title = record.get('title', '')
            
            if 'Paris Hotel Star Ratings' in title:
                real_samples = record.get('real_data_samples', [])
                for sample in real_samples:
                    if isinstance(sample, dict):
                        search_text = " ".join([
                            str(sample.get('star_rating', '')),
                            str(sample.get('description', '')),
                            str(sample.get('typical_amenities', ''))
                        ])
                        
                        if not query or search_in_text(search_text, query):
                            results.append({
                                'star_rating': sample.get('star_rating', ''),
                                'description': sample.get('description', ''),
                                'typical_amenities': sample.get('typical_amenities', ''),
                                'source': 'paris_hotel_ratings_article',
                                'article_title': title,
                                'article_url': record.get('url', '')
                            })
    
    # Remove duplicates based on star_rating
    seen = set()
    unique_results = []
    for item in results:
        rating = item.get('star_rating', '')
        if rating not in seen:
            seen.add(rating)
            unique_results.append(item)
    
    return unique_results[:max_results]

@mcp.tool
def submit_result(result) -> str:
    """
    Submit a result for persistence. Saves the result to submitted_result.json.
    
    Args:
        result: The result data to persist (any JSON-serializable type)
        
    Returns:
        The submitted result
    """
    output_file = "submitted_result.json"
    
    try:
        # Ensure the result is JSON serializable
        if hasattr(result, 'to_dict'):
            result_data = result.to_dict()
        elif hasattr(result, '__dict__'):
            result_data = result.__dict__
        else:
            result_data = result
        
        # Write to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        
        print(f"Result submitted successfully to {output_file}")
    except Exception as e:
        print(f"Error submitting result: {e}")
        # Try to save at least something
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(str(result))
        except:
            pass
    
    return result
