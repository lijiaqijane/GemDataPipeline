import json
import sqlite3
import csv
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
import re

# Fallback shim if mcp is not available
try:
    import mcp
except ImportError:
    # Create a minimal identity decorator as fallback
    class MockMCP:
        @staticmethod
        def tool(func):
            return func
    mcp = MockMCP()

# Set seed for deterministic behavior
random.seed(0)

# Get absolute base directory
BASE_DIR = Path(__file__).parent

@mcp.tool
def search_debugging_articles(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Search through debugging articles in records.json for relevant content."""
    results = []
    records_path = BASE_DIR / "records.json"
    
    if not records_path.exists():
        return [{"error": f"records.json not found at {records_path}"}]
    
    try:
        with open(records_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            return [{"error": "records.json does not contain a list"}]
        
        query_lower = query.lower()
        
        for item in data:
            score = 0
            matched_fields = []
            
            # Check title
            if 'title' in item and item['title']:
                title_lower = item['title'].lower()
                if query_lower in title_lower:
                    score += 3
                    matched_fields.append('title')
            
            # Check summary
            if 'summary' in item and item['summary']:
                summary_lower = item['summary'].lower()
                if query_lower in summary_lower:
                    score += 2
                    matched_fields.append('summary')
            
            # Check clean_content
            if 'clean_content' in item and item['clean_content']:
                content_lower = item['clean_content'].lower()
                if query_lower in content_lower:
                    score += 1
                    matched_fields.append('clean_content')
            
            # Check real_data_samples for approach steps
            if 'real_data_samples' in item and item['real_data_samples']:
                for sample in item['real_data_samples']:
                    if isinstance(sample, dict):
                        for key, value in sample.items():
                            if isinstance(value, str) and query_lower in value.lower():
                                score += 1
                                matched_fields.append(f'real_data_samples.{key}')
            
            if score > 0:
                result = {
                    'title': item.get('title', ''),
                    'summary': item.get('summary', ''),
                    'url': item.get('url', ''),
                    'source': item.get('source', ''),
                    'score': score,
                    'matched_fields': list(set(matched_fields))
                }
                
                # Add real_data_samples if available
                if 'real_data_samples' in item and item['real_data_samples']:
                    result['real_data_samples'] = item['real_data_samples'][:2]  # Limit samples
                
                results.append(result)
        
        # Sort by score descending
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # Limit results
        return results[:max_results]
        
    except Exception as e:
        return [{"error": f"Error reading records.json: {str(e)}"}]

@mcp.tool
def query_debugging_datasets(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Query the SQLite database for debugging datasets metadata."""
    results = []
    db_path = BASE_DIR / "data" / "datasets_metadata.db"
    
    if not db_path.exists():
        return [{"error": f"Database not found at {db_path}"}]
    
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Search in datasets table
        query_lower = query.lower()
        
        cursor.execute("SELECT * FROM datasets")
        rows = cursor.fetchall()
        
        for row in rows:
            row_dict = dict(row)
            score = 0
            matched_fields = []
            
            # Check name field
            if 'name' in row_dict and row_dict['name']:
                name_lower = row_dict['name'].lower()
                if query_lower in name_lower:
                    score += 2
                    matched_fields.append('name')
            
            # Check source_url field
            if 'source_url' in row_dict and row_dict['source_url']:
                url_lower = row_dict['source_url'].lower()
                if query_lower in url_lower:
                    score += 1
                    matched_fields.append('source_url')
            
            if score > 0:
                result = {
                    'id': row_dict.get('id'),
                    'name': row_dict.get('name', ''),
                    'source_url': row_dict.get('source_url', ''),
                    'has_real_data': bool(row_dict.get('has_real_data', 0)),
                    'record_count': row_dict.get('record_count', 0),
                    'created_at': row_dict.get('created_at', ''),
                    'score': score,
                    'matched_fields': matched_fields
                }
                results.append(result)
        
        conn.close()
        
        # Sort by score descending
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # Limit results
        return results[:max_results]
        
    except Exception as e:
        return [{"error": f"Error querying database: {str(e)}"}]

@mcp.tool
def search_csv_debugging_data(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Search through CSV files for debugging-related data."""
    results = []
    query_lower = query.lower()
    
    # Define CSV files to search
    csv_files = [
        BASE_DIR / "data" / "debugging-complex-backend-code-a-step-by.csv",
        BASE_DIR / "data" / "debugging-complex-issues-in-your-code-ca.csv",
        BASE_DIR / "data" / "how-to-approach-debugging-a-huge-not-so-.csv",
        BASE_DIR / "data" / "debugging-complex-codebases-a-comprehens.csv",
        BASE_DIR / "data" / "what-approaches-work-best-for-debugging-.csv"
    ]
    
    for csv_file in csv_files:
        csv_path = BASE_DIR / csv_file
        
        if not csv_path.exists():
            continue
        
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for i, row in enumerate(reader):
                    if i >= max_results * 2:  # Read slightly more than needed
                        break
                    
                    score = 0
                    matched_fields = []
                    
                    for key, value in row.items():
                        if value and query_lower in value.lower():
                            score += 1
                            matched_fields.append(key)
                    
                    if score > 0:
                        result = {
                            'source_file': csv_file,
                            'row_number': i + 1,
                            'data': row,
                            'score': score,
                            'matched_fields': matched_fields
                        }
                        results.append(result)
                        
        except Exception as e:
            results.append({
                'error': f"Error reading {csv_file}: {str(e)}",
                'source_file': csv_file
            })
    
    # Sort by score descending
    results.sort(key=lambda x: x.get('score', 0), reverse=True)
    
    # Limit results
    return results[:max_results]

@mcp.tool
def get_debugging_approaches(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Extract structured debugging approaches from the data."""
    results = []
    query_lower = query.lower()
    
    # First, try to get approaches from records.json
    records_path = BASE_DIR / "records.json"
    
    if records_path.exists():
        try:
            with open(records_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for item in data:
                # Look for items with real_data_samples containing approach steps
                if 'real_data_samples' in item and item['real_data_samples']:
                    for sample in item['real_data_samples']:
                        if isinstance(sample, dict) and 'approach_step' in sample:
                            step_text = sample.get('approach_step', '')
                            description = sample.get('step_description', '')
                            
                            # Calculate relevance score
                            score = 0
                            if query_lower in step_text.lower():
                                score += 2
                            if query_lower in description.lower():
                                score += 1
                            
                            if score > 0 or query == "":
                                result = {
                                    'approach_step': step_text,
                                    'step_description': description,
                                    'tools_mentioned': sample.get('tools_mentioned', []),
                                    'sub_steps': sample.get('sub_steps', []),
                                    'content_completeness': sample.get('content_completeness', ''),
                                    'source_title': item.get('title', ''),
                                    'score': score
                                }
                                results.append(result)
                                
        except Exception:
            pass
    
    # Also check the specific CSV file for approach steps
    csv_path = BASE_DIR / "data" / "debugging-complex-issues-in-your-code-ca.csv"
    
    if csv_path.exists():
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    step_text = row.get('approach_step', '')
                    description = row.get('step_description', '')
                    
                    if step_text or description:
                        score = 0
                        if query_lower in step_text.lower():
                            score += 2
                        if query_lower in description.lower():
                            score += 1
                        
                        if score > 0 or query == "":
                            result = {
                                'approach_step': step_text,
                                'step_description': description,
                                'tools_mentioned': eval(row.get('tools_mentioned', '[]')) if row.get('tools_mentioned') else [],
                                'sub_steps': eval(row.get('sub_steps', '[]')) if row.get('sub_steps') else [],
                                'content_completeness': row.get('content_completeness', ''),
                                'source_file': BASE_DIR / 'debugging-complex-issues-in-your-code-ca.csv',
                                'score': score
                            }
                            results.append(result)
                            
        except Exception:
            pass
    
    # Sort by score descending
    results.sort(key=lambda x: x['score'], reverse=True)
    
    # Limit results
    return results[:max_results]

@mcp.tool
def submit_result(result) -> Dict[str, Any]:
    """Submit a result and persist it to submitted_result.json."""
    output_path = BASE_DIR / "submitted_result.json"
    
    try:
        # Ensure the result is serializable
        if isinstance(result, (str, int, float, bool, type(None))):
            data_to_save = {"result": result}
        elif isinstance(result, (dict, list)):
            data_to_save = result
        else:
            data_to_save = {"result": str(result)}
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, indent=2, ensure_ascii=False)
        
        return {
            "status": "success",
            "message": f"Result submitted and saved to {output_path}",
            "saved_data": data_to_save
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to submit result: {str(e)}"
        }
