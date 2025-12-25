import json
import sqlite3
import csv
import random
from pathlib import Path
from typing import List, Dict, Any
import mcp

# Set seed for deterministic behavior
random.seed(0)

# Base directory for absolute paths
BASE_DIR = Path(__file__).parent

# Data source paths
CSV_PATHS = {
    "go_memory_leak": BASE_DIR / "data" / "deep-dive-into-go-memory-leak-debugging-.csv",
    "debugging_strategies": BASE_DIR / "data" / "effective-strategies-for-debugging-compl.csv",
    "backend_guide": BASE_DIR / "data" / "debugging-complex-backend-code-a-step-by.csv",
    "codebases_guide": BASE_DIR / "data" / "debugging-complex-codebases-a-comprehens.csv",
    "code_flows": BASE_DIR / "data" / "how-to-remember-the-code-flows-in-a-comp.csv"
}

JSON_PATHS = {
    "records": BASE_DIR / "records.json",
    "search_cache": BASE_DIR / "search_cache.json"
}

SQLITE_PATH = BASE_DIR / "data" / "datasets_metadata.db"

@mcp.tool
def search_debugging_techniques(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Search CSV files for debugging techniques, symptoms, and fixes."""
    results = []
    
    # Search go memory leak CSV
    if CSV_PATHS["go_memory_leak"].exists():
        try:
            with open(CSV_PATHS["go_memory_leak"], 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Check if query matches any field
                    matches = any(
                        query.lower() in str(value).lower() 
                        for value in row.values()
                    )
                    if matches:
                        results.append({
                            "source": "go_memory_leak_debugging",
                            "symptom": row.get("symptom", ""),
                            "possible_cause": row.get("possible_cause", ""),
                            "quick_fix": row.get("quick_fix", ""),
                            "data_type": "csv"
                        })
        except Exception as e:
            pass
    
    # Search debugging strategies CSV
    if CSV_PATHS["debugging_strategies"].exists():
        try:
            with open(CSV_PATHS["debugging_strategies"], 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    matches = any(
                        query.lower() in str(value).lower() 
                        for value in row.values()
                    )
                    if matches:
                        results.append({
                            "source": "debugging_strategies",
                            "title": row.get("title", ""),
                            "published_time": row.get("published_time", ""),
                            "tracking_image_count": row.get("tracking_image_count", ""),
                            "sample_tracking_url": row.get("sample_tracking_url", ""),
                            "parameters_present": row.get("parameters_present", ""),
                            "data_type": "csv"
                        })
        except Exception as e:
            pass
    
    # Limit results
    return results[:max_results]

@mcp.tool
def query_articles_metadata(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Query the SQLite database for article metadata and search JSON records."""
    results = []
    
    # Query SQLite database
    if SQLITE_PATH.exists():
        try:
            conn = sqlite3.connect(str(SQLITE_PATH))
            cursor = conn.cursor()
            
            # Search datasets table
            cursor.execute("""
                SELECT id, name, source_url, has_real_data, record_count, created_at 
                FROM datasets 
                WHERE name LIKE ? OR source_url LIKE ?
                LIMIT ?
            """, (f'%{query}%', f'%{query}%', max_results))
            
            for row in cursor.fetchall():
                results.append({
                    "id": row[0],
                    "title": row[1],
                    "url": row[2],
                    "has_real_data": bool(row[3]),
                    "record_count": row[4],
                    "created_at": row[5],
                    "data_type": "sqlite"
                })
            
            conn.close()
        except Exception as e:
            pass
    
    # Search records.json
    if JSON_PATHS["records"].exists():
        try:
            with open(JSON_PATHS["records"], 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        # Check if query matches title, summary, or URL
                        title = item.get("title", "")
                        summary = item.get("summary", "")
                        url = item.get("url", "")
                        
                        if (query.lower() in title.lower() or 
                            query.lower() in summary.lower() or 
                            query.lower() in url.lower()):
                            
                            result_item = {
                                "title": title,
                                "summary": summary,
                                "url": url,
                                "source": item.get("source", ""),
                                "data_type": "json_records"
                            }
                            
                            # Add real_data_samples if available
                            real_samples = item.get("real_data_samples")
                            if real_samples and isinstance(real_samples, list) and len(real_samples) > 0:
                                result_item["sample_data"] = real_samples[0]
                            
                            results.append(result_item)
        except Exception as e:
            pass
    
    # Search search_cache.json
    if JSON_PATHS["search_cache"].exists():
        try:
            with open(JSON_PATHS["search_cache"], 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        title = item.get("title", "")
                        summary = item.get("summary", "")
                        url = item.get("url", "")
                        
                        if (query.lower() in title.lower() or 
                            query.lower() in summary.lower() or 
                            query.lower() in url.lower()):
                            
                            results.append({
                                "title": title,
                                "summary": summary,
                                "url": url,
                                "data_type": "json_search_cache"
                            })
        except Exception as e:
            pass
    
    return results[:max_results]

@mcp.tool
def get_error_patterns(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Extract error patterns and security messages from CSV files."""
    results = []
    
    # Search backend guide CSV
    if CSV_PATHS["backend_guide"].exists():
        try:
            with open(CSV_PATHS["backend_guide"], 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    error_code = row.get("error_code", "")
                    security_msg = row.get("security_message", "")
                    
                    matches = (query.lower() in error_code.lower() or 
                              query.lower() in security_msg.lower() or
                              query.lower() in row.get("platform", "").lower())
                    
                    if matches:
                        results.append({
                            "source": "backend_debugging_guide",
                            "error_code": error_code,
                            "security_message": security_msg,
                            "platform": row.get("platform", ""),
                            "favicon_url": row.get("favicon_url", ""),
                            "data_type": "csv"
                        })
        except Exception as e:
            pass
    
    # Search codebases guide CSV
    if CSV_PATHS["codebases_guide"].exists():
        try:
            with open(CSV_PATHS["codebases_guide"], 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    error_code = row.get("error_code", "")
                    security_msg = row.get("security_message", "")
                    
                    matches = (query.lower() in error_code.lower() or 
                              query.lower() in security_msg.lower() or
                              query.lower() in row.get("platform", "").lower())
                    
                    if matches:
                        results.append({
                            "source": "codebases_debugging_guide",
                            "error_code": error_code,
                            "security_message": security_msg,
                            "platform": row.get("platform", ""),
                            "favicon_url": row.get("favicon_url", ""),
                            "data_type": "csv"
                        })
        except Exception as e:
            pass
    
    # Search code flows CSV
    if CSV_PATHS["code_flows"].exists():
        try:
            with open(CSV_PATHS["code_flows"], 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    error_type = row.get("error_type", "")
                    required_action = row.get("required_action", "")
                    
                    matches = (query.lower() in error_type.lower() or 
                              query.lower() in required_action.lower() or
                              query.lower() in row.get("login_url", "").lower())
                    
                    if matches:
                        results.append({
                            "source": "code_flows_reddit",
                            "error_type": error_type,
                            "required_action": required_action,
                            "login_url": row.get("login_url", ""),
                            "support_ticket_url": row.get("support_ticket_url", ""),
                            "data_type": "csv"
                        })
        except Exception as e:
            pass
    
    return results[:max_results]

@mcp.tool
def submit_result(result: Any) -> Any:
    """Submit and persist a result to submitted_result.json."""
    output_path = BASE_DIR / "submitted_result.json"
    
    try:
        # Convert result to JSON-serializable format if needed
        if hasattr(result, 'dict'):
            data = result.dict()
        elif isinstance(result, (dict, list, str, int, float, bool, type(None))):
            data = result
        else:
            data = str(result)
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return {"status": "success", "message": f"Result saved to {output_path}", "data": data}
    except Exception as e:
        return {"status": "error", "message": str(e), "data": result}
