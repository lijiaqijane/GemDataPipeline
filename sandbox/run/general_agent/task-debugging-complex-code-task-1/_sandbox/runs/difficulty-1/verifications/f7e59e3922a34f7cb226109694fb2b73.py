def verify(tools, answer):
    import json
    import re
    
    try:
        # Handle None answer
        if answer is None:
            return {'passed': False, 'message': 'Answer is None'}
        
        # Check if answer is wrapped in submit_result format
        if isinstance(answer, dict):
            # Check for submit_result wrapper keys
            if 'status' in answer and 'data' in answer:
                answer_data = answer.get('data')
            elif 'submitted_data' in answer:
                answer_data = answer.get('submitted_data')
            else:
                answer_data = answer
        else:
            return {'passed': False, 'message': 'Answer is not a dictionary'}
        
        if answer_data is None:
            return {'passed': False, 'message': 'Answer data is None'}
        
        # Verify required keys exist
        required_keys = ['total_approaches_found', 'approach_list', 'most_common_tools', 'data_sources_used']
        for key in required_keys:
            if key not in answer_data:
                return {'passed': False, 'message': f'Missing required key: {key}'}
            
            if answer_data[key] is None:
                return {'passed': False, 'message': f'Key {key} has None value'}
        
        # Verify total_approaches_found is integer
        if not isinstance(answer_data['total_approaches_found'], int):
            return {'passed': False, 'message': 'total_approaches_found must be an integer'}
        
        # Verify approach_list is list
        if not isinstance(answer_data['approach_list'], list):
            return {'passed': False, 'message': 'approach_list must be a list'}
        
        # Verify most_common_tools is list
        if not isinstance(answer_data['most_common_tools'], list):
            return {'passed': False, 'message': 'most_common_tools must be a list'}
        
        # Verify data_sources_used is list
        if not isinstance(answer_data['data_sources_used'], list):
            return {'passed': False, 'message': 'data_sources_used must be a list'}
        
        # Verify approach_list structure
        for i, approach in enumerate(answer_data['approach_list']):
            if not isinstance(approach, dict):
                return {'passed': False, 'message': f'Approach at index {i} is not a dictionary'}
            
            approach_keys = ['approach_name', 'source', 'key_steps', 'tools_mentioned', 'methodology']
            for key in approach_keys:
                if key not in approach:
                    return {'passed': False, 'message': f'Approach at index {i} missing key: {key}'}
                
                if approach[key] is None:
                    return {'passed': False, 'message': f'Approach at index {i} key {key} has None value'}
            
            # Verify key_steps is list of strings
            if not isinstance(approach['key_steps'], list):
                return {'passed': False, 'message': f'Approach at index {i} key_steps must be a list'}
            
            # Verify tools_mentioned is list of strings
            if not isinstance(approach['tools_mentioned'], list):
                return {'passed': False, 'message': f'Approach at index {i} tools_mentioned must be a list'}
        
        # Verify total_approaches_found matches approach_list length
        if answer_data['total_approaches_found'] != len(answer_data['approach_list']):
            return {'passed': False, 'message': 'total_approaches_found does not match approach_list length'}
        
        # Use a data tool to cross-check (query_debugging_datasets)
        datasets_check = tools['query_debugging_datasets']('debugging')
        
        # Verify data_sources_used contains valid tool names
        allowed_tools = ['search_debugging_articles', 'query_debugging_datasets', 
                         'search_csv_debugging_data', 'get_debugging_approaches']
        
        for source in answer_data['data_sources_used']:
            if source not in allowed_tools:
                return {'passed': False, 'message': f'Invalid data source: {source}'}
        
        # Verify at least two different data tools were used (excluding submit_result)
        if len(answer_data['data_sources_used']) < 2:
            return {'passed': False, 'message': 'At least two different data tools must be used'}
        
        # Check if approaches are non-empty
        if answer_data['total_approaches_found'] == 0:
            return {'passed': False, 'message': 'No approaches found'}
        
        # Check if most_common_tools is non-empty
        if len(answer_data['most_common_tools']) == 0:
            return {'passed': False, 'message': 'most_common_tools is empty'}
        
        # Additional consistency check: verify tools mentioned in approaches appear in most_common_tools
        all_mentioned_tools = []
        for approach in answer_data['approach_list']:
            if isinstance(approach.get('tools_mentioned'), list):
                all_mentioned_tools.extend(approach['tools_mentioned'])
        
        # Check if at least one tool from approaches is in most_common_tools
        common_tools_set = set(answer_data['most_common_tools'])
        mentioned_tools_set = set(all_mentioned_tools)
        
        if not common_tools_set.intersection(mentioned_tools_set):
            # This is not a failure, just a warning in details
            return {
                'passed': True,
                'message': 'Verification passed with note',
                'details': 'most_common_tools does not intersect with tools mentioned in approaches'
            }
        
        return {'passed': True, 'message': 'All verification checks passed'}
        
    except Exception as e:
        # Exception-safe: return False or dict with error
        return {'passed': False, 'message': f'Verification exception: {str(e)}'}
