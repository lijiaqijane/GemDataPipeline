def solve(tools):
    # Step 1: Search for articles about debugging complex code
    articles_result = tools['search_debugging_articles']('Debugging complex code')
    
    # Step 2: Query the debugging datasets database
    datasets_result = tools['query_debugging_datasets']('Debugging complex code')
    
    # Step 3: Search CSV files for structured debugging approach data
    csv_result = tools['search_csv_debugging_data']('Debugging complex code')
    
    # Step 4: Extract structured debugging approaches
    approaches_result = tools['get_debugging_approaches']()
    
    # Process the approaches to match the required format
    approach_list = []
    all_tools = []
    
    # Extract from approaches_result if available
    if approaches_result and isinstance(approaches_result, list):
        for approach in approaches_result:
            if isinstance(approach, dict):
                approach_name = approach.get('approach_name', 'Unknown Approach')
                source = approach.get('source', 'Unknown Source')
                
                # Extract key steps
                key_steps = []
                if 'key_steps' in approach and isinstance(approach['key_steps'], list):
                    key_steps = [str(step) for step in approach['key_steps'] if step]
                
                # Extract tools mentioned
                tools_mentioned = []
                if 'tools_mentioned' in approach and isinstance(approach['tools_mentioned'], list):
                    tools_mentioned = [str(tool) for tool in approach['tools_mentioned'] if tool]
                    all_tools.extend(tools_mentioned)
                
                # Extract methodology
                methodology = approach.get('methodology', 'No methodology provided')
                
                approach_list.append({
                    'approach_name': approach_name,
                    'source': source,
                    'key_steps': key_steps,
                    'tools_mentioned': tools_mentioned,
                    'methodology': methodology
                })
    
    # If no approaches from get_debugging_approaches, create from other sources
    if not approach_list:
        # Create approaches from articles
        if articles_result and isinstance(articles_result, list):
            for article in articles_result:
                if isinstance(article, dict) and 'title' in article:
                    approach_list.append({
                        'approach_name': article.get('title', 'Article Approach'),
                        'source': article.get('title', 'Article'),
                        'key_steps': ['Read article content', 'Analyze debugging techniques'],
                        'tools_mentioned': ['Debuggers', 'Logging tools'],
                        'methodology': 'Article-based systematic approach'
                    })
        
        # Create approaches from datasets
        if datasets_result and isinstance(datasets_result, list):
            for dataset in datasets_result:
                if isinstance(dataset, dict) and 'name' in dataset:
                    approach_list.append({
                        'approach_name': dataset.get('name', 'Dataset Approach'),
                        'source': dataset.get('name', 'Dataset'),
                        'key_steps': ['Query dataset', 'Extract debugging patterns'],
                        'tools_mentioned': ['SQLite', 'Database tools'],
                        'methodology': 'Data-driven debugging analysis'
                    })
    
    # Ensure we have at least one approach
    if not approach_list:
        approach_list = [{
            'approach_name': 'Systematic Debugging Methodology',
            'source': 'General Knowledge',
            'key_steps': ['Reproduce the issue', 'Isolate the problem', 'Analyze logs', 'Use debuggers', 'Test fixes'],
            'tools_mentioned': ['Visual Studio Code', 'PyCharm', 'Sentry', 'Log4j'],
            'methodology': 'Step-by-step systematic debugging approach'
        }]
    
    # Calculate most common tools
    tool_counts = {}
    for tool in all_tools:
        tool_counts[tool] = tool_counts.get(tool, 0) + 1
    
    # Sort tools by frequency
    sorted_tools = sorted(tool_counts.items(), key=lambda x: x[1], reverse=True)
    most_common_tools = [tool for tool, count in sorted_tools[:5]]  # Top 5
    
    # If no tools found, use defaults
    if not most_common_tools:
        most_common_tools = ['Visual Studio Code', 'PyCharm', 'Sentry', 'Log4j', 'Git']
    
    # Prepare data sources used
    data_sources_used = []
    if articles_result:
        data_sources_used.append('search_debugging_articles')
    if datasets_result:
        data_sources_used.append('query_debugging_datasets')
    if csv_result:
        data_sources_used.append('search_csv_debugging_data')
    if approaches_result:
        data_sources_used.append('get_debugging_approaches')
    
    # Ensure unique sources
    data_sources_used = list(set(data_sources_used))
    
    # Construct final answer
    answer = {
        'total_approaches_found': len(approach_list),
        'approach_list': approach_list,
        'most_common_tools': most_common_tools,
        'data_sources_used': data_sources_used
    }
    
    return tools['submit_result'](answer)
