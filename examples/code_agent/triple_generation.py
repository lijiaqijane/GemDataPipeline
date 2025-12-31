"""
Example: Using TripleGenerator with dual extraction modes

This example demonstrates how to use both PR-based and function-based
triple extraction methods.
"""

import os
import json
import logging
from pathlib import Path
from agent_gem.agents.code_agent.triple_generator import TripleGenerator

# Configure logging to show TripleGenerator logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def main():
    # Setup
    github_token = os.environ.get("GITHUB_TOKEN")
    if not github_token:
        print("⚠️  Warning: GITHUB_TOKEN not set. API rate limits will be low.")
        print("   Set it with: export GITHUB_TOKEN=your_token_here")
    
    # Initialize generator with config
    config_path = Path(__file__).parent / "config" / "code_agent.yaml"
    generator = TripleGenerator(
        github_token=github_token,
        config_path=config_path
    )
    
    # Test GitHub connection
    print("\n=== Testing GitHub Connection ===")
    if not generator.test_github_connection():
        print("❌ GitHub connection failed. Please check your token.")
        return
    
    # Generate triples using both methods
    print("\n=== Generating Triples (Dual Mode) ===")
    print("This will extract:")
    print("  1. PR-based triples: Issue + Solution + Tests")
    print("  2. Function-based triples: For feature requests")
    print()
    
    triples = generator.generate_triples()
    
    # Analyze results
    print(f"\n=== Results ===")
    print(f"Total triples generated: {len(triples)}")
    
    pr_based = [t for t in triples if t.get('type') == 'pr_based']
    func_based = [t for t in triples if t.get('type') == 'function_based']
    
    print(f"  PR-based triples: {len(pr_based)}")
    print(f"  Function-based triples: {len(func_based)}")
    
    # Show some examples
    if pr_based:
        print(f"\n=== Example PR-based Triple ===")
        example = pr_based[0]
        print(f"Repo: {example['repo_name']}")
        print(f"PR #{example['pr_number']}: {example['pr_title']}")
        print(f"Issue #{example['issue_number']}: {example['issue_title']}")
        print(f"Changed files: {example['pr_changed_files']}")
        print(f"Has test changes: {example['has_test_changes']}")
        print(f"Quality score: {example['quality_score']:.2f}")
        print(f"URL: {example['pr_url']}")
    
    if func_based:
        print(f"\n=== Example Function-based Triple ===")
        example = func_based[0]
        print(f"Repo: {example['repo_name']}")
        print(f"File: {example['file_path']}")
        print(f"Function: {example['function_name']}()")
        print(f"Lines: {example['line_start']}-{example['line_end']}")
        print(f"Has docstring: {example['has_docstring']}")
        print(f"Has type hints: {example['has_type_hints']}")
        print(f"Quality score: {example['quality_score']:.2f}")
    
    # Statistics
    print(f"\n=== Quality Statistics ===")
    scores = [t['quality_score'] for t in triples]
    print(f"Average quality: {sum(scores) / len(scores):.2f}")
    print(f"Min quality: {min(scores):.2f}")
    print(f"Max quality: {max(scores):.2f}")
    
    # Repository diversity
    repos = set(t['repo_name'] for t in triples)
    print(f"\n=== Repository Diversity ===")
    print(f"Unique repositories: {len(repos)}")
    print(f"Average triples per repo: {len(triples) / len(repos):.1f}")

if __name__ == "__main__":
    main()
