#!/usr/bin/env python3
import json
import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add current directory to path so we can import the package
sys.path.append(str(Path(__file__).parent))

from general_agent import (
    EnvironmentSynthesizer,
    LLMClient,
    SynthesisContext,
    TaskBundle,
)
from general_agent.executor import SandboxFusionExecutor

# Configure logging to show process details
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("Demo")

def print_step(step_name):
    print(f"\n{'='*60}")
    print(f"🚀 STEP: {step_name}")
    print(f"{'='*60}")

def mock_llm_response(prompt: str, **kwargs):
    """Generate realistic mock responses based on prompt keywords."""
    prompt_lower = prompt.lower()
    
    if "produce 3-5 structured records" in prompt_lower:
        return json.dumps([
            {"title": "Eiffel Tower", "summary": "Iconic iron lattice tower on the Champ de Mars in Paris."},
            {"title": "Louvre Museum", "summary": "World's largest art museum and a historic monument."},
            {"title": "Seine Cruise", "summary": "Scenic boat tour along the Seine River passing major landmarks."}
        ])
    
    elif "generate 2-3 specialized tools" in prompt_lower:
        return json.dumps([
            {"name": "check_attraction_status", "description": "Check if an attraction is open and available."},
            {"name": "book_ticket", "description": "Book entry tickets for a specific attraction."}
        ])
    
    elif "create a verifiable task" in prompt_lower:
        return json.dumps({
            "name": "Paris Itinerary Booking",
            "description": "Plan a visit to the Eiffel Tower: check if it's open, and if so, book a ticket.",
            "solution_code": """
def solve(tools):
    status = tools['check_attraction_status']('Eiffel Tower')
    result = {'status': status}
    if 'Open' in str(status):
        booking = tools['book_ticket']('Eiffel Tower')
        result['booking'] = booking
    return result
""",
            "verification_code": """
def verify(tools, answer):
    # Verify we got a booking confirmation if it was open
    if not isinstance(answer, dict):
        return False
    if 'booking' in answer:
        return 'Confirmed' in str(answer['booking'])
    return True
"""
        })
    
    elif "increase the task difficulty" in prompt_lower:
        return json.dumps({
            "name": "Complex Paris Tour",
            "description": "Plan a multi-stop tour. Check availability for Eiffel Tower AND Louvre. Book both if available.",
            "solution_code": """
def solve(tools):
    eiffel_status = tools['check_attraction_status']('Eiffel Tower')
    louvre_status = tools['check_attraction_status']('Louvre')
    
    bookings = {}
    if 'Open' in str(eiffel_status):
        bookings['Eiffel Tower'] = tools['book_ticket']('Eiffel Tower')
    if 'Open' in str(louvre_status):
        bookings['Louvre'] = tools['book_ticket']('Louvre')
        
    return {'bookings': bookings}
""",
            "verification_code": """
def verify(tools, answer):
    # Verify we have at least one booking
    bookings = answer.get('bookings', {})
    return len(bookings) > 0
"""
        })
        
    elif "generate 1-2 additional specialized tools" in prompt_lower:
         return json.dumps([
            {"name": "get_weather_forecast", "description": "Get weather forecast for a location."}
        ])
         
    elif "repair the bundle" in prompt_lower:
        return json.dumps({
            "name": "Repaired Task",
            "description": "Repaired task description",
            "solution_code": """
def solve(tools):
    # Simplified solution for repair
    return tools['check_attraction_status']('Eiffel Tower')
""",
            "verification_code": "def verify(tools, answer): return True"
        })

    return "{}"

def mock_sandbox_execution(code: str, language: str = "python"):
    """Simulate code execution in the sandbox."""
    # Determine if this is a solution run or verification run
    if "solve(" in code:
        print(f"    [Sandbox] Running Solution Code...")
        # Simulate tool outputs
        stdout = json.dumps({
            "status": "Open (09:00 - 23:00)", 
            "booking": "Booking Confirmed: #REF123",
            "bookings": {
                "Eiffel Tower": "Confirmed #ET1",
                "Louvre": "Confirmed #LV2"
            }
        })
    elif "verify(" in code:
        print(f"    [Sandbox] Running Verification Code...")
        stdout = "true"
    else:
        stdout = "{}"
        
    return {
        "status": "success",
        "stdout": stdout,
        "stderr": "",
        "return_code": 0
    }

def main():
    print(f"\n{'#'*80}")
    print(f"GENERAL AGENT BUNDLE - DEMO RUN")
    print(f"{'#'*80}\n")
    print("This demo simulates the full Environment Synthesis workflow using mocks.")
    print("It demonstrates: Data Curation -> Tool Synthesis -> Task Generation -> Verification.\n")

    # 1. Setup Mocks
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.simple_complete.side_effect = mock_llm_response
    
    mock_executor = MagicMock(spec=SandboxFusionExecutor)
    mock_executor.side_effect = mock_sandbox_execution
    
    # 2. Initialize Synthesizer
    synth = EnvironmentSynthesizer(mock_llm)
    # Ensure sandbox output is within the package directory
    sandbox_dir = Path(__file__).parent / "sandbox_demo"
    
    # Clean previous demo run
    if sandbox_dir.exists():
        import shutil
        try:
            shutil.rmtree(sandbox_dir)
        except:
            pass
            
    # 3. Patch dependencies to inject our mocks into the synthesis process
    print_step("Initializing Synthesis Context")
    print(f"Category: 'Paris Travel Planning'")
    print(f"Sandbox Directory: {sandbox_dir}")
    
    with patch('general_agent.synthesis.SandboxFusionExecutor', return_value=mock_executor):
        # Trigger sandbox usage via env vars
        with patch.dict('os.environ', {
            'SANDBOX_FUSION_URL': 'http://mock-service',
            'SANDBOX_FUSION_TIMEOUT': '10'
        }):
            
            # --- START SYNTHESIS ---
            
            # We call the public API, but we'll intercept logs to print progress
            print_step("Seeding Database & Synthesizing Tools")
            
            bundles = synth.synthesize(
                category="Paris Travel Planning",
                sandbox=sandbox_dir,
                rounds=2,
                validate=True,
                fail_soft=True,
                persist=True,
                use_sandbox_fusion=True,
                use_docker=True
            )
            
            # --- RESULTS ---
            
            print_step("Synthesis Complete")
            print(f"Generated {len(bundles)} verified task bundles.\n")
            
            for i, bundle in enumerate(bundles):
                print(f"📦 Task Bundle #{i+1}")
                print(f"   Name: {bundle.name}")
                print(f"   Description: {bundle.description}")
                print(f"   Difficulty: {bundle.difficulty}")
                print(f"   Execution Env: SandboxFusion")
                print(f"   Code Snippet: {bundle.solution_code.strip().splitlines()[0]} ...")
                print("-" * 40)

            # Check artifacts
            tasks_file = sandbox_dir / "tasks.json"
            if tasks_file.exists():
                print(f"\n✅ Artifacts persisted to: {tasks_file}")
                data = json.loads(tasks_file.read_text())
                print(f"   Database Records: {len(data['records'])}")
                print(f"   Tools Generated: {len(data['tooling'])}")
            else:
                print("\n❌ Failed to persist artifacts.")

if __name__ == "__main__":
    main()
