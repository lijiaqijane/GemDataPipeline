from pathlib import Path

from general_agent import EnvironmentSynthesizer, LLMClient


def main() -> None:
    llm = LLMClient.from_env()
    synth = EnvironmentSynthesizer(llm)
    bundles = synth.synthesize(
        category="travel itinerary planning",
        sandbox=Path("sandbox/travel"),
        rounds=2,
    )
    for bundle in bundles:
        print(f"[{bundle.difficulty}] {bundle.name}: {bundle.description}")


if __name__ == "__main__":
    main()

