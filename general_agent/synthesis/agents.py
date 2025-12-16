"""Agent classes for modular synthesis orchestration."""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Tuple

from .context import SynthesisContext
from .task_bundle import TaskBundle

if TYPE_CHECKING:
    from .synthesizer import EnvironmentSynthesizer


@dataclass
class EnvironmentAgent:
    """Agent responsible for building sandbox context and seeding the environment database."""

    synthesizer: "EnvironmentSynthesizer"

    def prepare(
        self,
        category: str,
        sandbox: Path,
        use_sandbox_fusion: bool,
    ) -> SynthesisContext:
        """Build SynthesisContext and seed the LocalDatabase."""
        ctx = self.synthesizer.build_context(
            category=category,
            sandbox=sandbox,
            use_sandbox_fusion=use_sandbox_fusion,
        )
        # According to paper design, environment construction stage is responsible for preparing database records
        self.synthesizer.seed_database(ctx)
        return ctx


@dataclass
class ToolAgent:
    """Agent that manages toolset construction and augmentation."""

    synthesizer: "EnvironmentSynthesizer"

    def build_initial_tools(self, ctx: SynthesisContext) -> None:
        """Synthesize the first batch of task-oriented tools from the environment."""
        self.synthesizer.synthesize_tools(ctx)

    def maybe_augment(
        self,
        ctx: SynthesisContext,
        bundle: TaskBundle,
        failure_reason: str,
        answer: Any = None,
    ) -> bool:
        """Try to augment the current toolset when tasks fail due to missing capabilities."""
        return self.synthesizer.augment_toolset(ctx, bundle, failure_reason, answer)


@dataclass
class TaskAgent:
    """Agent that proposes and refines task + solution code."""

    synthesizer: "EnvironmentSynthesizer"

    def propose_initial(self, ctx: SynthesisContext, difficulty: int = 1) -> TaskBundle:
        """Generate an initial, non-trivial task for the given environment."""
        raw = self.synthesizer.propose_task(ctx, difficulty=difficulty)
        return self.synthesizer._ensure_substantive_task(
            ctx, raw, "Initial task quality gate"
        )

    def refine(self, ctx: SynthesisContext, current: TaskBundle, round_index: int) -> TaskBundle:
        """Increase task difficulty by creating a refined variant."""
        refined = self.synthesizer.refine_task(ctx, current)
        reason = f"Refined task quality gate (round {round_index})"
        return self.synthesizer._ensure_substantive_task(ctx, refined, reason)


@dataclass
class ValidationAgent:
    """Agent that executes and verifies synthesized tasks, and repairs them when needed."""

    synthesizer: "EnvironmentSynthesizer"

    def ensure_valid(
        self,
        ctx: SynthesisContext,
        bundle: TaskBundle,
        fail_soft: bool,
    ) -> Tuple[TaskBundle, Any]:
        """Run solution + verifier loop with repair and tool augmentation."""
        return self.synthesizer.ensure_valid(ctx, bundle, fail_soft=fail_soft)

