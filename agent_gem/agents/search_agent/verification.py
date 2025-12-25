from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple

from agent_gem.llm import LLMClient

from .answer_generator import CandidateAnswer
from .prompt_mixin import PromptMixin
from .question_constructor import QuestionAnswerPair

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """Result of answer verification.

    Attributes:
        is_correct: Whether the answer is verified as correct
        verification_evidence: List of evidence strings used for verification
    """

    is_correct: bool
    verification_evidence: List[str]


class VerifierMixin(PromptMixin):
    """Verification agent with search capabilities that validates answers through multiple passes.

    This mixin provides functionality to verify answers by checking them against
    search context and determining whether samples should be retained based on
    verification criteria.
    """

    def _verify_candidates(
        self,
        llm: LLMClient,
        qa_pair: QuestionAnswerPair,
        candidates: List[CandidateAnswer],
        require_all_incorrect: bool = True,
    ) -> Tuple[bool, List[VerificationResult]]:
        """Verify ground-truth answer and all candidate answers through multiple passes.

        Args:
            llm: LLM client for verification
            qa_pair: The question-answer pair with ground-truth answer
            candidates: List of candidate answers to verify
            require_all_incorrect: If True, only retain samples where ground-truth is correct
                                  and all candidates are verifiably incorrect

        Returns:
            Tuple of (should_retain, verification_results):
            - should_retain: Whether this sample should be retained based on verification
            - verification_results: List of verification results for each candidate answer
        """
        # Verify ground-truth answer
        gt_result = self._verify_answer(llm, qa_pair.question, qa_pair.answer, qa_pair.search_context)
        logger.debug(
            f"Ground-truth verification result: {gt_result.is_correct} "
            f"for question: {qa_pair.question[:50]}..."
        )

        # Verify all candidate answers
        candidate_results: List[VerificationResult] = []
        for candidate in candidates:
            result = self._verify_answer(
                llm,
                qa_pair.question,
                candidate.answer,
                qa_pair.search_context,
            )
            candidate_results.append(result)
            logger.debug(
                f"Candidate answer verification result: {result.is_correct} "
                f"for config: {candidate.generator_config.name}"
            )

        # Determine if sample should be retained
        should_retain = self._should_retain(gt_result, candidate_results, require_all_incorrect)

        return should_retain, candidate_results

    def _verify_answer(
        self,
        llm: LLMClient,
        question: str,
        answer: str,
        search_context: List[str],
    ) -> VerificationResult:
        """Verify a single answer against search context.

        Args:
            llm: LLM client for verification
            question: The question being answered
            answer: The answer to verify
            search_context: List of search result strings to verify against

        Returns:
            VerificationResult with correctness status and evidence
        """
        # Early return if context or answer is missing
        if not search_context or not answer:
            logger.debug("Missing context or answer for verification")
            return VerificationResult(
                is_correct=False,
                verification_evidence=[],
            )

        context = "\n\n".join(search_context)
        prompt = self.VERIFICATION_PROMPT.format(question=question, answer=answer, context=context)

        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that verifies answers.",
                },
                {"role": "user", "content": prompt},
            ]
            response = llm.chat_completion(messages=messages, temperature=0.0, max_tokens=50)
            verification = "YES" in response.upper() or "CORRECT" in response.upper()
        except Exception as e:
            logger.warning(f"Error in answer verification: {e}")
            verification = False

        return VerificationResult(
            is_correct=verification,
            verification_evidence=search_context,
        )

    def _should_retain(
        self,
        gt_result: VerificationResult,
        candidate_results: List[VerificationResult],
        require_all_incorrect: bool,
    ) -> bool:
        """Determine if sample should be retained based on verification results.

        Args:
            gt_result: Verification result for the ground-truth answer
            candidate_results: List of verification results for candidate answers
            require_all_incorrect: If True, requires all candidates to be incorrect

        Returns:
            True if the sample should be retained, False otherwise
        """
        # Ground-truth must be correct
        if not gt_result.is_correct:
            logger.debug("Ground-truth answer is incorrect, rejecting sample")
            return False

        if require_all_incorrect:
            # All candidate answers must be incorrect
            all_incorrect = all(not r.is_correct for r in candidate_results)
            if all_incorrect:
                logger.debug("All candidate answers are incorrect, retaining sample")
            else:
                logger.debug("Not all candidate answers are incorrect, rejecting sample")
            return all_incorrect
        else:
            # At least one candidate answer must be incorrect
            has_incorrect = any(not r.is_correct for r in candidate_results)
            if has_incorrect:
                logger.debug("At least one candidate is incorrect, retaining sample")
            else:
                logger.debug("All candidates are correct, rejecting sample")
            return has_incorrect
