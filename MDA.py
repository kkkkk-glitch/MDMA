import openai
from typing import Dict, List, Optional, Tuple


class MetaIterativeAgent:
    """
    Meta-Decision Agent (MDA)
    Implements iterative optimization to reach consensus among multiple expert agents
    through confidence evaluation, reflective revision, and iterative negotiation.
    """

    def __init__(self, api_key: str, model: str = "gpt-4o-2024-11-20"):
        """
        Initialize the MDA agent.

        Args:
            api_key: OpenAI API key
            model: OpenAI model to use
        """
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.temperature = 0  # Set to 0 for deterministic outputs

    def _call_llm(self, prompt: str) -> str:
        """Helper method to call OpenAI API."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            max_tokens=256
        )
        return response.choices[0].message.content

    def score(self, rationale: str) -> float:
        """
        Score function: Evaluate reasoning quality across three dimensions.

        Args:
            rationale: The reasoning rationale to evaluate

        Returns:
            Confidence score (sum of relevance, logic, and persuasiveness)
        """
        prompt = f"""You are a Meta-Decision Agent. Evaluate the following reasoning rationale across three dimensions (each 0-1):
1. Relevance: How relevant is the reasoning to the task?
2. Logic: How logically sound is the reasoning?
3. Persuasiveness: How persuasive and convincing is the reasoning?

Rationale: {rationale}

Output only the scores as: Relevance: X, Logic: X, Persuasiveness: X"""

        response = self._call_llm(prompt)

        # Parse scores from response
        scores = {}
        for line in response.strip().split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                try:
                    scores[key.strip()] = float(value.strip())
                except ValueError:
                    scores[key.strip()] = 0.0

        return sum(scores.values())

    def reflect(self,
                agent_name: str,
                target_text: str,
                original_label: int,
                original_rationale: str,
                reference_clue: str) -> Tuple[int, str, str]:
        """
        Reflect function: Guide agent to revise its reasoning.

        Args:
            agent_name: Name of the agent to reflect
            target_text: The text being analyzed
            original_label: Original label from the agent
            original_rationale: Original reasoning rationale
            reference_clue: The reasoning clue from the highest-scoring agent

        Returns:
            Tuple of (new_label, new_clue, new_rationale)
        """
        prompt = f"""You are a Meta-Decision Agent guiding reflection. The {agent_name} previously analyzed the following text:

Target text: {target_text}
Original label: {original_label}
Original rationale: {original_rationale}

Now, review and revise your reasoning based on this reference clue from the highest-scoring agent:
{reference_clue}

Provide your revised analysis. Output in the following format:
Clue: [Updated reasoning clue]
Label: [Output 1 if sarcasm exists, output 0 if not]
Rationale: [Updated judgment rationale]"""

        response = self._call_llm(prompt)
        return self._parse_reflect_response(response)

    def _parse_reflect_response(self, response: str) -> Tuple[int, str, str]:
        """
        Parse the reflection response.

        Args:
            response: Raw LLM response from reflection

        Returns:
            Tuple of (label, clue, rationale)
        """
        clue = ""
        label = 0
        rationale = ""

        for line in response.strip().split('\n'):
            line_lower = line.lower()
            if line_lower.startswith('clue:'):
                clue = line.split(':', 1)[1].strip()
            elif line_lower.startswith('label:'):
                label = int(line.split(':', 1)[1].strip())
            elif line_lower.startswith('rationale:'):
                rationale = line.split(':', 1)[1].strip()

        return label, clue, rationale

    def iterative_optimization(self,
                               target_text: str,
                               agents_outputs: List[Dict],
                               max_iterations: int = 3) -> Dict:
        """
        Complete iterative optimization process (Algorithm 1).

        Args:
            target_text: The text being analyzed
            agents_outputs: List of dicts with keys: 'name', 'label', 'clue', 'rationale'
            max_iterations: Maximum number of iterations

        Returns:
            Dictionary with final label, rationale, and iteration history
        """
        # Extract agent data
        names = [out['name'] for out in agents_outputs]
        labels = [out['label'] for out in agents_outputs]
        clues = [out['clue'] for out in agents_outputs]
        rationales = [out['rationale'] for out in agents_outputs]

        iteration = 0
        history = []

        print("=== MIA: Iterative Optimization ===")

        while iteration < max_iterations:
            print(f"\n--- Iteration {iteration + 1} ---")

            # Score each agent's rationale
            scores = []
            for rationale in rationales:
                score = self.score(rationale)
                scores.append(score)
                print(f"Score for {names[len(scores) - 1]}: {score:.2f}")

            # Find highest-scoring agent
            best_idx = scores.index(max(scores))
            best_label = labels[best_idx]
            best_clue = clues[best_idx]
            best_rationale = rationales[best_idx]
            best_name = names[best_idx]

            print(f"Best agent: {best_name} with label {best_label}, score: {scores[best_idx]:.2f}")

            # Check for consensus
            if labels[0] == labels[1] == labels[2]:
                print("Consensus reached!")
                history.append({
                    "iteration": iteration + 1,
                    "consensus": True,
                    "labels": labels.copy(),
                    "scores": scores.copy()
                })
                return {
                    "final_label": best_label,
                    "final_rationale": best_rationale,
                    "best_agent": best_name,
                    "iteration_history": history,
                    "consensus_reached": True
                }

            # Reflect on non-consensus agents
            for i in range(len(agents_outputs)):
                if labels[i] != best_label:
                    print(f"Reflecting on {names[i]}...")
                    new_label, new_clue, new_rationale = self.reflect(
                        agent_name=names[i],
                        target_text=target_text,
                        original_label=labels[i],
                        original_rationale=rationales[i],
                        reference_clue=best_clue
                    )
                    labels[i] = new_label
                    clues[i] = new_clue
                    rationales[i] = new_rationale
                    print(f"  {names[i]} updated: label {new_label}")

            history.append({
                "iteration": iteration + 1,
                "consensus": False,
                "labels": labels.copy(),
                "scores": scores.copy(),
                "best_agent": best_name
            })

            iteration += 1

        # If no consensus reached, return best from last iteration
        print("\nMax iterations reached. Returning best agent's result.")
        final_scores = [self.score(r) for r in rationales]
        best_final_idx = final_scores.index(max(final_scores))

        return {
            "final_label": labels[best_final_idx],
            "final_rationale": rationales[best_final_idx],
            "best_agent": names[best_final_idx],
            "iteration_history": history,
            "consensus_reached": False
        }
