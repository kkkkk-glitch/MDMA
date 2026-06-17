import openai
from typing import Dict, List, Optional


class RuleReasoningAgent:
    """
    Rule Reasoning Agent (RRA)
    Summarizes general sarcasm discrimination rules from retrieved similar samples
    and identifies sarcasm in target texts based on the summarized rules.
    """

    def __init__(self, api_key: str, model: str = "gpt-4o-2024-11-20"):
        """
        Initialize the RRA agent.

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

    def rule_mining(self, samples: List[str]) -> str:
        """
        Step 1: Mine rules from similar samples.

        Args:
            samples: List of 4 similar reference samples

        Returns:
            Extracted rules as a string
        """
        prompt = f"""You are a professional linguistic agent. Your task is to formulate in-depth judgment rules. Your ultimate goal is to assess whether a target sample contains sarcasm. However, the target sample will not be provided to you directly; instead, you will receive four associated samples related to the target sample. Based on these associated samples, you need to extract high-level judgment rationale to determine whether such samples contain sarcasm, reflect on the existing rationale, and summarize the reflections into the rules. The rules you output must be general, concise, and high-level. You should focus on extracting judgment logic from the related samples to evaluate similar samples, rather than determining whether the given sample contains sarcasm. Each rule shall be concise, clear and easy to follow, with exactly four rules in total.

Here are related samples: 
{samples[0]}, 
{samples[1]}, 
{samples[2]}, 
{samples[3]}.

Your output should strictly follow the format: Rules: {{rule1}}, {{rule2}}, {{rule3}}, {{rule4}}."""

        return self._call_llm(prompt)

    def sarcasm_detection(self, target_text: str, rules: str) -> Dict[str, str]:
        """
        Step 2: Detect sarcasm using the mined rules.

        Args:
            target_text: The text to analyze
            rules: The rules extracted from similar samples

        Returns:
            Dictionary with 'label' and 'rationale'
        """
        prompt = f"""You are a professional linguistic agent. Your task is to strictly judge whether the target text contains sarcasm according to the summarized general high-level rules, and provide clear and reasonable judgment reasons. The judgment process must follow the established rules strictly without introducing subjective inferences beyond the rules, and the conclusion shall be concise, definite and logically coherent.

Target text: {target_text}

Here are the summarized rules: {rules}

Your output should strictly follow the format: 
Label: [Output 1 if sarcasm exists, output 0 if not. Only output 1 or 0]
Rationale: [Judgment rationale corresponding to the output label]"""

        response = self._call_llm(prompt)
        return self._parse_response(response)

    def _parse_response(self, response: str) -> Dict[str, str]:
        """Parse the LLM response to extract label and rationale."""
        label = "0"
        rationale = ""

        for line in response.strip().split('\n'):
            if line.lower().startswith('label:'):
                label = line.split(':', 1)[1].strip()
            elif line.lower().startswith('rationale:'):
                rationale = line.split(':', 1)[1].strip()

        return {"label": label, "rationale": rationale}

    def run(self, target_text: str, similar_samples: List[str]) -> Dict[str, str]:
        """
        Complete RRA pipeline.

        Args:
            target_text: The text to analyze
            similar_samples: List of 4 similar reference samples

        Returns:
            Dictionary with label, rules, and rationale
        """
        print("=== RRA: Rule Mining ===")
        rules = self.rule_mining(similar_samples)
        print(f"Rules: {rules}")

        print("=== RRA: Sarcasm Detection ===")
        result = self.sarcasm_detection(target_text, rules)
        print(f"Label: {result['label']}")
        print(f"Rationale: {result['rationale']}")

        return {
            "label": result["label"],
            "rules": rules,
            "rationale": result["rationale"]
        }
