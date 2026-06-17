import openai
from typing import Dict, List, Optional, Tuple


class CounterfactualReasoningAgent:
    """
    Counterfactual Reasoning Agent (CRA)
    Constructs dual reasoning branches for sarcastic and non-sarcastic hypotheses
    for comparative verification.
    """

    def __init__(self, api_key: str, model: str = "gpt-4o-2024-11-20"):
        """
        Initialize the CRA agent.

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

    def hypothesis_construction(self, target_text: str) -> Dict[str, str]:
        """
        Step 1: Construct dual hypotheses (sarcastic and non-sarcastic).

        Args:
            target_text: The text to analyze

        Returns:
            Dictionary with 'sarcasm_rationale' and 'non_sarcasm_rationale'
        """
        prompt = f"""You are a professional linguistic agent. Your task is to elaborate respectively on the grounds for classifying the target text sample as sarcastic and non-sarcastic, and explain in detail why the textual content conforms to the corresponding category, so as to demonstrate the rationality of both labeling schemes.

Target text: {target_text}

Your output should strictly follow the format: 
Rationale for sarcasm: [Assuming the label of the target sample is sarcasm, explain the rationality supporting this label]
Rationale for non-sarcasm: [Assuming the label of the target sample is non-sarcasm, explain the rationality supporting this label]"""

        response = self._call_llm(prompt)
        return self._parse_hypotheses(response)

    def _parse_hypotheses(self, response: str) -> Dict[str, str]:
        """Parse the hypothesis construction response."""
        sarcasm_rationale = ""
        non_sarcasm_rationale = ""

        for line in response.strip().split('\n'):
            line_lower = line.lower()
            if line_lower.startswith('rationale for sarcasm:'):
                sarcasm_rationale = line.split(':', 1)[1].strip()
            elif line_lower.startswith('rationale for non-sarcasm:'):
                non_sarcasm_rationale = line.split(':', 1)[1].strip()

        return {
            "sarcasm_rationale": sarcasm_rationale,
            "non_sarcasm_rationale": non_sarcasm_rationale
        }

    def sarcasm_detection(self, target_text: str, hypotheses: Dict[str, str]) -> Dict[str, str]:
        """
        Step 2: Detect sarcasm by comparing the two hypotheses.

        Args:
            target_text: The text to analyze
            hypotheses: The constructed hypotheses

        Returns:
            Dictionary with 'label' and 'rationale'
        """
        prompt = f"""You are a professional linguistic agent. Your task is to judge whether the target text sample belongs to sarcastic text. The system will provide detailed grounds for judging it as sarcastic and non-sarcastic respectively. You need to sort out and discuss the reasoning process step by step, and finally determine whether the target sample contains sarcasm.

Target text: {target_text}

Here are the grounds for sarcasm and non-sarcasm: 
Rationale for sarcasm: {hypotheses['sarcasm_rationale']}
Rationale for non-sarcasm: {hypotheses['non_sarcasm_rationale']}

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

    def run(self, target_text: str) -> Dict[str, str]:
        """
        Complete CRA pipeline.

        Args:
            target_text: The text to analyze

        Returns:
            Dictionary with label, hypotheses, and rationale
        """
        print("=== CRA: Hypothesis Construction ===")
        hypotheses = self.hypothesis_construction(target_text)
        print(f"Sarcasm Rationale: {hypotheses['sarcasm_rationale']}")
        print(f"Non-Sarcasm Rationale: {hypotheses['non_sarcasm_rationale']}")

        print("=== CRA: Sarcasm Detection ===")
        result = self.sarcasm_detection(target_text, hypotheses)
        print(f"Label: {result['label']}")
        print(f"Rationale: {result['rationale']}")

        return {
            "label": result["label"],
            "sarcasm_rationale": hypotheses["sarcasm_rationale"],
            "non_sarcasm_rationale": hypotheses["non_sarcasm_rationale"],
            "rationale": result["rationale"]
        }
