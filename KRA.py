import openai
from typing import Dict, List, Optional


class KnowledgeReasoningAgent:
    """
    Knowledge Reasoning Agent (KRA)
    Explores the deep pragmatic meaning of texts relying on external commonsense knowledge,
    addressing the limitation that implicit sarcasm cannot be identified merely through literal features.
    """

    def __init__(self, api_key: str, model: str = "gpt-4o-2024-11-20"):
        """
        Initialize the KRA agent.

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

    def knowledge_summary(self, target_text: str) -> str:
        """
        Step 1: Extract keywords and summarize commonsense knowledge.

        Args:
            target_text: The text to analyze

        Returns:
            Commonsense knowledge about keywords
        """
        prompt = f"""You are a professional linguistic agent. Your task is to extract keywords from given text samples. Select 2 to 4 words, then sort out the corresponding basic common sense for each word, focusing on summarizing their inherent characteristics, as well as public perceptions and mainstream social views towards them.

Target text: {target_text}

Your output should strictly follow the format: Knowledge: {{Knowledge1}}, {{Knowledge2}}, {{Knowledge3}}, {{Knowledge4}}."""

        return self._call_llm(prompt)

    def sarcasm_detection(self, target_text: str, knowledge: str) -> Dict[str, str]:
        """
        Step 2: Detect sarcasm using commonsense knowledge.

        Args:
            target_text: The text to analyze
            knowledge: Commonsense knowledge about keywords

        Returns:
            Dictionary with 'label' and 'rationale'
        """
        prompt = f"""You are a professional linguistic agent. Your task is to analyze whether the target text contains sarcastic implications based on acquired knowledge, and finally output the judgment result (Sarcastic / Non-sarcastic). The judgment must strictly follow established knowledge and criteria without any subjective speculation beyond the rules, and the conclusion shall be concise, explicit and logically rigorous.

Target text: {target_text}

Here is the summarized Knowledge: {knowledge}

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
        Complete KRA pipeline.

        Args:
            target_text: The text to analyze

        Returns:
            Dictionary with label, knowledge, and rationale
        """
        print("=== KRA: Knowledge Summary ===")
        knowledge = self.knowledge_summary(target_text)
        print(f"Knowledge: {knowledge}")

        print("=== KRA: Sarcasm Detection ===")
        result = self.sarcasm_detection(target_text, knowledge)
        print(f"Label: {result['label']}")
        print(f"Rationale: {result['rationale']}")

        return {
            "label": result["label"],
            "knowledge": knowledge,
            "rationale": result["rationale"]
        }
