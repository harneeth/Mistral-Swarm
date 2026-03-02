import os
import json
from typing import Dict, Any
from jsonschema import validate, ValidationError
from mistralai import Mistral


class InteractionModule:
    """
    Module 1:
    - Handles conversation
    - Creates initial state
    - Extracts structured goal
    - Signals escalation needs to Module 6
    """

    def __init__(self, model="mistral-large-latest"):
        self.client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
        self.model = model

        self.schema = {
            "type": "object",
            "properties": {
                "goal": {"type": "string"},
                "entities": {"type": "object"},
                "permissions_required": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "requires_info_module": {"type": "boolean"},
                "requires_writing_module": {"type": "boolean"},
                "missing_information": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": [
                "goal",
                "entities",
                "permissions_required",
                "requires_info_module",
                "requires_writing_module",
                "missing_information"
            ]
        }

    def initialize_run(self, user_input: str) -> Dict[str, Any]:

        system_prompt = """
You are Module 1 of a PC automation system.

Extract structured intent from the user request.

Rules:
- If unclear, list fields in missing_information.
- If task requires external knowledge (how to use an app), set requires_info_module=true.
- If task requires complex writing, set requires_writing_module=true.
- List permissions/passwords needed.
- Output STRICT JSON only. No explanations.

You should be helpful to the user. For example, if they are writing a mail, you can ask for the recipient, the attachments, etc., but you can also write the body of the mail using information and through "requires_writing_module".

Format:
{
  "goal": "...",
  "entities": {...},
  "permissions_required": [...],
  "requires_info_module": true/false,
  "requires_writing_module": true/false,
  "missing_information": [...]
}

CRITICAL:
- Do NOT use markdown.
- Do NOT wrap in ```json
- Output raw JSON only.
- No explanation text.
"""

        response = self.client.chat.complete(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            temperature=0
        )

        output = response.choices[0].message.content
        # print("\nRAW MODEL OUTPUT:\n", repr(output))

        try:
            structured = json.loads(output)
            validate(instance=structured, schema=self.schema)
        except (json.JSONDecodeError, ValidationError) as e:
            raise ValueError(f"Invalid Interaction output: {e}")

        # Initialize base state for Module 6
        initial_state = {
            "goal": structured["goal"],
            "entities": structured["entities"],
            "permissions_granted": [],
            "step_history": [],
            "requires_info_module": structured["requires_info_module"],
            "requires_writing_module": structured["requires_writing_module"],
            "missing_information": structured["missing_information"]
        }

        return initial_state
    
    def request_user_input(self, prompt: str) -> str:
        return input(prompt)
