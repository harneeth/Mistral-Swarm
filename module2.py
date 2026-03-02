import os
import json
from typing import Dict, Any
from jsonschema import validate, ValidationError
from mistralai import Mistral


class ParserModule:
    """
    Module 2:
    - Decomposes goal into atomic steps
    - Signals escalation needs
    - Can re-run with additional info from Module 3
    """

    def __init__(self, model="mistral-large-latest"):
        self.client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
        self.model = model

        self.schema = {
            "type": "object",
            "properties": {
                "goal": {"type": "string"},
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "step": {"type": "string"},
                            "description": {"type": "string"},
                            "requires_info_module": {"type": "boolean"},
                            "requires_writing_module": {"type": "boolean"},
                            "permission_required": {"type": "boolean"}
                        },
                        "required": [
                            "step",
                            "description",
                            "requires_info_module",
                            "requires_writing_module",
                            "permission_required"
                        ]
                    }
                },
                "requires_info_module": {"type": "boolean"}
            },
            "required": ["steps", "requires_info_module"]
        }

    def generate_plan(self, state: Dict[str, Any], external_knowledge: Dict[str, Any] = None) -> Dict[str, Any]:

        system_prompt = """
You are Module 2 of a PC automation system.

Break the goal into small executable steps.

Rules:
1. Each step must perform ONE action.
2. If you do not know how to perform a step (for example in apps that you do not know how to operate), mark requires_info_module=true. Do not hallucinate.
3. If step requires writing (long email, document), mark requires_writing_module=true.
4. Mark if permission or password required.
5. Include estimated_risk: low, medium, high.
6. Output STRICT JSON only.

Your role:
- DO NOT describe UI actions.
- DO NOT describe clicks, typing, or button presses.
- That is Module 4's responsibility.

You must output TASK-LEVEL steps only.

Example:
GOOD:
- open_email_client
- compose_email
- attach_file
- send_email

BAD:
- click compose button
- type into subject field
- click send button

Rules:
- Each task must represent a logical operation.
- No UI-level detail.
- If you do not know how to perform a task, mark requires_info_module=true.
- If long text writing is required, mark requires_writing_module=true.
- Output raw JSON only.
- No markdown.

CRITICAL:
- Output raw JSON only.
- Do NOT wrap in markdown.
- Do NOT include ```json.
- Do NOT include explanations.
- You MUST include these top-level fields:
  - "requires_info_module"
  - "requires_writing_module"
- These fields MUST always be present.
- Even if false.
- If any required field is missing, the system will crash.
- Do not omit any required field.
- Output raw JSON only.
- Do not wrap in markdown.

================== STRICT OUTPUT CONTRACT ==================

You MUST return a JSON object that EXACTLY matches the following schema.

Do NOT:
- Add extra fields
- Omit required fields
- Rename any field
- Change field names
- Add new properties
- Remove properties
- Use synonyms
- Use markdown
- Add explanations
- Add comments

If you violate the schema, the system will crash.

The JSON structure MUST be:

{
  "steps": [
    {
      "step": "string",
      "description": "string",
      "requires_info_module": true or false,
      "requires_writing_module": true or false,
      "permission_required": true or false
    }
  ],
  "requires_info_module": true or false
}

Rules:
- All fields are mandatory.
- All boolean fields must be explicitly true or false.
- "description" must always be present.
- Do NOT include any other fields such as:
  - password_required
  - estimated_risk
  - notes
  - metadata
  - extra properties
- No additional properties are allowed.

Before returning your answer:
1. Verify that all required fields exist.
2. Verify that there are NO extra fields.
3. Verify the JSON is syntactically valid.
4. Verify field names match exactly.
"""

        user_payload = {
            "state": state,
            "external_knowledge": external_knowledge
        }

        response = self.client.chat.complete(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload)}
            ],
            temperature=0
        )

        output = response.choices[0].message.content.strip()

        try:
            structured = json.loads(output)
            validate(instance=structured, schema=self.schema)
        except (json.JSONDecodeError, ValidationError) as e:
            raise ValueError(f"Invalid Parser output: {e}")

        return structured
