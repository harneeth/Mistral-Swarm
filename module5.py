# module5.py
import json
import os
from dotenv import load_dotenv
from mistralai import Mistral

load_dotenv()


class ContentWriterClient:
    """Module 5: Text Generation"""

    def __init__(self) -> None:
        self._client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
        self._agent_id = "ag_019ca537b52c7165a736508c847fe484"

    def generate_text(self, task_description: str, context: dict | list | str) -> str:
        context_block = (
            json.dumps(context, indent=2, ensure_ascii=False)
            if isinstance(context, (dict, list))
            else str(context)
        )

        prompt = (
            "You are agent 5 in a multi-agent desktop automation system. You are used to write text.\n\n"
            f"Task:\n{task_description}\n\n"
            f"Context:\n{context_block}\n"
        )

        response = self._client.beta.conversations.start(
            agent_id=self._agent_id,
            inputs=[{"role": "user", "content": prompt}],
        )

        chunks = response.outputs[0].content
        texts = [
            c.text for c in chunks
            if getattr(c, "type", None) == "text" and hasattr(c, "text")
        ]

        return "\n".join(texts) if texts else str(chunks)