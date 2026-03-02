"""
module4.py — Intelligent Visual Execution Engine (Agent 4)
===========================================================

Architecture:
Logical Step → Visual Planner → Low-level Actions → Execute → Loop until done

Requires:
- module5.py (ContentWriterClient)
- MISTRAL_API_KEY in .env
"""

from __future__ import annotations

import os
import re
import json
import time
from enum import Enum
from typing import Any, List

import pyautogui
import torch
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from pydantic import BaseModel, Field
from ultralytics import YOLO
from mistralai import Mistral

from module5 import ContentWriterClient

load_dotenv()
pyautogui.FAILSAFE = True


# ============================================================
# ACTION DEFINITIONS
# ============================================================

class ActionType(str, Enum):
    MOVE = "MOVE"
    CLICK = "CLICK"
    DOUBLE_CLICK = "DOUBLE_CLICK"
    SCROLL = "SCROLL"
    TYPE = "TYPE"
    PRESS = "PRESS"
    WAIT = "WAIT"
    OPEN_APP = "OPEN_APP"
    OPEN_URL = "OPEN_URL"
    ZOOM = "ZOOM"


class Action(BaseModel):
    action_type: ActionType
    params: dict[str, Any] = Field(default_factory=dict)
    description: str = ""


# ============================================================
# TASK MODELS
# ============================================================

class ExecutionStep(BaseModel):
    step_id: int
    description: str
    generate_text: bool = False
    text_task_description: str | None = None
    text_context: dict | list | str | None = None


class HighLevelTask(BaseModel):
    task_id: str
    description: str
    steps: List[ExecutionStep]
    context: dict | list | str = Field(default_factory=dict)


class ExecutionResult(BaseModel):
    task_id: str
    status: str
    actions: List[Action] = Field(default_factory=list)
    error: str | None = None


# ============================================================
# GUI DETECTOR (Vision)
# ============================================================

class GUIDetector:
    _REPO_ID = "Salesforce/GPA-GUI-Detector"
    _FILENAME = "model.pt"

    def __init__(self, confidence: float = 0.05):
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        model_path = hf_hub_download(repo_id=self._REPO_ID, filename=self._FILENAME)
        self._model = YOLO(model_path)
        self._confidence = confidence

    def detect(self) -> list[dict]:
        screenshot = pyautogui.screenshot()
        results = self._model.predict(
            source=screenshot,
            conf=self._confidence,
            device=self._device,
            verbose=False
        )[0]

        detections = []

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            detections.append({
                "label": results.names[int(box.cls)],
                "confidence": float(box.conf),
                "bbox": {
                    "x": x1,
                    "y": y1,
                    "width": x2 - x1,
                    "height": y2 - y1
                }
            })

        return detections


# ============================================================
# VISUAL PLANNER (LLM)
# ============================================================

class VisualPlanner:

    def __init__(self, model: str = "mistral-medium-latest"):
        self.client = Mistral(api_key="pMA2zFwdxy34txIjHZMrbCmvAKvijDxa")
        self.model = model

        self.system_prompt = self.system_prompt = """
You are a visual planner for a desktop automation agent.

You receive:
1. A logical step description.
2. Current screen detections.
3. Full execution state including:
   - previous failures
   - last error
   - consecutive failure count
   - execution history

Your job:
- Decide the NEXT low-level actions needed.
- If repeated failures occurred (>=3), change strategy.
- Use execution_history.ui_change to verify whether prior actions changed the UI.
- If the UI does not contain required elements, return:
    status = "error"
    message explaining what is missing.

Available action types:
MOVE {x,y}
CLICK {button}
TYPE {text}
PRESS {key}
WAIT {ms}
SCROLL {direction, amount}
OPEN_APP {name}
OPEN_URL {url}

STRICT JSON:

{
  "actions": [
    {
      "action_type": "MOVE",
      "params": {"x":100,"y":200},
      "description": "reason"
    }
  ],
  "status": "in_progress" | "done" | "error",
  "message": "explanation"
}

Rules:
- If stuck after multiple attempts, return status="error".
- Do not hallucinate UI elements.
- Use bounding box coordinates for clicks.
- Do not output markdown.
- Only valid JSON.
- If recent interactive actions caused no UI change, avoid repeating them unchanged.

Termination Rules:

- If the logical step objective is already satisfied on screen,
  return:
      "actions": [],
      "status": "done"

- Do NOT repeat the same action twice if it already succeeded.

- If the target application is already open, do not reopen it.
  Return status="done".

Tip:
- It is a good practice to try and utilize the Window key or a different system in Mac which you can find out by requesting for information, to open applications as it is more reliable than trying to open the app directly which can result in errors if the app's name is not recognized or if there are multiple versions of the app installed. However, this is just a suggestion, and only following this could lead to problems. Thus, use your best judgement. An example is: apps are simpler to open with windows key, but files can be simpler to open directly with their location, but the choice and course of action depends on you - so use your best judgement and do not be restricted.
"""

    def plan(self, step_description: str, detections: list[dict], state: dict | None = None) -> dict:

        payload = json.dumps({
            "step": step_description,
            "detections": detections,
            "state": state or {}
        })

        response = self.client.chat.complete(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": payload}
            ],
            temperature=0
        )

        raw = response.choices[0].message.content.strip()

        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise ValueError("VisualPlanner returned invalid JSON")

        return json.loads(match.group())


# ============================================================
# LOW-LEVEL EXECUTOR
# ============================================================

class PyAutoGUIExecutor:

    def execute(self, action: Action):

        t = action.action_type
        p = action.params

        if t == ActionType.MOVE:
            pyautogui.moveTo(p["x"], p["y"])

        elif t == ActionType.CLICK:
            pyautogui.click(button=p.get("button", "left"))

        elif t == ActionType.DOUBLE_CLICK:
            pyautogui.doubleClick()

        elif t == ActionType.TYPE:
            pyautogui.write(p["text"], interval=0.03)

        elif t == ActionType.PRESS:
            keys = p["key"].split("+")
            if len(keys) == 1:
                pyautogui.press(keys[0])
            else:
                pyautogui.hotkey(*keys)

        elif t == ActionType.SCROLL:
            amount = p.get("amount", 3)
            clicks = amount if p.get("direction", "down") == "up" else -amount
            pyautogui.scroll(clicks)

        elif t == ActionType.WAIT:
            time.sleep(p.get("ms", 500) / 1000)

        elif t == ActionType.OPEN_APP:
            os.system(f'start {p["name"]}')

        elif t == ActionType.OPEN_URL:
            os.system(f'start {p["url"]}')

        elif t == ActionType.ZOOM:
            pyautogui.hotkey("ctrl", "+" if p.get("direction") == "in" else "-")


# ============================================================
# BASIC TASK PERFORMER (CLOSED LOOP)
# ============================================================

class BasicTaskPerformer:

    def __init__(self):
        self.detector = GUIDetector()
        self.planner = VisualPlanner()
        self.executor = PyAutoGUIExecutor()
        self.writer = ContentWriterClient()

    @staticmethod
    def _bbox_iou(a: dict[str, int], b: dict[str, int]) -> float:
        ax1, ay1 = a["x"], a["y"]
        ax2, ay2 = a["x"] + a["width"], a["y"] + a["height"]
        bx1, by1 = b["x"], b["y"]
        bx2, by2 = b["x"] + b["width"], b["y"] + b["height"]

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        a_area = max(0, a["width"]) * max(0, a["height"])
        b_area = max(0, b["width"]) * max(0, b["height"])
        union = a_area + b_area - inter_area

        return (inter_area / union) if union > 0 else 0.0

    def _summarize_ui_change(self, before: list[dict], after: list[dict]) -> dict[str, Any]:
        used_after: set[int] = set()
        matched_pairs: list[tuple[dict, dict]] = []
        appeared: list[str] = []
        disappeared: list[str] = []
        moved: list[str] = []

        for b in before:
            best_idx = None
            best_iou = 0.0
            for idx, a in enumerate(after):
                if idx in used_after:
                    continue
                if a.get("label") != b.get("label"):
                    continue
                iou = self._bbox_iou(b["bbox"], a["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx

            if best_idx is not None and best_iou >= 0.5:
                used_after.add(best_idx)
                matched_pairs.append((b, after[best_idx]))
            else:
                disappeared.append(str(b.get("label", "unknown")))

        for idx, a in enumerate(after):
            if idx not in used_after:
                appeared.append(str(a.get("label", "unknown")))

        for b, a in matched_pairs:
            if self._bbox_iou(b["bbox"], a["bbox"]) < 0.85:
                moved.append(str(a.get("label", "unknown")))

        changed = bool(appeared or disappeared or moved)

        return {
            "changed": changed,
            "appeared": appeared[:10],
            "disappeared": disappeared[:10],
            "moved": moved[:10],
            "before_count": len(before),
            "after_count": len(after)
        }

    def run_task(self, task: HighLevelTask) -> ExecutionResult:

        all_actions: list[Action] = []

        try:
            for step in task.steps:
                step_actions = self._run_step_loop(step)
                all_actions.extend(step_actions)

            return ExecutionResult(
                task_id=task.task_id,
                status="success",
                actions=all_actions
            )

        except Exception as e:
            return ExecutionResult(
                task_id=task.task_id,
                status="error",
                actions=all_actions,
                error=str(e)
            )

    def _run_step_loop(self, step: ExecutionStep):

        max_iterations = 10
        iteration = 0
        executed_actions: list[Action] = []
        execution_history: list[dict[str, Any]] = []
        previous_actions: list[dict[str, Any]] | None = None
        consecutive_repeat_count = 0
        no_change_streak = 0

        while iteration < max_iterations:
            iteration += 1

            detections = self.detector.detect()

            step_description = step.description

            if step.generate_text:
                text = self.writer.generate_text(
                    step.text_task_description,
                    step.text_context
                )
                step_description += f"\nText to use:\n{text}"

            planner_state = (
                step.text_context.copy()
                if isinstance(step.text_context, dict)
                else {}
            )
            planner_state.update({
                "iteration": iteration,
                "previous_actions": previous_actions or [],
                "consecutive_repeat_count": consecutive_repeat_count,
                "no_change_streak": no_change_streak,
                "execution_history": execution_history[-10:]
            })

            plan = self.planner.plan(
                step_description=step_description,
                detections=detections,
                state=planner_state
            )

            actions = plan.get("actions", [])
            status = plan.get("status")

            if previous_actions is not None and actions == previous_actions and status == "in_progress":
                consecutive_repeat_count += 1
            else:
                consecutive_repeat_count = 0

            if consecutive_repeat_count >= 3:
                raise Exception("Planner produced the same in-progress action plan repeatedly.")

            # Execute actions
            for a in actions:
                before_action_detections = detections
                action = Action(
                    action_type=ActionType[a["action_type"]],
                    params=a.get("params", {}),
                    description=a.get("description", "")
                )

                self.executor.execute(action)
                executed_actions.append(action)

                # Short pause for UI update
                time.sleep(0.5)

                # Refresh detections after each action
                detections = self.detector.detect()
                ui_change = self._summarize_ui_change(before_action_detections, detections)

                if action.action_type in {
                    ActionType.CLICK,
                    ActionType.DOUBLE_CLICK,
                    ActionType.TYPE,
                    ActionType.PRESS,
                    ActionType.SCROLL,
                    ActionType.OPEN_APP,
                    ActionType.OPEN_URL,
                }:
                    if ui_change["changed"]:
                        no_change_streak = 0
                    else:
                        no_change_streak += 1

                execution_history.append({
                    "iteration": iteration,
                    "action_type": action.action_type.value,
                    "params": action.params,
                    "description": action.description,
                    "ui_change": ui_change
                })

                if no_change_streak >= 4:
                    raise Exception("No meaningful UI change after multiple interactive actions.")

            previous_actions = actions

            if status == "done":
                return executed_actions

            if status == "error":
                # Allow internal retries before escalating
                if iteration < max_iterations:
                    continue
                else:
                    raise Exception(plan.get("message", "Planner returned error status."))

            if status != "in_progress":
                raise Exception(f"Planner returned unknown status: {status!r}")

        raise Exception("Max iterations reached for step")
