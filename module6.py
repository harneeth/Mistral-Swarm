from typing import Dict, Any
from module4 import HighLevelTask, ExecutionStep, ActionType


class Overseer:

    def __init__(self, module1, module2, module3=None, module4=None, module5=None):
        self.m1 = module1
        self.m2 = module2
        self.m3 = module3
        self.m4 = module4

        self.state: Dict[str, Any] = {}
        self.external_knowledge: Dict[str, Any] = {}
        self.max_loops = 10

        self.current_step_index = 0
        self.current_plan = None

    def run(self, user_input: str):

        print("\n[Module 6] Starting new run...\n")

        self.state = self.m1.initialize_run(user_input)

        if self.state.get("missing_information"):
            for item in self.state["missing_information"]:
                answer = self.m1.request_user_input(f"Provide {item}: ")
                self.state["entities"][item] = answer

        loop_counter = 0

        while loop_counter < self.max_loops:
            loop_counter += 1
            print(f"\n[Module 6] Planning cycle {loop_counter}")

            plan = self.m2.generate_plan(
                state=self.state,
                external_knowledge=self.external_knowledge
            )

            if plan.get("requires_info_module"):
                if not self.m3:
                    print("No Module 3 available.")
                    return
                info = self.m3.get_information(self.state)
                self.external_knowledge.update(info)
                continue

            if self.current_plan is None:
                self.current_plan = plan
                self.current_step_index = 0

            steps = self.current_plan["steps"]

            if self.current_step_index >= len(steps):
                print("\n[Module 6] Task completed successfully.")
                return

            single_step_plan = {
                "goal": plan.get("goal"),
                "steps": [steps[self.current_step_index]]
            }

            task = self._convert_plan_to_task(single_step_plan)

            result = self.m4.run_task(task)

            if result.status == "success":
                self.current_step_index += 1
                continue

            if result.status == "success":
                print("\n[Module 6] Task completed successfully.")
                return

            if result.status == "error":

                print(f"\n[Module 6] Execution error: {result.error}")

                # Track failure
                self.state.setdefault("failure_counts", {})
                step_name = task.description

                self.state["failure_counts"][step_name] = \
                    self.state["failure_counts"].get(step_name, 0) + 1

                # If failed 3 times → escalate to Module 3
                if self.state["failure_counts"][step_name] >= 3:

                    print("[Module 6] Escalating to Module 3 for additional information...")

                    if self.m3:
                        info = self.m3.get_information(self.state)
                        self.external_knowledge.update(info)
                        self.state["failure_counts"][step_name] = 0
                        continue

                # Otherwise just replan
                continue

        print("\n[Module 6] Max planning cycles reached. Aborting.")

    def _convert_plan_to_task(self, plan: Dict[str, Any]) -> HighLevelTask:

        steps = []

        for idx, s in enumerate(plan["steps"], start=1):

            step = ExecutionStep(
                step_id=idx,
                description=s["description"],
                action_type=self._map_step_to_action_type(s["step"]),
                target=None,
                params={},
                generate_text=s["requires_writing_module"],
                text_task_description=s["description"],
                text_context=self.state
            )

            steps.append(step)

        return HighLevelTask(
            task_id="task_001",
            description=self.state.get("goal", "Task"),
            steps=steps,
            context=self.state
        )

    def _map_step_to_action_type(self, step_name: str) -> ActionType:
        """
        Maps logical task names to ActionType.
        """

        mapping = {
            "open_email_client": ActionType.OPEN_APP,
            "compose_email": ActionType.TYPE,
            "attach_file": ActionType.OPEN_URL,
            "send_email": ActionType.PRESS
        }

        return mapping.get(step_name, ActionType.WAIT)