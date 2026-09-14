"""Small local-tool protocol for the bounded planner Agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


ToolRunner = Callable[[dict[str, Any]], dict[str, Any]]
TOOL_STATUSES = {"PASS", "FAIL", "UNCERTAIN", "BLOCKED"}


def tool_result(
    tool: str, status: str, *, observations: dict[str, Any] | None = None,
    artifacts: dict[str, Any] | None = None, warnings: list[str] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    if status not in TOOL_STATUSES:
        raise ValueError(f"Invalid planner tool status: {status}")
    return {
        "tool": tool, "status": status, "observations": observations or {},
        "artifacts": artifacts or {}, "warnings": warnings or [], "error": error,
    }


@dataclass(frozen=True)
class PlannerAgentTool:
    name: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    run: ToolRunner
    requires_gpu: bool = False
    retry_on_failure: bool = False
    requires_generation_unlock: bool = False


class PlannerToolRegistry:
    def __init__(self, tools: list[PlannerAgentTool] | None = None) -> None:
        self._tools = {tool.name: tool for tool in tools or []}

    def register(self, tool: PlannerAgentTool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Duplicate planner tool: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> PlannerAgentTool:
        try:
            return self._tools[name]
        except KeyError as error:
            raise KeyError(f"Unknown planner tool: {name}") from error

    def execute(self, name: str, state: dict[str, Any]) -> dict[str, Any]:
        tool = self.get(name)
        if tool.requires_generation_unlock and not state.get("generation_unlocked"):
            return tool_result(name, "BLOCKED", error="generation_hard_gates_not_unlocked")
        try:
            result = tool.run(state)
            return tool_result(
                name, str(result.get("status", "FAIL")),
                observations=dict(result.get("observations") or {}),
                artifacts=dict(result.get("artifacts") or {}),
                warnings=list(result.get("warnings") or []),
                error=result.get("error"),
            )
        except Exception as error:  # Tool failures are observations for replanning.
            return tool_result(name, "FAIL", error=f"{type(error).__name__}: {error}")

    def allows_retry(self, name: str) -> bool:
        return self.get(name).retry_on_failure

