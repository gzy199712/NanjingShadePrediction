"""Stable error codes without exposing Python tracebacks."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


def classify_message(message: str) -> tuple[str, int, str]:
    lowered = message.lower()
    if "hour" in lowered:
        return "INVALID_HOUR", 422, "时间必须位于06:00—18:00。"
    if "mode" in lowered:
        return "INVALID_MODE", 422, "交通模式无效。"
    if "objective" in lowered:
        return "INVALID_OBJECTIVE", 422, "路线优化目标无效。"
    if "snap pair" in lowered or "within 100 m" in lowered:
        return "SNAP_DISTANCE_EXCEEDED", 422, "起点或终点距可用路网超过100米，或不在同一连通分量。"
    if "no path" in lowered or "no_path" in lowered:
        return "NO_PATH", 422, "未找到满足正式约束的路径。"
    if "detour" in lowered:
        return "DETOUR_LIMIT_UNSATISFIED", 422, "无法在指定最大绕行率内找到该路线。"
    if "export" in lowered or "drawable" in lowered:
        return "EXPORT_FAILED", 500, "路线导出失败。"
    if "schema" in lowered:
        return "INVALID_SCHEMA", 422, "请求字段不符合正式接口规范。"
    return "INTERNAL_TOOL_ERROR", 500, "本地确定性工具调用失败。"


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        errors = exc.errors()
        code = "INVALID_SCHEMA"
        for error in errors:
            location = ".".join(str(value) for value in error.get("loc", []))
            if "hour" in location:
                code = "INVALID_HOUR"
            elif "mode" in location:
                code = "INVALID_MODE"
            elif "objective" in location:
                code = "INVALID_OBJECTIVE"
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": code,
                    "message": "输入不符合应用API规范。",
                    "request_valid": False,
                }
            },
        )

    @app.exception_handler(Exception)
    async def internal_error(request: Request, exc: Exception) -> JSONResponse:
        code, status, message = classify_message(str(exc))
        return JSONResponse(
            status_code=status,
            content={"error": {"code": code, "message": message}},
        )

