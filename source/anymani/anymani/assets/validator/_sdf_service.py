r"""单 CUDA context 的 post-mutate SDF validator actor 与 worker client。

CPU generation workers 通过 spawn-context request queue 提交完整 hand/config；每个请求携带
独立 one-way Pipe，因此响应不会被其他 worker 消费。GPU actor 在短时间窗内收集最多
``batch_size`` 个请求，统一调用 ragged micro-batch。actor 一旦出现异常会进入 fatal 状态，
后续请求只返回同一错误；worker 不允许静默回到本地 CUDA 或 CPU 路线。
"""

from __future__ import annotations

import multiprocessing as mp
import queue
import time
import traceback
from dataclasses import dataclass
from multiprocessing.connection import Connection
from typing import Any
from uuid import uuid4

from ..asset_base import HandCfg
from ._sdf_batch import evaluate_finger_sdf_clearance_batch
from ._sdf_clearance import SdfClearanceConfig, SdfClearanceResult, evaluate_finger_sdf_clearance

_STOP = "__ANYMANI_SDF_SERVICE_STOP__"
_WORKER_REQUEST_QUEUE: Any | None = None
_RESPONSE_TIMEOUT_SECONDS = 300.0


class CentralSdfServiceError(RuntimeError):
    r"""中央 GPU validator 启动、通信或数值执行失败。

    mother worker 用这一专用异常区分“候选本身未通过 validator”和“validator 服务已失去
    科研可信度”。前者是正常 rejection sampling，后者必须穿透 worker safe-report 边界，
    立即使 dataset build 失败，禁止退回本地 backend 后继续混合生成。
    """


@dataclass
class _SdfServiceRequest:
    r"""一只 candidate hand 的 GPU clearance 请求与独占响应 Pipe。"""

    request_id: str
    hand: HandCfg
    config: SdfClearanceConfig
    response_connection: Connection


def configure_worker_sdf_service(request_queue: Any | None) -> None:
    r"""在 CPU worker initializer 中绑定或清除中央 GPU service queue。"""

    global _WORKER_REQUEST_QUEUE
    _WORKER_REQUEST_QUEUE = request_queue


def evaluate_finger_sdf_clearance_routed(
    hand: HandCfg,
    config: SdfClearanceConfig,
) -> SdfClearanceResult:
    r"""有 central service 时远程验证，否则保持现有 scalar 本地行为。"""

    if _WORKER_REQUEST_QUEUE is None:
        return evaluate_finger_sdf_clearance(hand, config)
    context = mp.get_context("spawn")
    receive_connection, send_connection = context.Pipe(duplex=False)
    request = _SdfServiceRequest(
        request_id=uuid4().hex,
        hand=hand,
        config=config,
        response_connection=send_connection,
    )
    try:
        _WORKER_REQUEST_QUEUE.put(request)
        # 300 s 是服务级失效上限，不是单 kernel 性能目标。正常 micro-batch 应在秒级返回；
        # 有限等待保证 actor crash、Pipe 句柄损坏或 CUDA driver 卡死不会让 build 永久挂起。
        if not receive_connection.poll(_RESPONSE_TIMEOUT_SECONDS):
            raise CentralSdfServiceError(
                f"central GPU SDF service timed out after {_RESPONSE_TIMEOUT_SECONDS:.0f} seconds"
            )
        response = receive_connection.recv()
    finally:
        receive_connection.close()
        send_connection.close()
    if not isinstance(response, dict) or response.get("request_id") != request.request_id:
        raise CentralSdfServiceError("central GPU SDF service returned an invalid response")
    if not bool(response.get("ok")):
        raise CentralSdfServiceError(f"central GPU SDF service failed: {response.get('error', 'unknown error')}")
    result = response.get("result")
    if not isinstance(result, SdfClearanceResult):
        raise CentralSdfServiceError("central GPU SDF service returned an invalid clearance result")
    return result


def run_sdf_service(
    request_queue: Any,
    startup_connection: Connection,
    *,
    batch_size: int,
    batch_window_ms: float = 5.0,
) -> None:
    r"""初始化唯一 CUDA owner，并持续处理 micro-batches 直到收到 stop sentinel。"""

    if batch_size < 1:
        raise ValueError("central GPU SDF batch_size must be >= 1")
    if batch_window_ms < 0.0:
        raise ValueError("central GPU SDF batch_window_ms must be non-negative")
    fatal_error: str | None = None
    startup_sent = False
    try:
        import torch
        import warp as wp

        torch.cuda.init()
        wp.init()
        if not torch.cuda.is_available() or not wp.is_cuda_available():
            raise RuntimeError("central GPU SDF service requires PyTorch and Warp CUDA")
        startup_connection.send({"ok": True, "pid": mp.current_process().pid})
        startup_connection.close()
        startup_sent = True
        while True:
            first = request_queue.get()
            if first == _STOP:
                return
            requests = [_require_request(first)]
            deadline = time.monotonic() + batch_window_ms / 1000.0
            while len(requests) < batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                try:
                    item = request_queue.get(timeout=remaining)
                except queue.Empty:
                    break
                if item == _STOP:
                    _respond_fatal(requests, "central GPU SDF service stopped with requests in flight")
                    return
                requests.append(_require_request(item))
            if fatal_error is not None:
                _respond_fatal(requests, fatal_error)
                continue
            try:
                results = evaluate_finger_sdf_clearance_batch(
                    [request.hand for request in requests],
                    [request.config for request in requests],
                )
                for request, result in zip(requests, results):
                    request.response_connection.send(
                        {"request_id": request.request_id, "ok": True, "result": result}
                    )
                    request.response_connection.close()
            except Exception as exc:
                fatal_error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
                _respond_fatal(requests, fatal_error)
    except Exception as exc:
        if not startup_sent:
            try:
                startup_connection.send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
            finally:
                startup_connection.close()
        raise


def stop_sdf_service(request_queue: Any) -> None:
    r"""向中央 GPU actor 发送有序停止 sentinel。"""

    request_queue.put(_STOP)


def _require_request(value: Any) -> _SdfServiceRequest:
    r"""拒绝 queue 中不属于本协议的对象。"""

    if not isinstance(value, _SdfServiceRequest):
        raise TypeError(f"central GPU SDF queue received {type(value).__name__}, expected request")
    return value


def _respond_fatal(requests: list[_SdfServiceRequest], error: str) -> None:
    r"""向一批等待 worker 返回相同 fatal error，并关闭 actor 持有的 Pipe 端。"""

    for request in requests:
        try:
            request.response_connection.send(
                {"request_id": request.request_id, "ok": False, "error": error}
            )
        finally:
            request.response_connection.close()


__all__ = [
    "CentralSdfServiceError",
    "configure_worker_sdf_service",
    "evaluate_finger_sdf_clearance_routed",
    "run_sdf_service",
    "stop_sdf_service",
]
