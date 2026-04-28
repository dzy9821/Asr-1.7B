"""
WebSocket 端点 /tuling/asr/v3 —— 核心处理逻辑。

处理流程：
  1. 连接 → 检查并发上限 → 启动握手超时
  2. status=0 → 校验 → 初始化会话 → 回复握手成功
  3. status=1 → 解码音频 → VAD → 若触发断句 → ASR → ITN → 推送结果
  4. status=2 → 刷空缓冲区 → 推送终态结果 → 断开

每个 segment 响应的 header.message 包含 JSON 格式的耗时信息：
  {"asr_ms": 123.4, "total_ms": 145.6}
"""

from __future__ import annotations

import asyncio
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from uvicorn.protocols.utils import ClientDisconnected

from src.api.connection_manager import connection_manager
from src.api.metrics import (
    asr_connections_current,
    asr_errors_total,
    asr_processing_latency_ms,
    asr_segments_total,
)
from src.api.session import ASRSession
from src.core.config import settings
from src.core.logging import get_logger, trace_id_var
from src.models.schemas import (
    CWItem,
    ClientMessage,
    ResponseHeader,
    ResponsePayloadWrapper,
    ResultPayload,
    ServerMessage,
    WSItem,
)
from src.services.asr_service import ASRService, build_hotword_context
from src.services.itn_pool import ITNPool
from src.services.vad_service import vad_processor
from src.utils.audio import decode_base64_pcm, samples_to_cs, samples_to_ms

logger = get_logger(__name__)

router = APIRouter()

# 服务实例（由 main.py 生命周期管理器初始化）
asr_service: ASRService = ASRService()
itn_pool: ITNPool = ITNPool()


@router.websocket("/tuling/asr/v3")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """ASR 实时流式转录 WebSocket 端点。"""

    # ---- 并发控制 ----
    if not connection_manager.try_acquire():
        await websocket.close(code=1013, reason="Try Again Later")
        logger.warning("Connection rejected: max connections reached")
        return

    await websocket.accept()
    session = None
    connection_slot_released = False

    try:
        # ---- 握手阶段（带超时） ----
        session = await _handle_handshake(websocket)
        if session is None:
            await _wait_for_client_disconnect(websocket)

        # 注册连接
        connection_manager.register(session.sid, session.trace_id)
        asr_connections_current.inc()
        trace_id_var.set(session.trace_id)

        # 回复握手成功
        await _send_response(websocket, session, status=0, seg_id=0)
        session.set_streaming()
        logger.info("Handshake OK: sid=%s, biz_id=%s", session.sid, session.biz_id)

        # ---- 流式处理循环 ----
        while True:
            raw = await websocket.receive_text()
            msg = ClientMessage.model_validate_json(raw)

            if msg.header.status == 1:
                await _handle_audio_frame(websocket, session, msg)

            elif msg.header.status == 2:
                await _handle_end_frame(websocket, session)
                await _wait_for_client_disconnect(websocket, session)

    except WebSocketDisconnect:
        logger.info("Client disconnected: sid=%s", session.sid if session else "?")
    except asyncio.TimeoutError:
        logger.warning("Handshake timeout")
        asr_errors_total.labels(error_type="handshake_timeout").inc()
        await _wait_for_client_disconnect_safely(websocket, session)
    except Exception as exc:
        logger.exception("Unexpected error: %s", exc)
        asr_errors_total.labels(error_type="internal").inc()
        try:
            await _send_error(websocket, session, str(exc))
        except Exception:
            pass
        await _wait_for_client_disconnect_safely(websocket, session)
    finally:
        if session:
            session.close()  # 从 VAD 批处理器注销
            connection_manager.unregister(session.sid)
            connection_slot_released = True
            asr_connections_current.dec()
        elif not connection_slot_released:
            connection_manager.release_slot()
            connection_slot_released = True


# ============================================================
# 内部处理函数
# ============================================================


async def _handle_handshake(websocket: WebSocket) -> ASRSession | None:
    """等待并处理握手帧；除并发上限外不主动关闭连接。"""
    try:
        raw = await asyncio.wait_for(
            websocket.receive_text(),
            timeout=settings.HANDSHAKE_TIMEOUT,
        )
    except asyncio.TimeoutError:
        raise

    msg = ClientMessage.model_validate_json(raw)
    if msg.header.status != 0:
        logger.warning("First message must be handshake (status=0)")
        return None

    session = ASRSession(
        trace_id=msg.header.traceId,
        biz_id=msg.header.bizId,
        app_id=msg.header.appId or "",
    )

    # 追加客户端热词（与环境变量默认热词合并）
    if msg.payload and msg.payload.text and msg.payload.text.text:
        client_ctx = build_hotword_context(msg.payload.text.text)
        if client_ctx:
            base = build_hotword_context(settings.HOTWORDS)
            session.hotword_context = f"{base}\n{client_ctx}" if base else client_ctx

    return session


async def _handle_audio_frame(
    websocket: WebSocket,
    session: ASRSession,
    msg: ClientMessage,
) -> None:
    """处理音频数据帧。"""
    if not msg.payload or not msg.payload.audio:
        return

    # 追加客户端热词（与环境变量默认热词合并）
    if msg.payload.text and msg.payload.text.text:
        client_ctx = build_hotword_context(msg.payload.text.text)
        if client_ctx:
            base = build_hotword_context(settings.HOTWORDS)
            session.hotword_context = f"{base}\n{client_ctx}" if base else client_ctx

    # Base64 解码
    pcm_int16 = decode_base64_pcm(msg.payload.audio.audio)

    # 喂入 VAD（通过全局批处理器异步推理）
    segments = await session.vad.feed_audio(pcm_int16)

    # 对每个触发的语音段执行 ASR + ITN
    for seg in segments:
        await _process_segment(websocket, session, seg)


async def _handle_end_frame(websocket: WebSocket, session: ASRSession) -> None:
    """处理结束帧：刷空 VAD 缓冲区并推送终态结果。"""
    session.set_closing()

    # 强制刷出残余音频
    seg = session.vad.flush()
    if seg is not None:
        await _process_segment(websocket, session, seg, is_final=True)
    else:
        # 没有残余音频，发送纯终态信号（复用最后一个 seg_id，不递增）
        last_seg_id = max(0, session.seg_id - 1)
        await _send_response(websocket, session, status=2, seg_id=last_seg_id)


async def _wait_for_client_disconnect(
    websocket: WebSocket,
    session: ASRSession | None = None,
) -> None:
    """最终响应发出后保持连接打开，等待客户端主动关闭。"""
    logger.info(
        "Waiting for client close: sid=%s",
        session.sid if session else "?",
    )
    while True:
        await websocket.receive_text()


async def _wait_for_client_disconnect_safely(
    websocket: WebSocket,
    session: ASRSession | None = None,
) -> None:
    """等待客户端关闭；吞掉断开异常，避免覆盖原始处理分支。"""
    try:
        await _wait_for_client_disconnect(websocket, session)
    except (WebSocketDisconnect, RuntimeError):
        logger.info("Client disconnected: sid=%s", session.sid if session else "?")


async def _process_segment(
    websocket: WebSocket,
    session: ASRSession,
    seg: dict,
    is_final: bool = False,
) -> None:
    """对一个语音段执行 ASR → ITN → 推送结果。"""
    import json as _json

    t0 = time.monotonic()
    seg_id = session.next_seg_id()

    audio_int16 = seg["audio"]
    start_sample = seg["start_sample"]
    end_sample = seg["end_sample"]

    try:
        # ASR 推理（单独计时）
        t_asr_start = time.monotonic()
        raw_text = await asr_service.recognize(
            audio_int16,
            sr=16000,
            context=session.hotword_context,
        )
        t_asr_end = time.monotonic()
        asr_ms = (t_asr_end - t_asr_start) * 1000

        # ITN 后处理（通过多进程池）
        final_text = await itn_pool.normalize(raw_text)

        total_ms = (time.monotonic() - t0) * 1000

        # 构建结果并推送
        bg_ms = samples_to_ms(start_sample)
        ed_ms = samples_to_ms(end_sample)
        bg_cs = samples_to_cs(start_sample)
        ed_cs = samples_to_cs(end_sample)

        ws_item = WSItem(
            bg=bg_cs,
            cw=[
                CWItem(
                    w=final_text,
                    wp="n",
                    wb=bg_cs,
                    we=ed_cs,
                )
            ],
        )

        result = ResultPayload(
            segId=seg_id,
            bg=bg_ms,
            ed=ed_ms,
            msgtype="sentence",
            ws=[ws_item],
        )

        resp_status = 2 if is_final else 1

        # 在 header.message 中附带耗时 JSON，客户端可解析
        timing_msg = _json.dumps({
            "asr_ms": round(asr_ms, 1),
            "total_ms": round(total_ms, 1),
        })

        response = ServerMessage(
            header=ResponseHeader(
                code=0,
                message=timing_msg,
                sid=session.sid,
                traceId=session.trace_id,
                status=resp_status,
            ),
            payload=ResponsePayloadWrapper(result=result),
        )

        await websocket.send_text(response.model_dump_json())

        asr_processing_latency_ms.observe(total_ms)
        asr_segments_total.inc()

        logger.info(
            "Segment processed: seg_id=%d, text=%s, asr=%.0fms, total=%.0fms",
            seg_id,
            final_text,
            asr_ms,
            total_ms,
        )

    except (WebSocketDisconnect, ClientDisconnected):
        logger.info(
            "Client disconnected while sending segment: sid=%s, seg_id=%d",
            session.sid,
            seg_id,
        )
        raise
    except Exception as exc:
        logger.exception("Error processing segment %d: %s", seg_id, exc)
        asr_errors_total.labels(error_type="asr_inference").inc()
        await _send_error(websocket, session, f"ASR error: {exc}")


async def _send_response(
    websocket: WebSocket,
    session: ASRSession,
    status: int,
    seg_id: int,
    text: str = "",
) -> None:
    """发送一个简单的状态响应。"""
    result = ResultPayload(segId=seg_id, msgtype="sentence")
    if text:
        result.ws = [WSItem(cw=[CWItem(w=text)])]

    response = ServerMessage(
        header=ResponseHeader(
            code=0,
            message="success",
            sid=session.sid if session else "",
            traceId=session.trace_id if session else "",
            status=status,
        ),
        payload=ResponsePayloadWrapper(result=result),
    )
    await websocket.send_text(response.model_dump_json())


async def _send_error(
    websocket: WebSocket,
    session: ASRSession | None,
    detail: str,
) -> None:
    """推送错误消息。"""
    error_resp = {
        "header": {
            "code": -1,
            "message": detail,
            "sid": session.sid if session else "",
            "traceId": session.trace_id if session else "",
            "status": 2,
        },
    }
    import json

    await websocket.send_text(json.dumps(error_resp, ensure_ascii=False))
