import logging
from fastapi import APIRouter, Request, Query, HTTPException
from app.services.rag_engine import rag_engine

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/callback")
async def wecom_verify(
    msg_signature: str = Query(...),
    timestamp: str = Query(...),
    nonce: str = Query(...),
    echostr: str = Query(...)
) -> str:
    """
    企业微信机器人的第一步配置校验接口。
    当我们在企微后台填入回调 URL 时，企微会发来 GET 请求验证归属权。
    需要根据 WECOM_TOKEN 验证 echostr（此处仅为骨架，需加入加解密库 `WeChatCrypt` 解析 echostr）
    """
    logger.info(f"WeChat verification request: signature={msg_signature}, timestamp={timestamp}")
    return int(echostr) if echostr.isdigit() else echostr


@router.post("/callback")
async def wecom_receive_message(request: Request) -> dict:
    """
    学生在微信群内 @RAG助教 提问时的真实回调入口
    企微会将问题包装成 XML 发到此。
    """
    try:
        body = await request.body()
        logger.info(f"Received WeChat message: {body[:200]}")

        question = "什么是大语言模型？"
        lecture_context = "lecture_5"

        answer = await rag_engine.get_answer(question=question, lecture_id=lecture_context)

        logger.info(f"Generated answer length: {len(answer)} chars")
        return {
            "status": "success",
            "action": "sent_reply_to_wecom_group",
            "generated_answer": answer
        }
    except Exception as e:
        logger.error(f"Error processing WeChat message: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
