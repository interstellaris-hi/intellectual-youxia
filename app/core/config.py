import os
import logging
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator

logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    PROJECT_NAME: str = "NCU RAG AI Tutor"
    API_V1_STR: str = "/api/v1"
    GATEWAY_AUTH_TOKEN: str = "AIIA_LAB_NCU_RAG_SEC_2026_xYz"

    WECOM_CORP_ID: str = "your_corp_id"
    WECOM_CORP_SECRET: str = "your_corp_secret"
    WECOM_TOKEN: str = "your_webhook_token"
    WECOM_ENCODING_AES_KEY: str = "your_encoding_aes_key"

    API: str = Field(default="", description="环境变量API密钥")

    LLM_API_BASE: str = "https://api.siliconflow.cn/v1"
    LLM_API_KEY: str = Field(default="")
    LLM_MODEL_NAME: str = "Qwen/Qwen2.5-7B-Instruct"

    LLM_EMBEDDING_API_BASE: str = "https://api.siliconflow.cn/v1"
    LLM_EMBEDDING_MODEL_NAME: str = "BAAI/bge-large-zh-v1.5"
    CHROMA_PERSIST_DIR: str = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "chroma")

    RERANKER_ENDPOINT: str = "https://api.siliconflow.cn/v1/rerank"
    RERANKER_MODEL_NAME: str = "BAAI/bge-reranker-v2-mini"

    VLM_API_BASE: str = "https://api.siliconflow.cn/v1"
    VLM_MODEL_NAME: str = "Qwen/Qwen2-VL-72B-Instruct"

    @field_validator("LLM_API_KEY", mode="before")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        import os
        api_val = os.environ.get("API", "")
        if api_val:
            return api_val
        return v if v else ""

    @field_validator("CHROMA_PERSIST_DIR", mode="before")
    @classmethod
    def validate_chroma_dir(cls, v: str) -> str:
        if v:
            os.makedirs(v, exist_ok=True)
            logger.info(f"Chroma persistence directory: {v}")
        return v

    @field_validator("LLM_API_BASE", "LLM_EMBEDDING_API_BASE", "VLM_API_BASE", mode="before")
    @classmethod
    def validate_api_url(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            logger.warning(f"Invalid API URL format: {v}, adding http://")
            return f"http://{v}"
        return v

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"

def get_settings() -> Settings:
    settings = Settings()
    logger.info(f"Project: {settings.PROJECT_NAME}")
    logger.info(f"LLM API: {settings.LLM_API_BASE}")
    return settings

settings = get_settings()
