from fastapi import Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from app.core.config import settings

# --- Configuration (Must match smart-class-backend/auth.py) ---
SECRET_KEY = "your-secret-key-should-be-changed-in-production"
ALGORITHM = "HS256"

# Note: In a true microservice architecture, these configs might be sourced from environment variables.
# But for now we match the hardcoded values in smart-class-backend.

# OAuth2 scheme extracts token from the Authorization header (Bearer token)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)

async def verify_jwt(request: Request, token: str = Depends(oauth2_scheme)):
    """
    Stateless JWT verification.
    Ensures that the request comes from an authenticated user of the smart-class system.
    """
    # 0. 本地开发模式：localhost 请求直接放行，方便学生本地调试
    client_ip = request.client.host if request.client else ""
    if client_ip in ("127.0.0.1", "::1", "localhost"):
        return {"sub": "local_dev", "role": "admin"}

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    # 1. 如果带有内部系统的网关令牌，直接放行 (针对 smart-class-backend 的后端调用)
    if request.headers.get("X-AIIA-Gateway-Auth") == settings.GATEWAY_AUTH_TOKEN:
        return {"sub": "system", "role": "admin"}
        
    # 2. 否则校验 JWT Token (针对前端直接调用管理台)
    if not token:
        raise credentials_exception
        
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        
        # Only teachers/admins should access RAG tutor admin functionalities
        role: str = payload.get("role")
        if role not in ["teacher", "admin"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Administrator or Teacher privileges required to manage AI resources."
            )
            
        return payload
    except JWTError:
        raise credentials_exception
