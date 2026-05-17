import sys
import os
import warnings
warnings.filterwarnings("ignore") # 忽略一些不必要的 huggingface 警告

# 将上层目录加入系统路径以便引用内网配置
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from app.core.config import settings

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter, Language
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

def ingest_master_md(file_path: str, lecture_id: str):
    print(f"🚀 开始阅读教材并进行数据切片: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"❌ 找不到文件: {file_path}")
        return
        
    with open(file_path, 'r', encoding='utf-8') as f:
        md_text = f.read()
        
    # 定义要分离的 Markdown 标题层级
    headers_to_split_on = [
        ("#", "H1"),
        ("##", "H2"),
        ("###", "H3"),
    ]
    
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False # 保留结构信息
    )
    
    md_header_splits = markdown_splitter.split_text(md_text)
    
    # ==== 引入成熟的 Recursive 层级切分机制以解决生僻技术短语的 Embedding Dilution (Token稀释) 问题 ====
    # 使用 Markdown AST 语义树切片，保留完整的推导段与代码块
    print(f"🧩 正在执行第二级语义化切分 (Markdown Semantic Elastic Chunking)...")
    text_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.MARKDOWN,
        chunk_size=1000,
        chunk_overlap=150
    )
    final_splits = text_splitter.split_documents(md_header_splits)
    
    # ==== 强悍的 Namespace 命名空间注入 ====
    for split in final_splits:
        # 给每个切片打上烙印，属于哪一讲的知识，不能串门
        split.metadata["lecture_id"] = lecture_id
        split.metadata["source"] = os.path.basename(file_path)
        
    print(f"✂️ 成功将 `.master.md` 切分出 {len(final_splits)} 个带高级重叠上下文的独立微型碎片！")
    
    # ==== 使用本机的 Qwen2-7B-instruct GPU 向量化 ====
    print(f"🧠 正在请求远端加速生成大维度语义向量 ({settings.LLM_EMBEDDING_MODEL_NAME})...")
    embeddings = OpenAIEmbeddings(
        openai_api_base=settings.LLM_EMBEDDING_API_BASE,
        openai_api_key=settings.LLM_API_KEY,
        model=settings.LLM_EMBEDDING_MODEL_NAME
    )
    
    # ==== 写入本地 ChromaDB ====
    os.makedirs(settings.CHROMA_PERSIST_DIR, exist_ok=True)
    print(f"💾 开始写入本地向量数据库: {settings.CHROMA_PERSIST_DIR}")

    vector_store = Chroma(
        persist_directory=settings.CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"}
    )

    # [清道夫逻辑] 在全量插入之前，先通过命名空间揪出旧日支配者并全数抹除，严防数据多重影分身
    try:
        existing = vector_store.get(where={"lecture_id": lecture_id})
        if existing and existing["ids"]:
            vector_store.delete(ids=existing["ids"])
            print(f"🧹 已扫描并强制清理同一命名空间 ({lecture_id}) 的旧版数据残留 ({len(existing['ids'])} 条)！")
    except Exception as e:
        print(f"⚠️ 无法扫描清理旧残留向量数据 (首次写入可忽略该提示): {e}")

    vector_store.add_documents(documents=final_splits)
    
    print("✅ 注入完成！您的 RAG Tutor 大脑已经掌握了这一讲的内容。")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python ingest.py <path_to_md_file> <lecture_id_namespace>")
        print("Example: python ingest.py ../test/第一讲_起源_讲义.master.md lecture_1")
        sys.exit(1)
    
    ingest_master_md(sys.argv[1], sys.argv[2])
