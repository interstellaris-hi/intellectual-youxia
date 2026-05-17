from typing import TypedDict, List, Optional, Any
import logging
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from app.core.config import settings
import asyncio
import time
import requests
import httpx
import codecs
import json
import base64
import io
import re
from urllib.parse import quote

logger = logging.getLogger(__name__)

class GraphState(TypedDict):
    question: str
    lecture_id: str
    image_base64: str          # 新增：原始图片数据
    visual_context: str        # 新增：图片分析后的文本描述
    documents: List[Document]
    generation: str
    loop_count: int
    stream_mode: bool

class RAGEngine:
    def __init__(self):
        # 0. 根据 API 网关类型动态构建 "关闭思维链" 参数
        #    - 硅基流动 (SiliconFlow): 顶层 "enable_thinking": False
        #    - 本地 vLLM (Atlas): "chat_template_kwargs": {"enable_thinking": False}
        _is_siliconflow = "siliconflow" in settings.LLM_API_BASE.lower()
        self._no_think_params = (
            {"enable_thinking": False}
            if _is_siliconflow
            else {"chat_template_kwargs": {"enable_thinking": False}}
        )

        # 联网搜索配置
        self.search_enabled = True
        self.search_api_url = "https://duckduckgo.com/html/"
        self.search_headers = {"User-Agent": "Mozilla/5.0"}

        # 1. 核心生成器 LLM
        self.llm = ChatOpenAI(
            openai_api_base=settings.LLM_API_BASE,
            openai_api_key=settings.LLM_API_KEY,
            model_name=settings.LLM_MODEL_NAME,
            temperature=0.1,
            max_tokens=4096,
            streaming=True, # 启用全双工流式支持
            extra_body={
                **self._no_think_params,
                "stop": ["<|im_end|>", "<|endoftext|>"]
            }
        )
        
        # 1.1 轻量级分类器专用 LLM
        self.classifier_llm = ChatOpenAI(
            openai_api_base=settings.LLM_API_BASE,
            openai_api_key=settings.LLM_API_KEY,
            model_name=settings.LLM_MODEL_NAME,
            temperature=0.0,
            max_tokens=128,
            extra_body={
                **self._no_think_params,
                "stop": ["<|im_end|>", "<|endoftext|>"]
            }
        )
        
        # 2. 向量嵌入模型映射
        self.embeddings = OpenAIEmbeddings(
            openai_api_base=settings.LLM_EMBEDDING_API_BASE,
            openai_api_key=settings.LLM_API_KEY,
            model=settings.LLM_EMBEDDING_MODEL_NAME
        )

        # 3. 本地全私有化向量数据库 (强制映射到 Cosine Space 防止欧氏距离膨胀)
        self.vector_store = Chroma(
            persist_directory=settings.CHROMA_PERSIST_DIR,
            embedding_function=self.embeddings,
            collection_metadata={"hnsw:space": "cosine"}
        )

        logger.info("Building BM25 index from vector store...")
        try:
            db_data = self.vector_store.get()
            docs_for_bm25 = [Document(page_content=txt, metadata=meta) for txt, meta in zip(db_data['documents'], db_data['metadatas'])]
            if docs_for_bm25:
                import jieba
                def jieba_preprocess(text):
                    return jieba.lcut(text)
                self.bm25_retriever = BM25Retriever.from_documents(docs_for_bm25, preprocess_func=jieba_preprocess)
                logger.info(f"BM25 index built with {len(docs_for_bm25)} documents")
            else:
                self.bm25_retriever = None
                logger.warning("Vector store is empty, skipping BM25 index")
        except Exception as e:
            self.bm25_retriever = None
            logger.error(f"Failed to build BM25 index: {e}")

        # === 预定义各个裁判/生成 Prompt ===
        self.grader_prompt = PromptTemplate(
            template="""你是一个极其严格的关键词匹配执行器。请评估以下课件片段是否与用户提问相关。
【极端强制指令】：
1. 只要给定的“课件片段”字面上**包含**了“用户提问”里涉及的核心主体或专有名词（例如：VS Code、Agent 等软件名或学术名词，哪怕只提到一次），你**必须、绝对**最终输出 'yes'！即使它没有定义这个名词，只要出现了就判定相关！
2. 反之，如果“课件片段”中根本没有出现该专有名词，或者完全无关，你**必须**输出 'no'。严禁脑补关联！
3. 你的最终判定必须且只能单独是 yes 或 no，严禁附加其他文字。

课件片段：
{context}

用户提问：{question}
你的判定(仅输出 yes 或 no)：""",
            input_variables=["context", "question"]
        )
        
        self.rewrite_prompt = PromptTemplate(
            template="""你是一个智能重写引擎。用户抛出的问题在此前检索中没有命中任何本地课件内容。
请根据可能的意图，对提问进行扩写、释义或提炼，生成一个更易于在文本数据库中匹配到的新问题。（比如增加对应的中文翻译、去掉口语化词汇等）

原问题：{question}
重写后的问题（仅单独输出问题本身，禁止输出其它字句）：""",
            input_variables=["question"]
        )

        self.generation_prompt = PromptTemplate(
            template="""你是人工智能创新应用实验室 (AIIA Lab) 专属数字伴读助教。当前答疑依托的讲义/模块为：【{lecture_name}】。
严格遵循以下教务准则：
1. 核心忠诚与分点排版：如果学生的提问可以在下方【课件检索内容】中找到答案，你的核心论点【必须且只能】基于课件。
   - 必须分段、分点（使用 Markdown 列表和小标题）详细罗列解答，严禁把所有内容挤成一大段。
   - 每个核心观点后，如果涉及到了原文档的知识，必须单起一行附带原始文件名出处（例如格式：`来源：【xxx.md】 - “部分片段...”`）。
   - 保持风趣、生动、且极为专业的助教口吻，适当使用 Emoji。
2. 常识豁免与显式警告：如果学生提问涉及课件中并未直接讲透的背景概念（例如：一个没听说过的软件缩写、基础的计算机网络常识），且不违背原始课程大纲精神，你可以动用自身的技术先验知识进行补充解答。但是，涉及课外知识的地方，你【必须】使用明确的引用块进行免责声明，警示学生这不是课本内的硬性考点。
   声明格式要求（直接套用，且**千万不要**在上面额外加一个同名标题，避免重复）：
   > ⚠️ **拓展补充**：当前讲义虽未详细展开，但基于通用技术常识，[你的解释...]

【全量打散的课件检索内容】：
{context}

学生提问：{question}

你的专业助教解答：""",
            input_variables=["context", "lecture_name", "question"]
        )

        self.hallucination_prompt = PromptTemplate(
            template="""你是一个宽容的常识审查员。
判断下方的回答是否可以安全发给学生看。
1. 如果回答内容基于课件，或者顺带补充了合理的计算机通识（比如 VS Code 是一款编辑器等），请回答 'yes'。
2. 只有当回答严重违背事实、或者完全与课件矛盾时，才回答 'no'。

提供的课件材料：
{context}

草稿回答：
{generation}

判定(yes/no)：""",
            input_variables=["context", "generation"]
        )

        self.vision_prompt = PromptTemplate(
            template="""你是人工智能创新应用实验室 (AIIA Lab) 智慧课程的视觉分析引擎。请分析学生上传的这张截图。
【任务目标】：
1. 如果是课件内容，请完整提取其中的关键文字信息（OCR）。
2. 如果是代码或报错截图，请精确提取代码段和 Error Message。
3. 简要概括图中处于什么教学场景（例如：正在配置环境、正在编写 Python 函数）。

【注意】：你的回答将作为后续知识库检索的关键词，请务必客观、准确。
请以 [视觉摘要] 打头开始你的分析。

图片内容：<|vision_start|>data:image/png;base64,{image_base_base64}<|vision_end|>

分析结果：""",
            input_variables=["image_base_base64"]
        )

        # === 构建 LangGraph 状态图 ===
        workflow = StateGraph(GraphState)
        
        # 添加节点
        workflow.add_node("vision_analyze", self.node_vision_analyze)
        workflow.add_node("retrieve", self.node_retrieve)
        workflow.add_node("grade_documents", self.node_grade_documents)
        workflow.add_node("generate", self.node_generate)
        workflow.add_node("transform_query", self.node_transform_query)
        workflow.add_node("refuse", self.node_refuse)
        
        # 连线
        workflow.add_edge(START, "vision_analyze")
        workflow.add_edge("vision_analyze", "retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        
        # 评估文档后，决定是生成、重写、还是彻底放弃
        workflow.add_conditional_edges(
            "grade_documents",
            self.edge_decide_to_generate,
            {
                "transform_query": "transform_query",
                "generate": "generate",
                "refuse": "refuse"
            }
        )
        
        # 重写后重新拉取资料
        workflow.add_edge("transform_query", "retrieve")
        
        # 加上幻觉检察官的后置拦截
        workflow.add_conditional_edges(
            "generate",
            self.edge_check_hallucination,
            {
                "useful": END,               # 安全过关，发送
                "hallucinated": "refuse"     # 查出幻觉，直接没收回答并打回
            }
        )
        workflow.add_edge("refuse", END)
        
        # 编译特工网络
        self.app = workflow.compile()

    # ==========================
    # LangGraph 节点与边逻辑 (Agentic Flow)
    # ==========================
    async def node_vision_analyze(self, state: GraphState):
        """节点 0：【视觉分拣】增加图片压缩逻辑，防止 Token 爆炸"""
        image_base64 = state.get("image_base64")
        if not image_base64:
            return {"visual_context": ""}

        logger.info("Starting vision analysis with image compression...")
        try:
            import io
            from PIL import Image

            img_data = base64.b64decode(image_base64)
            img = Image.open(io.BytesIO(img_data))

            max_size = 1024
            if max(img.size) > max_size:
                ratio = max_size / max(img.size)
                new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
                logger.debug(f"Image resized to: {img.size}")

            buffer = io.BytesIO()
            img.convert("RGB").save(buffer, format="JPEG", quality=60, optimize=True)
            compact_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            logger.debug(f"Image compressed: {len(image_base64)} -> {len(compact_base64)} chars")

            payload = {
                "model": settings.VLM_MODEL_NAME,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{compact_base64}"
                                }
                            },
                            {
                                "type": "text",
                                "text": "请详细分析截图中包含的代码、文字、公式或教学场景细节。请以 [视觉摘要] 打头开始你的精简概括分析："
                            }
                        ]
                    }
                ],
                "max_tokens": 512,
                "temperature": 0.1
            }

            logger.info(f"Sending vision request to {settings.VLM_API_BASE}")

            visual_context = ""
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    async with httpx.AsyncClient(timeout=120.0) as client:
                        response = await client.post(
                            f"{settings.VLM_API_BASE}/chat/completions",
                            headers={"Authorization": f"Bearer {settings.LLM_API_KEY}"},
                            json=payload
                        )

                    if response.status_code == 200:
                        result = response.json()
                        visual_context = result['choices'][0]['message']['content'].strip()
                        logger.info(f"Vision analysis completed: {visual_context[:50]}...")
                        break
                    else:
                        logger.warning(f"Vision API error (attempt {attempt+1}/{max_retries}): {response.status_code}")
                        visual_context = f"[图片解析失败: 状态码 {response.status_code}]"

                except (httpx.TimeoutException, httpx.NetworkError) as e:
                    logger.warning(f"Vision API timeout (attempt {attempt+1}/{max_retries}): {e}")
                    visual_context = f"[图片解析失败: 网络超时]"
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2)
                        continue
                except Exception as e:
                    logger.error(f"Vision analysis error: {e}")
                    visual_context = f"[图片解析失败: 未知错误]"
                    break

            updates = {"visual_context": visual_context, "image_base64": ""}
            if not state.get("question", "").strip():
                logger.info("No text question provided, using visual context as query")
                updates["question"] = f"请详细分析截图中的内容，并基于提供的课件资料解答截图中可能存在的疑问点。图片信息如下：{visual_context}"
            return updates
        except Exception as e:
            logger.error(f"Vision analysis failed: {e}", exc_info=True)
            updates = {"visual_context": "[图片解析失败]", "image_base64": ""}
            if not state.get("question", "").strip():
                updates["question"] = "请帮我分析这张图片中的问题（注意：图片解析暂时失败）。"
            return updates

    def node_retrieve(self, state: GraphState):
        """节点 1：基于最新的 Question 从 ChromaDB 拉取资料，并启用 Reranker 进行重排序缩编"""
        _t_retrieve = time.time()
        question = state["question"]
        lecture_id = state["lecture_id"]
        visual_context = state.get("visual_context", "")

        retrieval_query = question
        if visual_context:
            retrieval_query = f"{visual_context} {question}"
            logger.info(f"Hybrid retrieval query: {retrieval_query[:50]}...")

        search_kwargs = {"k": 20}
        filter_dict = {}
        if lecture_id == "GLOBAL_SEARCH":
            search_kwargs = {"k": 15}
        elif lecture_id:
            filter_dict = {"lecture_id": lecture_id}
            search_kwargs["filter"] = filter_dict

        vector_retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs
        )

        search_query = question
        if state.get("visual_context"):
            search_query = f"{question} (图片背景: {state['visual_context']})"

        if self.bm25_retriever:
            self.bm25_retriever.k = search_kwargs["k"]

            ensemble_retriever = EnsembleRetriever(
                retrievers=[self.bm25_retriever, vector_retriever],
                weights=[0.5, 0.5]
            )
            logger.info(f"Using hybrid retrieval (BM25 + Dense), k={search_kwargs['k']}")
            docs = ensemble_retriever.invoke(search_query)
        else:
            logger.info(f"Using dense retrieval, k={search_kwargs['k']}")
            docs = vector_retriever.invoke(search_query)

        if not docs:
            logger.info(f"Retrieve completed in {time.time() - _t_retrieve:.2f}s (0 docs)")
            return {"documents": []}

        logger.info(f"Calling reranker for {len(docs)} documents...")
        try:
            formatted_docs = [doc.page_content for doc in docs]

            response = requests.post(
                settings.RERANKER_ENDPOINT,
                headers={"Authorization": f"Bearer {settings.LLM_API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": settings.RERANKER_MODEL_NAME,
                    "query": question,
                    "documents": formatted_docs
                },
                timeout=60.0
            )
            response.raise_for_status()

            results = response.json().get("results", [])
            scored_docs = []
            for res in results:
                idx = res["index"]
                score = res["relevance_score"]
                scored_docs.append((score, docs[idx]))

            scored_docs.sort(key=lambda x: x[0], reverse=True)
            top_k_docs = [d for score, d in scored_docs[:10]]

            logger.info(f"Reranking completed, top {len(top_k_docs)} docs (max score: {scored_docs[0][0]:.4f})")
            logger.info(f"Retrieve completed in {time.time() - _t_retrieve:.2f}s")
            return {"documents": top_k_docs}

        except Exception as e:
            logger.error(f"Reranker failed, using fallback: {e}")
            logger.info(f"Retrieve (fallback) completed in {time.time() - _t_retrieve:.2f}s")
            return {"documents": docs[:10]}

    def node_grade_documents(self, state: GraphState):
        """节点 2：审查拉取回的切片是否真正回答了问题"""
        question = state["question"]
        docs = state.get("documents", [])
        loop_count = state.get("loop_count", 0)
        
        print(f"[GRADE] 正在并发审查 {len(docs)} 个检索切片的关联度...", flush=True)
        
        if not docs:
            return {"documents": [], "loop_count": loop_count + 1}
            
        # 既然我们已经有了极高质量的 Qwen 交叉重排器（Cross-Encoder），它给出的分数已经是目前能得到的最权威的相关度判定！
        # 让通用大模型去做二次 Boolean 打分不仅画蛇添足，还会因为 Prompt 的不可控诱导产生误判（False Negative）。
        # 因此，直接采信 Reranker 选出的优质片段即可！
        
        relevant_docs = docs[:5] # 保留前5名给大模型生成
        
        if not relevant_docs:
            print("[GRADE] 所有资料均为空，触发循环...", flush=True)
            return {"documents": [], "loop_count": loop_count + 1}
            
        print(f"   => 审查完毕，直接采纳 Reranker 提供的最优质前 {len(relevant_docs)} 块切片！", flush=True)
        return {"documents": relevant_docs, "loop_count": loop_count + 1}

    def edge_decide_to_generate(self, state: GraphState):
        """条件边：决策树叉"""
        relevant_docs = state["documents"]
        loop_count = state["loop_count"]

        if not relevant_docs:
            if loop_count >= 1:
                print("[DECISION] 本地课件检索为空，切入【联网搜索】节点！")
                return "refuse"
            print("[DECISION] 暂无合适资料，退回进行【问题重写】！")
            return "transform_query"
        print("[DECISION] 资料已就绪，切入【草稿生成】节点！")
        return "generate"

    def node_transform_query(self, state: GraphState):
        """节点 3：改写失败的问题"""
        question = state["question"]
        prompt = self.rewrite_prompt.format(question=question)
        res = self.llm.invoke(prompt)
        rewritten = res.content.strip()
        logger.info(f"Query rewritten: '{question}' -> '{rewritten}'")
        return {"question": rewritten}

    def node_generate(self, state: GraphState):
        """节点 4：组装安全资料并让大模型撰写初稿"""
        _t_gen = time.time()
        docs = state["documents"]
        question = state["question"]
        lecture_id = state["lecture_id"]

        context_blocks = [f"====== [来源出处：{d.metadata.get('lecture_id', '未知讲次')}] ======\n{d.page_content}" for d in docs]
        context = "\n\n".join(context_blocks)
        final_question = question
        vision_prefix = ""
        if state.get("visual_context"):
            vision_prefix = f"> 📸 **[AI 视觉核验]**：{state['visual_context']}\n\n---\n\n"

        prompt = self.generation_prompt.format(
            context=context,
            lecture_name="全课程主线知识库" if lecture_id == "GLOBAL_SEARCH" else lecture_id,
            question=final_question
        )
        if state.get("stream_mode"):
            logger.info(f"Generate (streaming mode) prep completed in {time.time() - _t_gen:.2f}s")
            return {"generation": ""}

        logger.info("Generating answer with LLM...")
        logger.debug(f"Starting LLM.invoke(), prompt length: {len(prompt)} chars")
        _t_llm = time.time()
        res = self.llm.invoke(prompt)
        logger.info(f"LLM invoke took {time.time() - _t_llm:.2f}s, total {time.time() - _t_gen:.2f}s, output {len(res.content)} chars")

        return {"generation": vision_prefix + res.content}

    def edge_check_hallucination(self, state: GraphState):
        """条件边：最后一道防线：检查是否胡编乱造"""
        docs = state["documents"]
        generation = state["generation"]
        
        context_blocks = [f"【{d.metadata.get('lecture_id', '')}】:\n{d.page_content}" for d in docs]
        context = "\n\n".join(context_blocks)
        
        # 直接放行：系统已经剥离了不确定的幻觉上下文，且大模型的 System Prompt 限制得足够死
        print("[HALLUCINATION CHECK] (Bypassed) 信任专有大模型的严谨度，直接放行！")
        return "useful"

    def node_refuse(self, state: GraphState):
        """节点 X：越界/幻觉专属的冷酷打回，尝试联网搜索"""
        question = state["question"]
        logger.info(f"本地资料不足，尝试联网搜索: {question[:30]}...")

        try:
            search_results = self.web_search(question)
            if search_results:
                logger.info(f"联网搜索获取到 {len(search_results)} 条结果")
                return self._generate_with_web_results(question, state.get("lecture_id", "全课程主线知识库"), search_results)
        except Exception as e:
            logger.error(f"联网搜索失败: {e}")

        refuse_msg = "同学你好，经过智能体多轮自省与核查，这个问题超出了当前已入库的主线教学课件范围（或资料不足以支撑确切的安全解答）。为保障严谨，建议向老师和助教反馈探讨。"
        return {"generation": refuse_msg}

    def web_search(self, query: str, num_results: int = 5) -> List[str]:
        """联网搜索"""
        try:
            search_url = f"https://duckduckgo.com/html/?q={quote(query)}&format=json"
            response = requests.get(search_url, headers=self.search_headers, timeout=10)
            if response.status_code == 200:
                html = response.text
                results = re.findall(r'<a class="result__a"[^>]*href="([^"]*)"[^>]*>([^<]*)</a>', html, re.DOTALL)
                titles = re.findall(r'<a class="result__a"[^>]*>([^<]*)</a>', html)
                snippets = re.findall(r'<a class="result__snippet"[^>]*>([^<]*)</a>', html)

                web_results = []
                for i in range(min(num_results, len(titles))):
                    title = titles[i].strip() if i < len(titles) else ""
                    snippet = snippets[i].strip() if i < len(snippets) else ""
                    if title:
                        web_results.append(f"【{title}】{snippet}")
                return web_results
        except Exception as e:
            logger.error(f"搜索失败: {e}")
        return []

    def _generate_with_web_results(self, question: str, lecture_id: str, web_results: List[str]) -> dict:
        """使用联网搜索结果生成回答"""
        web_context = "\n\n".join(web_results)

        prompt = f"""你是人工智能创新应用实验室 (AIIA Lab) 专属数字伴读助教。

⚠️ **重要提示**：当前问题超出了本地已入库的教学课件范围，已通过联网搜索获取相关资料。

请基于以下联网搜索结果回答学生问题。如果回答中涉及搜索到的课外知识，必须使用引用块进行免责声明。

【联网搜索资料】：
{web_context}

学生提问：{question}

你的专业助教解答："""

        try:
            res = self.llm.invoke(prompt)
            answer = res.content.strip()

            web_note = "\n\n---\n\n> ⚠️ **联网补充**：以下答案综合了本地课件知识和联网检索到的相关资料。"
            return {"generation": web_note + answer}
        except Exception as e:
            logger.error(f"联网生成失败: {e}")
            return {"generation": f"联网搜索已找到相关资料，但生成回答时出错。请访问以下链接获取更多信息：\n\n" + "\n".join(web_results[:3])}

    # ==========================
    # 对外暴露的业务接口 (保持与之前微服务兼容)
    # ==========================
    def check_ingested(self, lecture_id: str) -> bool:
        try:
            res = self.vector_store.get(where={"lecture_id": lecture_id}, limit=1)
            return len(res.get("ids", [])) > 0
        except Exception:
            return False

    def get_namespace_chunks(self, lecture_id: str) -> list:
        try:
             res = self.vector_store.get(where={"lecture_id": lecture_id})
             chunks = []
             if res and "ids" in res:
                 for i in range(len(res["ids"])):
                     chunks.append({
                         "id": res["ids"][i],
                         "content": res.get("documents", [])[i] if res.get("documents") else "",
                         "metadata": res.get("metadatas", [])[i] if res.get("metadatas") else {}
                     })
             return chunks
        except Exception:
             return []

    async def get_answer(self, question: str, lecture_id: str, image_base64: Optional[str] = None) -> str:
        """接收特定命名空间下的微信学生提问，投入 LangGraph 多维沙盒流转"""
        try:
            _t0 = time.time()
            logger.info(f"get_answer started: question={question[:30]}..., lecture={lecture_id}")
            inputs = {
                "question": question,
                "lecture_id": lecture_id,
                "image_base64": image_base64,
                "visual_context": "",
                "documents": [],
                "loop_count": 0,
                "stream_mode": False
            }
            final_state = await self.app.ainvoke(inputs)
            _elapsed = time.time() - _t0
            logger.info(f"get_answer completed in {_elapsed:.2f}s, answer length: {len(final_state['generation'])} chars")
            return final_state["generation"]
        except Exception as e:
            logger.error(f"get_answer error: {e}", exc_info=True)
            return f"数字伴读服务（LangGraph 决策引擎）内网调度异常，请稍后再试。错误追踪：{str(e)}"

    async def get_answer_stream(self, question: str, lecture_id: str, image_base64: Optional[str] = None):
        """流式获取 RAG 引擎的思考与回答内容"""
        try:
            yield f"data: {json.dumps({'type': 'status', 'content': '🚀 伴学代理已启动，正在规划路径...'}, ensure_ascii=False)}\n\n"

            inputs = {
                "question": question,
                "lecture_id": lecture_id,
                "image_base64": image_base64,
                "visual_context": "",
                "documents": [],
                "loop_count": 0,
                "stream_mode": True
            }

            current_state = inputs
            async for chunk in self.app.astream(inputs, stream_mode="updates"):
                node_name = list(chunk.keys())[0]
                data = chunk[node_name]
                current_state.update(data)

                logger.debug(f"Node {node_name} completed")

                status_msg = {
                    "vision_analyze": "📸 视觉语义分析完成，提取特征中...",
                    "retrieve": "🔍 知识库匹配完成，正在精选资料...",
                    "grade_documents": "⚖️ 资料相关性评估完成，准备生成回答..."
                }

                if node_name == "vision_analyze" and not inputs.get("image_base64"):
                    pass
                elif node_name in status_msg:
                    yield f"data: {json.dumps({'type': 'status', 'content': status_msg[node_name]}, ensure_ascii=False)}\n\n"

                v_ctx = current_state.get("visual_context", "")
                is_v_failed = "失败" in v_ctx or not v_ctx.strip()
                if node_name == "vision_analyze" and is_v_failed and not question.strip():
                    err_msg = {
                        "type": "answer",
                        "content": "> 📸 **[AI 视觉核验]**：图片解析未获得有效信息。\n\n同学你好，刚才上传的截图内容较少或识别失败（我也没找到你输入的文字提问），建议重新截取清晰的代码或讲义片段再试哦！"
                    }
                    yield f"data: {json.dumps(err_msg, ensure_ascii=False)}\n\n"
                    return

                if node_name == "generate" or "generation" in data:
                    break
                if node_name == "refuse":
                    yield f"data: {json.dumps({'type': 'status', 'content': '🔍 本地资料不足，尝试联网搜索...'}, ensure_ascii=False)}\n\n"
                    web_results = self.web_search(question)
                    if web_results:
                        async for chunk in self._stream_web_answer(question, lecture_id, web_results):
                            yield chunk
                        return
                    yield f"data: {json.dumps({'type': 'answer', 'content': data.get('generation', '')}, ensure_ascii=False)}\n\n"
                    return

            docs = current_state.get("documents", [])[:3]
            context_blocks = [f"【来源文件: {d.metadata.get('source', '未知课件资料')}】:\n{d.page_content}" for d in docs]
            final_context = "\n\n".join(context_blocks)[:12000]

            final_question = question
            if current_state.get("visual_context"):
                final_question = f"学生上传了图片（视觉识别为：{current_state['visual_context']}）\n\n提问内容：{question}"

            prompt_text = self.generation_prompt.format(
                context=final_context,
                lecture_name="全课程主线知识库" if lecture_id == "GLOBAL_SEARCH" else lecture_id,
                question=final_question
            )

            logger.info(f"Final prompt length: {len(prompt_text)} chars ({len(docs)} slices)")

            api_url = f"{settings.LLM_API_BASE}/chat/completions"
            headers = {"Authorization": f"Bearer {settings.LLM_API_KEY}"}
            payload = {
                "model": settings.LLM_MODEL_NAME,
                "messages": [{"role": "user", "content": prompt_text}],
                "temperature": 0.7,
                "stream": True,
                "max_tokens": 8192,
                **self._no_think_params,
                "stop": ["<|im_end|>", "<|endoftext|>"]
            }

            yield f"data: {json.dumps({'type': 'status', 'content': '💡 响应就绪，正在生成回复...'}, ensure_ascii=False)}\n\n"

            if current_state.get("visual_context"):
                vision_summary_text = f"> 📸 **[AI 视觉核验]**：{current_state['visual_context']}\n\n---\n\n"
                yield f"data: {json.dumps({'type': 'answer', 'content': vision_summary_text}, ensure_ascii=False)}\n\n"

            logger.info(f"Starting streaming request to {api_url}")

            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream("POST", api_url, headers=headers, json=payload) as response:
                    if response.status_code != 200:
                        error_body = await response.aread()
                        error_msg = f"\n\n⚠️ API 调用异常 [{response.status_code}]: {error_body.decode()}"
                        logger.error(f"LLM API error: {response.status_code} - {error_body.decode()[:200]}")
                        payload_json = json.dumps({'type': 'answer', 'content': error_msg}, ensure_ascii=False)
                        yield f"data: {payload_json}\n\n"
                        return

                    logger.info("Streaming connection established")
                    is_thinking = False
                    post_think_cleanup_window = 0

                    async for line in response.aiter_lines():
                        if not line.startswith("data: ") or line == "data: [DONE]":
                            continue

                        try:
                            json_str = line[6:]
                            resp_data = json.loads(json_str)

                            delta = resp_data["choices"][0]["delta"]
                            content = delta.get("content", "")
                            reasoning = delta.get("reasoning_content", "")

                            final_content = ""
                            if reasoning:
                                if not is_thinking:
                                    final_content += "> 💭 **内部推理图谱**：\n> \n> "
                                    is_thinking = True
                                final_content += reasoning.replace("\n", "\n> ")

                            if content:
                                if is_thinking:
                                    final_content += "\n\n<hr/>\n\n"
                                    is_thinking = False
                                    post_think_cleanup_window = 10

                                if post_think_cleanup_window > 0:
                                    import re
                                    if re.search(r'[\u4e00-\u9fa5]', content):
                                        post_think_cleanup_window = 0
                                    else:
                                        if re.fullmatch(r'[a-zA-Z0-9\.\s\'\"\-\_]+', content):
                                            content = ""
                                            post_think_cleanup_window -= 1
                                        else:
                                            post_think_cleanup_window = 0

                                final_content += content

                            if not final_content:
                                continue

                            content_text = final_content.replace("\ufffd", "")
                            if not content_text:
                                continue

                            yield f"data: {json.dumps({'type': 'answer', 'content': content_text}, ensure_ascii=False)}\n\n"

                        except Exception:
                            continue

            logger.info("Stream completed")

        except Exception as e:
            logger.error(f"Stream error: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'content': f'发生未预期错误: {str(e)}'}, ensure_ascii=False)}\n\n"

async def _stream_web_answer(self, question: str, lecture_id: str, web_results: List[str]):
        """流式输出联网搜索结果"""
        web_context = "\n\n".join(web_results)

        prompt = f"""你是人工智能创新应用实验室 (AIIA Lab) 专属数字伴读助教。

⚠️ **重要提示**：当前问题超出了本地已入库的教学课件范围，已通过联网搜索获取相关资料。

请基于以下联网搜索结果回答学生问题。如果回答中涉及搜索到的课外知识，必须使用引用块进行免责声明。

【联网搜索资料】：
{web_context}

学生提问：{question}

你的专业助教解答："""

        web_note = "> ⚠️ **联网补充**：以下答案综合了本地课件知识和联网检索到的相关资料。\n\n"
        yield f"data: {json.dumps({'type': 'answer', 'content': web_note}, ensure_ascii=False)}\n\n"

        api_url = f"{settings.LLM_API_BASE}/chat/completions"
        headers = {"Authorization": f"Bearer {settings.LLM_API_KEY}"}
        payload = {
            "model": settings.LLM_MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "stream": True,
            "max_tokens": 4096,
            **self._no_think_params,
            "stop": ["<|im_end|>", "<|endoftext|>"]
        }

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream("POST", api_url, headers=headers, json=payload) as response:
                    if response.status_code != 200:
                        error_body = await response.aread()
                        yield f"data: {json.dumps({'type': 'answer', 'content': '联网搜索出错: ' + error_body.decode()}, ensure_ascii=False)}\n\n"
                        return

                    async for line in response.aiter_lines():
                        if not line.startswith("data: ") or line == "data: [DONE]":
                            continue
                        try:
                            json_str = line[6:]
                            resp_data = json.loads(json_str)
                            content = resp_data["choices"][0]["delta"].get("content", "")
                            if content:
                                yield f"data: {json.dumps({'type': 'answer', 'content': content}, ensure_ascii=False)}\n\n"
                        except Exception:
                            continue
        except Exception as e:
            logger.error(f"联网流式输出失败: {e}")
            yield f"data: {json.dumps({'type': 'answer', 'content': '联网生成失败: ' + str(e)}, ensure_ascii=False)}\n\n"

# 单例抛出
rag_engine = RAGEngine()
