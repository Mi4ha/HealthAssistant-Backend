from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma

# 1. 设置你的阿里云百炼 API KEY
MY_ALIYUN_API_KEY = "sk-281cb6a56a954bfea647a2d1e6e0ee49"

def build_and_search_rag():
    print("🚀 正在启动 RAG 知识引擎...")

    # ==========================================
    # 第一阶段：读取文档并“切块” (Chunking)
    # ==========================================
    # 为什么要切块？大模型上下文有限，且切小块后搜索更精准。
    print("📖 正在读取本地医学知识库...")
    loader = TextLoader("knowledge.txt", encoding="utf-8")
    docs = loader.load()

    # 设置切块规则：每块大约 150 个字，块与块之间重叠 20 个字（防止一句话被从中间截断）
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=20)
    chunks = text_splitter.split_documents(docs)
    print(f"✂️ 文档已切分为 {len(chunks)} 个文本块。")

    # ==========================================
    # 第二阶段：向量化 (Embedding) 并存入 Chroma 数据库
    # ==========================================
    # 使用阿里云的 text-embedding-v3 模型，把文字变成多维数学向量
    embeddings = DashScopeEmbeddings(
        dashscope_api_key=MY_ALIYUN_API_KEY, 
        model="text-embedding-v4"
    )
    
    # 将向量数据持久化保存到本地的 "local_chroma_db" 文件夹中
    print("🧠 正在将文字转化为向量，并构建本地 Chroma 数据库...")
    vectorstore = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory="./local_chroma_db" # 你的数据会存在这个文件夹里
    )

    # ==========================================
    # 第三阶段：模拟用户提问，执行“相似度检索”
    # ==========================================
    # 假设用户刚吃了炸鸡，你的系统偷偷拿这个关键词去查书
    query = "炸鸡吃多了，第二天应该怎么运动和饮食来补救？"
    print(f"\n🔍 正在检索问题：【{query}】\n")

    # k=2 表示只找出最相关的 2 个文本块
    retrieved_docs = vectorstore.similarity_search(query, k=2)

    print("✅ 检索到的最相关医学知识如下：")
    print("-" * 50)
    # 将找到的文本块拼接成一个长字符串
    rag_context = ""
    for i, doc in enumerate(retrieved_docs):
        print(f"【片段 {i+1}】: {doc.page_content}")
        rag_context += doc.page_content + "\n"
    print("-" * 50)

    return rag_context

# 运行测试
if __name__ == "__main__":
    found_knowledge = build_and_search_rag()