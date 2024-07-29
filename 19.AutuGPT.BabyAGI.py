# 导入所需的库和模块
from typing import Dict, List, Optional

# from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from utils import create_llm

from MyBabyAGI.BabyAGI import BabyAGI

# 定义嵌入模型
embeddings_model = OpenAIEmbeddings()
# 初始化向量存储
embedding_size = 1536
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})


# 主执行部分
if __name__ == "__main__":
    OBJECTIVE = "分析一下北京市今天的气候情况，写出鲜花储存策略。"
    llm = create_llm()
    verbose = False
    max_iterations: Optional[int] = 6
    baby_agi = BabyAGI.from_llm(
        llm=llm, vectorstore=vectorstore, verbose=verbose, max_iterations=max_iterations
    )
    baby_agi({"objective": OBJECTIVE})
