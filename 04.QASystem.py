from utils import (
    create_memory_vectorstore,
    create_qa_chain,
    load_documents,
    split_documents,
)


# 加载 document
base_dir = ".\\static"
documents = load_documents(base_dir)

# print(documents)

# 2. split 将 Documents 切分成块以便后续进行嵌入和向量存储
chunked_document = split_documents(documents)

print(f"Number of documents after chunking: {len(chunked_document)}")

# 3. 将分隔嵌入并存储在向量库 Qdrant 中
vectorstore = create_memory_vectorstore(
    chunked_document=chunked_document,
)

# 4. Retrieval 准备模型和 Retrieval链
import logging

# 设置 logging
logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

qa_chain = create_qa_chain(vectorstore)

q = "董事长致辞中提到的企业精神指的是什么？"
q = "Tulip 的花语是什么？"
q = "易速鲜花员工离职有什么手续？"
q = "上班时间员工离岗怎么办？"
result = qa_chain({"query": q})
print(result["result"])
