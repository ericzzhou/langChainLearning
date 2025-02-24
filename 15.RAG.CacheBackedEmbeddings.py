# 导入内存存储库，该库允许我们在RAM中临时存储数据
from langchain.storage import InMemoryStore

# 创建一个InMemoryStore的实例
store = InMemoryStore()

# 导入与嵌入相关的库。OpenAIEmbeddings是用于生成嵌入的工具，而CacheBackedEmbeddings允许我们缓存这些嵌入
from langchain.embeddings import CacheBackedEmbeddings
from langchain_openai import OpenAIEmbeddings

# 创建一个OpenAIEmbeddings的实例，这将用于实际计算文档的嵌入
embeddings_model = OpenAIEmbeddings()

# 创建一个CacheBackedEmbeddings的实例。
# embeddings_model
# 我们还为缓存指定了一个命名空间，以确保不同的嵌入模型之间不会出现冲突。
embedder = CacheBackedEmbeddings.from_bytes_store(
    embeddings_model,  # 实际生成嵌入的工具
    store,  # 嵌入的缓存位置
    namespace=embeddings_model.model,  # 嵌入缓存的命名空间
)

# 使用embedder为两段文本生成嵌入。
# 结果，即嵌入向量，将被存储在上面定义的内存存储中。
embeddings = embedder.embed_documents(["你好", "智能鲜花客服"])
# print(embeddings[:3])
