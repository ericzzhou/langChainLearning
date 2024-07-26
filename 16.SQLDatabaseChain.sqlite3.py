# 导入langchain的实用工具和相关的模块
from langchain_community.utilities import SQLDatabase

from langchain_experimental.sql import SQLDatabaseChain

from utils import create_llm

# 连接到FlowerShop数据库（之前我们使用的是Chinook.db）
db = SQLDatabase.from_uri("sqlite:///FlowerShop.db")

# 创建OpenAI的低级语言模型（LLM）实例，这里我们设置温度为0，意味着模型输出会更加确定性
llm = create_llm(temperature=0)

# 创建SQL数据库链实例，它允许我们使用LLM来查询SQL数据库
db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)

# 运行与鲜花运营相关的问题
response = db_chain.invoke("有多少种不同的鲜花？")
print(response)

response = db_chain.invoke("哪种鲜花的存货数量最少？")
print(response)

response = db_chain.invoke("平均销售价格是多少？")
print(response)

response = db_chain.invoke("从法国进口的鲜花有多少种？")
print(response)

response = db_chain.invoke("哪种鲜花的销售量最高？")
print(response)
