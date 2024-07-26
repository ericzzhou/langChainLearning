from langchain_community.utilities import SQLDatabase

from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType

from utils import create_llm

# 连接到FlowerShop数据库
db = SQLDatabase.from_uri("sqlite:///FlowerShop.db")
llm = create_llm(temperature=0)

sqlToolkit = SQLDatabaseToolkit(db=db, llm=llm)

# 创建SQL Agent
agent_executor = create_sql_agent(
    llm=llm,
    toolkit=sqlToolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)

# 使用Agent执行SQL查询

questions = [
    "哪种鲜花的存货数量最少？",
    "有多少种不同的鲜花？",
]

for question in questions:
    response = agent_executor.invoke(question)
    print(response)
