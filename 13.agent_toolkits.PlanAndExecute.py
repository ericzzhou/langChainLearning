from langchain_community.utilities import SerpAPIWrapper
from langchain.agents import Tool
from langchain.chains.llm_math.base import LLMMathChain
from langchain_experimental.plan_and_execute import (
    PlanAndExecute,
    load_agent_executor,
    load_chat_planner,
)

from utils import create_llm, load_env_variables

llm = create_llm(is_chat_model=True)

serpapi_api_key = load_env_variables("SERPAPI_API_KEY")
search = SerpAPIWrapper(serpapi_api_key=serpapi_api_key)

llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=False)

tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for calculation, computation, or any other task that requires searching internet",
    ),
    Tool(
        name="Calculator",
        func=llm_math_chain.invoke,
        description="useful for when you need to answer questions about math",
    ),
]

model = create_llm(is_chat_model=True, temperature=0)

planner = load_chat_planner(model)
executor = load_agent_executor(model, tools=tools, verbose=True)

agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)

input = "在纽约，100美元能买几束玫瑰？"
result = agent.invoke(input)
