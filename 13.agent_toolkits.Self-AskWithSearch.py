# Self-Ask with Search 也是 LangChain 中的一个有用的代理类型（SELF_ASK_WITH_SEARCH）。
# 它利用一种叫做 “Follow-up Question（追问）”加“Intermediate Answer（中间答案）”的技巧，来辅助大模型寻找事实性问题的过渡性答案，从而引出最终答案。
from langchain_community.utilities import SerpAPIWrapper

from utils import create_llm, load_env_variables


from langchain.agents import AgentType, initialize_agent, Tool


llm = create_llm(is_chat_model=True)

serpapi_api_key = load_env_variables("SERPAPI_API_KEY")
search = SerpAPIWrapper(serpapi_api_key=serpapi_api_key)

# 指定使用 bing 搜索
# search = SerpAPIWrapper(serpapi_api_key=serpapi_api_key,params={
#     "engine": "bing",
#     "gl": "us",
#     "hl": "en",
# })

tools = [
    Tool(
        name="Intermediate Answer",
        func=search.run,
        description="useful for calculation, computation, or any other task that requires searching internet",
    )
]

self_ask_with_search = initialize_agent(
    tools,
    llm,
    agent=AgentType.SELF_ASK_WITH_SEARCH,
    verbose=True,
)

input = "使用玫瑰花作为国花的国家的首都是哪里？"
self_ask_with_search.run(input)


# 使用玫瑰作为国花的国家的首都是哪里?”这个问题不是一个简单的问题，它其实是一个多跳问题——在问题和最终答案之间，存在中间过程。
# 
# 多跳问题（Multi-hop question）是指为了得到最终答案，需要进行多步推理或多次查询。这种问题不能直接通过单一的查询或信息源得到答案，而是需要跨越多个信息点，或者从多个数据来源进行组合和整合。
# 
# 也就是说，问题的答案依赖于另一个子问题的答案，这个子问题的答案可能又依赖于另一个问题的答案。这就像是一连串的问题跳跃，对于人类来说，解答这类问题可能需要从不同的信息源中寻找一系列中间答案，然后结合这些中间答案得出最终结论。
# 
# “使用玫瑰作为国花的国家的首都是哪里？”这个问题并不直接询问哪个国家使用玫瑰作为国花，也不是直接询问英国的首都是什么。而是先要推知使用玫瑰作为国花的国家（英国）之后，进一步询问这个国家的首都。这就需要多跳查询。
# 
# 为什么 Self-Ask with Search 代理适合解决多跳问题呢？有下面几个原因。
# 1. 工具集合：代理包含解决问题所必须的搜索工具，可以用来查询和验证多个信息点。这里我们在程序中为代理武装了 SerpAPIWrapper 工具。
# 2. 逐步逼近：代理可以根据第一个问题的答案，提出进一步的问题，直到得到最终答案。这种逐步逼近的方式可以确保答案的准确性。
# 3. 自我提问与搜索：代理可以自己提问并搜索答案。例如，首先确定哪个国家使用玫瑰作为国花，然后确定该国家的首都是什么。
# 4. 决策链：代理通过一个决策链来执行任务，使其可以跟踪和处理复杂的多跳问题，这对于解决需要多步推理的问题尤为重要。
# 
# 在上面的例子中，通过大模型的两次 follow-up 追问，搜索工具给出了两个中间答案，最后给出了问题的最终答案——伦敦。