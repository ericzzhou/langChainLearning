# https://time.geekbang.org/column/article/703556
# 1. 构建处理模板：为鲜花护理和鲜花装饰分别定义两个字符串模板。
# 2. 提示信息：使用一个列表来组织和存储这两个处理模板的关键信息，如模板的键、描述和实际内容。
# 3. 初始化语言模型：导入并实例化语言模型。
# 4. 构建目标链：根据提示信息中的每个模板构建了对应的 LLMChain，并存储在一个字典中。
# 5. 构建 LLM 路由链：这是决策的核心部分。首先，它根据提示信息构建了一个路由模板，然后使用这个模板创建了一个 LLMRouterChain。
# 6. 构建默认链：如果输入不适合任何已定义的处理模板，这个默认链会被触发。
# 7. 构建多提示链：使用 MultiPromptChain 将 LLM 路由链、目标链和默认链组合在一起，形成一个完整的决策系统。

from utils import create_llm, create_prompt_from_template


flower_care_template = """
你是一个经验丰富的园丁，擅长解答关于养花育花的问题。
下面是需要你来回答的问题：
{input}
"""

flower_deco_template = """
你是一位网红插花大师，擅长解答关于鲜花装饰的问题。
下面是需要你来回答的问题：
{input}
"""

prompt_info = [
    {
        "key": "flower_care",
        "description": "适合回答关于鲜花养护的问题",
        "template": flower_care_template,
    },
    {
        "key": "flower_deco",
        "description": "适合回答关于鲜花装饰的问题",
        "template": flower_deco_template,
    },
]

llm = create_llm()

from langchain.chains.llm import LLMChain

# 构建目标链,循环 prompt_infos 这个列表，构建出两个目标链，分别负责处理不同的问题
chain_map = {}
for info in prompt_info:
    prompt_template = create_prompt_from_template(info["template"])
    print("目标提示：\n", prompt_template)

    chain = LLMChain(llm=llm, prompt=prompt_template, verbose=True)
    chain_map[info["key"]] = chain

# 构建路由链，负责查看用户输入的问题，确定问题的类型
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import (
    MULTI_PROMPT_ROUTER_TEMPLATE as RouterTemplate,
)

destinations = [f"{p['key']}: {p['description']}" for p in prompt_info]
router_template = RouterTemplate.format(destinations="\n".join(destinations))
print("路由提示模板：\n", router_template)
router_prompt = create_prompt_from_template(
    router_template, output_parser=RouterOutputParser()
)

print("路由提示词:\n", router_prompt)
router_chain = LLMRouterChain.from_llm(llm=llm, prompt=router_prompt, verbose=True)

# 构建默认链
from langchain.chains.conversation.base import ConversationChain

default_chain = ConversationChain(llm=llm, output_key="text", verbose=True)

# 构建多提示链
from langchain.chains.router.multi_prompt import MultiPromptChain

chain = MultiPromptChain(
    router_chain=router_chain,
    destination_chains=chain_map,
    default_chain=default_chain,
    verbose=True,
)

input = "如何为茉莉花浇水？"
input = "怎么给婚礼场地布置花朵？"
input = "如何考入哈佛大学？"
result = chain.run(input)
print(result)
