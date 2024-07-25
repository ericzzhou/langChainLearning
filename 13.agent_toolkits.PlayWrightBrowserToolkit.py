from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.tools.playwright.utils import create_async_playwright_browser

from utils import create_llm

async_browser = create_async_playwright_browser()
toolkits = PlayWrightBrowserToolkit.from_browser(async_browser=async_browser)

# 获取工具
tools = toolkits.get_tools()
# print(tools)

from langchain.agents import AgentType, initialize_agent


llm = create_llm(is_chat_model=True)

agent_chain = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

# response = agent_chain.run("What are the headers on langchain.com?")
# print(response)
input = "帮我抓取 www.yamibuy.com 的网页标题"
input = "What are the headers on https://www.yamibuy.com/zh ?"
input = "帮我抓出来 https://www.yamibuy.com/zh 网页的所有H标签"
async def main():
    response = await agent_chain.arun(input)
    print(response)


import asyncio

loop = asyncio.get_event_loop()
loop.run_until_complete(main())
