# https://time.geekbang.org/column/article/709523
# 导入与Gmail交互所需的工具包
from langchain.agents.agent_toolkits import GmailToolkit

# 初始化Gmail工具包
# toolkit = GmailToolkit()

# 从gmail工具中导入一些有用的功能
from langchain_community.tools.gmail.utils import (
    build_resource_service,
    get_gmail_credentials,
)

from utils import create_llm

# 获取 Gmail API 的凭证，并指定相关的权限范围
credentials = get_gmail_credentials(
    token_file=r"GmailAIAssistant\token.json",
    scopes=[
        "https://www.googleapis.com/auth/gmail.readonly",
        # "https://www.googleapis.com/auth/gmail.modify",
        # "https://www.googleapis.com/auth/gmail.compose",
        "https://www.googleapis.com/auth/gmail.send"
    ],
    client_secrets_file=r"GmailAIAssistant\credentials.json",
)

# 使用凭证构建API资源服务
api_resource = build_resource_service(credentials)
toolkit = GmailToolkit(api_resource=api_resource)

# 获取工具
tools = toolkit.get_tools()
# print(tools)


llm = create_llm(is_chat_model=True,model="gpt-4")

from langchain.agents import AgentType, initialize_agent

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)


# input = "总结一下昨天的未读邮件";
input = "总结一下我收到的未读邮件"
input = "今天Microsoft给我发邮件了吗？最新的邮件是谁发给我的？"
input = "帮我写一封邮件，内容为”你好，这是由 Gmail AI Assistant 发出的测试邮件“,并发送到 eric.zhou@yamibuy.com 和 wheat.zhang@yamibuy.com"
result = agent.invoke(input)

print(result)