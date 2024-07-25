# https://time.geekbang.org/column/article/704183
from langchain.chains.conversation.base import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

from utils import create_llm

llm = create_llm(temperature=0.5)

# 初始化对话链
conv_chain = ConversationChain(
    llm=llm, verbose=True, memory=ConversationBufferWindowMemory(k=1)
)

# 打印对话模板
# print(conv_chain.prompt.template)
# ```
# The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

# Current conversation:
# {history}
# Human: {input}
# AI:
# ```

# 对话1
conv_chain("我姐姐明天要过生日，我需要一束花生日花束。")
print("第一次对话后的记忆：\n", conv_chain.memory.to_json(), "\n")

# 对话2
conv_chain("她喜欢粉色，你有推荐的吗？")
print("第2次对话后的记忆：\n", conv_chain.memory.to_json(), "\n")

# 对话3
conv_chain("我又来了，还记得我昨天来做什么以及为什么来吗？？")
print("第3次对话后的提示：\n", conv_chain.prompt.template)
print("第3次对话后的记忆：\n", conv_chain.memory.to_json(), "\n")

# 实际上，这些聊天历史信息，都被传入了 ConversationChain 的提示模板中的 {history} 参数，构建出了包含聊天记录的新的提示输入。
# 有了记忆机制，LLM 能够了解之前的对话内容，这样简单直接地存储所有内容为 LLM 提供了最大量的信息，但是新输入中也包含了更多的 Token（所有的聊天历史记录），这意味着响应时间变慢和更高的成本。
# 而且，当达到 LLM 的令牌数（上下文窗口）限制时，太长的对话无法被记住（对于 text-davinci-003 和 gpt-3.5-turbo，每次的最大输入限制是 4096 个 Token）。

# 下面我们来看看针对 Token 太多、聊天历史记录过长的一些解决方案。
# 使用 ConversationBufferWindowMemory
# 说到记忆，我们人类的大脑也不是无穷无尽的。所以说，有的时候事情太多，我们只能把有些遥远的记忆抹掉。
# 毕竟，最新的经历最鲜活，也最重要。ConversationBufferWindowMemory 是缓冲窗口记忆，它的思路就是只保存最新最近的几次人类和 AI 的互动。因此，它在之前的“缓冲记忆”基础上增加了一个窗口值 k。
# 这意味着我们只保留一定数量的过去互动，然后“忘记”之前的互动。
