# https://time.geekbang.org/column/article/704183

# ConversationSummaryBufferMemory，即对话总结缓冲记忆，它是一种混合记忆模型，结合了上述各种记忆机制，包括 ConversationSummaryMemory 和 ConversationBufferWindowMemory 的特点。
# 这种模型旨在在对话中总结早期的互动，同时尽量保留最近互动中的原始内容。
#
# 它是通过 max_token_limit 这个参数做到这一点的。
# 当最新的对话文字长度在 100 字之内的时候，LangChain 会记忆原始对话内容；当对话文字超出了这个参数的长度，那么模型就会把所有超过预设长度的内容进行总结，以节省 Token 数量。


from langchain.chains.conversation.base import ConversationChain
from langchain.chains.conversation.memory import ConversationSummaryBufferMemory

from utils import create_llm

llm = create_llm(temperature=0.5)

# 初始化对话链
conv_chain = ConversationChain(
    llm=llm,
    verbose=True,
    memory=ConversationSummaryBufferMemory(llm=llm, max_token_limit=10),
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

# 在第二回合，记忆机制完整地记录了第一回合的对话，但是在第三回合，它察觉出前两轮的对话已经超出了 100 个字节，就把早期的对话加以总结，以节省 Token 资源。
# 
# ConversationSummaryBufferMemory 的优势是通过总结可以回忆起较早的互动，而且有缓冲区确保我们不会错过最近的互动信息。
# 当然，对于较短的对话，ConversationSummaryBufferMemory 也会增加 Token 数量。
# 
# 总体来说，ConversationSummaryBufferMemory 为我们提供了大量的灵活性。
# 它是我们迄今为止的唯一记忆类型，可以回忆起较早的互动并完整地存储最近的互动。在节省 Token 数量方面，ConversationSummaryBufferMemory 与其他方法相比，也具有竞争力。
