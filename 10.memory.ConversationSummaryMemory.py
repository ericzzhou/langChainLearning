# https://time.geekbang.org/column/article/704183

# ConversationSummaryMemory（对话总结记忆）的思路就是将对话历史进行汇总，然后再传递给 {history} 参数。
# 这种方法旨在通过对之前的对话进行汇总来避免过度使用 Token。
#
# ConversationSummaryMemory 有这么几个核心特点。
# 1. 汇总对话：此方法不是保存整个对话历史，而是每次新的互动发生时对其进行汇总，然后将其添加到之前所有互动的“运行汇总”中。
# 2. 使用 LLM 进行汇总：该汇总功能由另一个 LLM 驱动，这意味着对话的汇总实际上是由 AI 自己进行的。
# 3. 适合长对话：对于长对话，此方法的优势尤为明显。虽然最初使用的 Token 数量较多，但随着对话的进展，汇总方法的增长速度会减慢。与此同时，常规的缓冲内存模型会继续线性增长。


from langchain.chains.conversation.base import ConversationChain
from langchain.chains.conversation.memory import ConversationSummaryMemory

from utils import create_llm

llm = create_llm(temperature=0.5)

# 初始化对话链
conv_chain = ConversationChain(
    llm=llm, verbose=True, memory=ConversationSummaryMemory(llm=llm)
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

# 看得出来，这里的 'history'，不再是之前人类和 AI 对话的简单复制粘贴，而是经过了总结和整理之后的一个综述信息。
# 
# 这里，我们不仅仅利用了 LLM 来回答每轮问题，还利用 LLM 来对之前的对话进行总结性的陈述，以节约 Token 数量。这里，帮我们总结对话的 LLM，和用来回答问题的 LLM，可以是同一个大模型，也可以是不同的大模型。
# 
# ConversationSummaryMemory 的优点是对于长对话，可以减少使用的 Token 数量，因此可以记录更多轮的对话信息，使用起来也直观易懂。不过，它的缺点是，对于较短的对话，可能会导致更高的 Token 使用。
# 另外，对话历史的记忆完全依赖于中间汇总 LLM 的能力，还需要为汇总 LLM 使用 Token，这增加了成本，且并不限制对话长度。
# 
# 通过对话历史的汇总来优化和管理 Token 的使用，ConversationSummaryMemory 为那些预期会有多轮的、长时间对话的场景提供了一种很好的方法。
# 然而，这种方法仍然受到 Token 数量的限制。在一段时间后，我们仍然会超过大模型的上下文窗口限制。
# 而且，总结的过程中并没有区分近期的对话和长期的对话（通常情况下近期的对话更重要），所以我们还要继续寻找新的记忆管理方法。


