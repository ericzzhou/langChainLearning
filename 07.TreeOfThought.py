# 思维树
# ToT 是一种解决复杂问题的框架，它在需要多步骤推理的任务中，引导语言模型搜索一棵由连贯的语言序列（解决问题的中间步骤）组成的思维树，而不是简单地生成一个答案。
# ToT 框架的核心思想是：让模型生成和评估其思维的能力，并将其与搜索算法（如广度优先搜索和深度优先搜索）结合起来，进行系统性地探索和验证。

# 思维链模板：参考 https://time.geekbang.org/column/article/701505
# 作为一个为花店电商公司工作的 AI 助手，我的目标是帮助客户根据他们的喜好做出明智的决定。

from utils import create_chat_prompt_template_from_template, create_llm

# 创建聊天模型
llm = create_llm(temperature=0,is_chat_model=True)

# 设定AI的角色和目标
system_template = "你是一个为花店电商公司工3作的AI助手, 你的目标是帮助客户根据他们的喜好做出明智的决定"

# CoT 的关键部分，AI 解释推理过程，并加入一些先前的对话示例（Few-Shot Learning）
cot_template = """
假设一个顾客在鲜花网站上询问：“我想为我的妻子购买一束鲜花，但我不确定应该选择哪种鲜花。她喜欢淡雅的颜色和花香。” 
AI（使用 ToT 框架）： 

思维步骤 1：理解顾客的需求。
顾客想为妻子购买鲜花。
顾客的妻子喜欢淡雅的颜色和花香。 

思维步骤 2：考虑可能的鲜花选择。
候选 1：百合，因为它有淡雅的颜色和花香。
候选 2：玫瑰，选择淡粉色或白色，它们通常有花香。
候选 3：紫罗兰，它有淡雅的颜色和花香。
候选 4：桔梗，它的颜色淡雅但不一定有花香。
候选 5：康乃馨，选择淡色系列，它们有淡雅的花香。 

思维步骤 3：根据顾客的需求筛选最佳选择。
百合和紫罗兰都符合顾客的需求，因为它们都有淡雅的颜色和花香。
淡粉色或白色的玫瑰也是一个不错的选择。
桔梗可能不是最佳选择，因为它可能没有花香。
康乃馨是一个可考虑的选择。 

思维步骤 4：给出建议。
“考虑到您妻子喜欢淡雅的颜色和花香，我建议您可以选择百合或紫罗兰。
淡粉色或白色的玫瑰也是一个很好的选择。希望这些建议能帮助您做出决策！”
"""

# 用户的询问
human_template = "{human_input}"

chat_prompt_template = create_chat_prompt_template_from_template(
    human_template=human_template,
    system_template=system_template,
    cot_template=cot_template,
)

prompt = chat_prompt_template.format_prompt(
    human_input="我想为我的女朋友购买一些花。开了一家书店，我买花做庆祝用。你有什么建议吗?"
)

response = llm.invoke(prompt)
print(response)
