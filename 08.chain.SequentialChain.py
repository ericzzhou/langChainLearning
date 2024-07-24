from utils import create_llm, create_prompt_from_template
from langchain.chains.llm import LLMChain

llm = create_llm(temperature=0.1)

# LLMChain 1
template = """
你是一个植物学家。给定花的名称和类型，你需要为这种花写一个200字左右的介绍。

花名：{name}
颜色：{color}

植物学家：这是关于上述花的介绍：
"""
prompt_template = create_prompt_from_template(template)
introduction_chain = LLMChain(
    llm=llm, prompt=prompt_template, output_key="introduction"
)


# LLMChain 2
template = """你是一位鲜花评论家。给定一种花的介绍，你需要为这种花写一篇200字左右的评论。

鲜花介绍:
{introduction}

花评人对上述花的评论：
"""
prompt_template = create_prompt_from_template(template)
comment_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="comment")


# LLMChain 3
template = """
你是一家花店的社交媒体经理。给定一种花的介绍和评论，你需要为这种花写一篇300字左右的社交媒体的帖子。

鲜花介绍:
{introduction}

花评人对上述花的评论：
{comment}

社交媒体帖子：
"""
prompt_template = create_prompt_from_template(template)
social_media_chain = LLMChain(
    llm=llm, prompt=prompt_template, output_key="social_media"
)


# 顺序链
from langchain.chains.sequential import SequentialChain

overall_chain = SequentialChain(
    chains=[introduction_chain, comment_chain, social_media_chain],
    input_variables=["name", "color"],
    output_variables=["introduction", "comment", "social_media"],
    verbose=True,
)

result = overall_chain({"name": "玫瑰", "color": "粉色"})
print(result)
