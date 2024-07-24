# 定义一个模板字符串，这个模板将用于生成提问
template = """Based on the user question, provide an Action and Action Input for what step should be taken.
{format_instructions}
Question: {query}
Response:"""

# 定义一个Pydantic数据格式，它描述了一个"行动"类及其属性
from pydantic import BaseModel, Field

from utils import create_llm, create_pydantic_output_parser


class Action(BaseModel):
    action: str = Field(description="action to take")
    action_input: str = Field(description="input to the action")


# 使用Pydantic格式Action来初始化一个输出解析器
from langchain.output_parsers import (
    PydanticOutputParser,
    OutputFixingParser,
    RetryWithErrorOutputParser,
)

parser = PydanticOutputParser(pydantic_object=Action)

# 定义一个提示模板，它将用于向模型提问
from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)
prompt_value = prompt.format_prompt(query="What are the colors of Orchid?")

# 定义一个错误格式的字符串
bad_response = '{"action": "search"}'
# parser.parse(bad_response)  # 如果直接解析，它会引发一个错误

# fix_parser = OutputFixingParser.from_llm(llm=create_llm(), parser=parser)

# parse_result = fix_parser.parse(bad_response)

# print("OutputFixingParser 的 Parse 结果：", parse_result)

retry_parser = RetryWithErrorOutputParser.from_llm(
    llm=create_llm(), parser=parser, max_retries=3
)

parse_result = retry_parser.parse_with_prompt(bad_response, prompt_value)
print("RetryWithErrorOutputParser的parse结果:", parse_result)

# 结论：
# 运行失败，按照教程说明，使用一个错误的json格式不会再报错。
# 实际上运行结果依然报错