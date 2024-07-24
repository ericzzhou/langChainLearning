from utils import create_llm
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain

template = "{flower}的花语是？"
llm = create_llm(temperature=0)

llm_chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template(template))

result = llm_chain("玫瑰")

print(result)
