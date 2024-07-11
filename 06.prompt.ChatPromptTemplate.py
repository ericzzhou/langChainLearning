from utils import create_chat_prompt_template_from_template, create_llm


prompt_template = create_chat_prompt_template_from_template(
    system_template="你是一位专业顾问，负责为专注于{product}的公司起名。",
    human_template="公司主打产品是{product_details}。",
    ai_template="正在思考......",
)
# 格式化提示消息生成提示
prompt = prompt_template.format_messages(
    product="鲜花装饰", product_details="创新的鲜花设计"
)
prompt = prompt_template.format_messages(
    product="亚洲零食在线销售", product_details="亚洲零食，快消品"
)
print(prompt)

llm = create_llm()

result = llm.invoke(prompt)
print(result)
