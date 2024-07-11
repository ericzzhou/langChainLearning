# 海报文案生成器

# 1. 初始化图像字幕生成模型 (HuggingFace 中的 image-caption 模型)
# 2. 定义 LangChain 图像字幕生成工具
# 3. 初始化并运行 LangChain Agent , 这个 Agent 是OpenAI 的大语言模型，会自动进行分析，调用工具，完成任务

# pip install --upgrade langchain
# pip install transformers
# pip install pillow
# pip install torch torchvision torchaudio

import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from langchain.tools import BaseTool
from langchain_openai import OpenAI
from langchain.agents import initialize_agent,AgentType
from io import BytesIO

from utils import create_llm

#--- part Ⅰ 初始化图像字幕生成模型
# 指定要使用的工具模型 (HuggingFace 中的 image-caption 模型)
hf_model = "Salesforce/blip-image-captioning-large"

# 初始化处理器和工具模型
# 预处理器将准备图像供模型使用
processor = BlipProcessor.from_pretrained(hf_model)
# 初始化工具模型本身
model = BlipForConditionalGeneration.from_pretrained(hf_model)

#-- Part Ⅱ 定义图像字幕生成工具类
class ImageGapTool(BaseTool):
    name = "Image Caption Generator"
    description = "为图片创作说明文案"

    def _run(self, query: str) -> str:
        # 查询转换为图像 URL
        image_url = query.strip()

        # 下载图像
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))

        # 预处理图像
        inputs = processor(image, return_tensors="pt")
        # print("inputs:", inputs)
        # 生成图像描述(字幕)
        outputs = model.generate(**inputs, max_new_tokens=100)
        # print("outputs:", outputs)
        caption = processor.decode(outputs[0], skip_special_tokens=True)

        return caption

    async def _async_run(self, query: str) -> str:
        raise NotImplementedError("tool do not support async")
    

#---- PartIII 初始化并运行LangChain智能代理
# 设置OpenAI的API密钥并初始化大语言模型（OpenAI的Text模型）

# temperature=0.2 代表模型拥有一定的随机性
llm = create_llm()

# 使用工具初始化智能代理并运行它
tool = [ImageGapTool()]
agent = initialize_agent(
    tool, 
    llm, 
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
    verbose=True
)

# img_url = "https://cdn.yamibuy.net/item/5357daab9ee5e66463b1b801b5560cb6_400x400.webp"
img_url = "https://mir-s3-cdn-cf.behance.net/project_modules/hd/eec79e20058499.563190744f903.jpg"
# agent.invoke({"input": img_url}, include_run_info=True)
agent.invoke(input=f"{img_url}\n请创作合适的中文推广文案")
