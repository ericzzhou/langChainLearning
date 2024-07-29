import os
from typing import List, Optional
from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEndpoint


def load_env_variables(key):
    """Load environment variables from .env file."""
    load_dotenv()
    return os.environ[key]


def load_openai_api_key():
    """Load the OpenAI API key from the environment variable."""
    load_dotenv()
    return os.environ["OPENAI_API_KEY"]


def load_huggingfacehub_api_token():
    """
    从环境变量或其他配置中加载 Hugging Face API token。
    返回：
        str: Hugging Face API token。
    """
    load_dotenv()
    return os.environ["HUGGINGFACEHUB_API_TOKEN"]


def create_llm(
    openai_api_key=None,
    model="gpt-3.5-turbo-instruct",
    temperature=0.3,
    is_chat_model=False,
    max_tokens=200,
):
    """
    创建一个 OpenAI 语言模型实例（LLM）。

    参数:
    - openai_api_key (str, optional): OpenAI API 密钥。如果未提供，则从环境中加载。
    - model (str, optional): 要使用的模型名称。默认为 "gpt-3.5-turbo-instruct",制定使用聊天模型时，默认为 gpt-4o。
    - temperature (float, optional): 生成文本的随机性。较低的值会使输出更确定。默认为 0.3。
    - is_chat_model (bool, optional): 指定是否使用聊天模型。默认为 False。
    - max_tokens (int, optional): 聊天模型使用，生成的最大标记数。默认为 200。

    返回:
    - OpenAI 或 ChatOpenAI 实例，取决于 is_chat_model 的值。
    """
    if openai_api_key is None:
        openai_api_key = load_openai_api_key()

    if is_chat_model:
        model = "gpt-4o"
        # 如果 is_chat_model 为 True，则返回 ChatOpenAI 实例
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=openai_api_key,
        )
    # 如果 is_chat_model 为 False，则返回 OpenAI 实例
    return OpenAI(
        model=model,
        temperature=temperature,
        openai_api_key=openai_api_key,
    )


def create_hf_model(
    huggingfacehub_api_token=None,
    model_name="bigscience/bloom-1b7",
    temperature=0,
    max_tokens=200,
    timeout=300,
):
    """
    创建并返回一个 Hugging Face 模型实例。

    参数：
        huggingfacehub_api_token (str): Hugging Face API token。如果未提供，则从配置中加载。
        model_name (str): 模型名称或仓库 ID，默认为 "bigscience/bloom-1b7"。
        temperature (float): 生成文本的温度，影响输出的随机性。默认为 0。
        max_tokens (int): 生成文本的最大 token 数。默认为 200。
        timeout (int): 请求超时时间（秒）。默认为 300。

    返回值：
        HuggingFaceEndpoint: 配置好的 Hugging Face 模型实例。
    """

    # 如果未提供 API token，则从配置中加载
    if huggingfacehub_api_token is None:
        huggingfacehub_api_token = load_huggingfacehub_api_token()

    # 创建并返回 Hugging Face 模型实例
    return HuggingFaceEndpoint(
        repo_id=model_name,  # 模型名称或仓库 ID
        huggingfacehub_api_token=huggingfacehub_api_token,  # Hugging Face API token
        timeout=timeout,  # 请求超时时间
        temperature=temperature,  # 生成文本的温度
        max_tokens=max_tokens,  # 生成文本的最大 token 数
    )


from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader


def load_documents(base_dir):
    """
    从指定目录中加载文档，并根据文件类型使用相应的加载器处理。

    参数：
        base_dir (str): 包含文档的目录路径。

    返回：
        list: 加载的文档列表，每个文档是一个字符串。
    """
    documents = []
    for file in os.listdir(base_dir):
        # 构建完整的文件路径
        file_path = os.path.join(base_dir, file)

        if file.endswith(".txt"):
            loader = TextLoader(file_path)
            documents.extend(loader.load())
        elif file.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
        elif file.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
            documents.append(loader.load()[0])
    return documents


from langchain.text_splitter import RecursiveCharacterTextSplitter


def split_documents(
    documents, chunk_size=200, chunk_overlap=0, separators=["\n\n", "\n", ".", " ", ""]
):
    """
    将文档分割成小块，以便更好地处理和分析。

    参数：
        documents (list): 要分割的文档列表。
        chunk_size (int): 每个分块的最大字符数。默认为 200。
        chunk_overlap (int): 分块之间的重叠字符数。默认为 0。
        separators (list): 用于分割文档的分隔符列表。默认为 ["\n\n", "\n", ".", " ", ""]。

    返回：
        list: 分割后的文档列表，每个分块是一个字符串。
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
    )
    chunked_document = text_splitter.split_documents(documents)
    return chunked_document


from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings


def create_memory_vectorstore(chunked_document, openai_api_key=None):
    """
    创建一个向量存储器，将已分块的文档向量化并存储在内存中。

    参数：
        chunked_document (list): 分块后的文档列表，每个分块是一个字符串。
        openai_api_key (str, optional): OpenAI API 的访问密钥。如果为 None，则尝试从环境变量或其他配置加载。

    返回：
        Qdrant: 创建的向量存储器对象。

    注：
        - 这里使用 OpenAI 的 Embedding Model 作为向量化提供者。
        - 可以根据需求替换为其他的 Embedding Provider。
    """

    if openai_api_key is None:
        openai_api_key = load_openai_api_key()

    vectorstore = Qdrant.from_documents(
        documents=chunked_document,  # 已分块的文档
        embedding=OpenAIEmbeddings(),  # 用OpenAI的Embedding Model 做向量化
        location=":memory:",  # 内存存储
        collection_name="my_documents",  # 指定存储的集合名称
        openai_api_key=openai_api_key,
    )
    return vectorstore


from langchain.retrievers.multi_query import (
    MultiQueryRetriever,
)  # MultiQueryRetriever 工具
from langchain.chains import RetrievalQA  # RetrievalQA链


def create_qa_chain(vectorstore, llm=None):
    """
    创建一个问答系统链，将检索器与语言模型连接起来。

    参数：
        vectorstore (Qdrant): 向量存储器对象，用于文档检索。
        llm (LanguageModel, optional): 语言模型对象，用于生成答案。如果为 None，则会创建一个默认的聊天模型。

    返回：
        RetrievalQA: 创建的问答系统链对象，结合了检索器和语言模型。

    注：
        - 使用 MultiQueryRetriever 从语言模型和向量存储器中创建检索器。
        - 使用 RetrievalQA.from_chain_type 创建一个检索问答系统链。
    """
    if llm is None:
        llm = create_llm(model_name="gpt-4o", is_chat_model=True)
    # 实例化一个 MultiQueryRetriever
    retriever_from_llm = MultiQueryRetriever.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        # chain_type="stuff",
        retriever=retriever_from_llm,
        # return_source=True,
    )

    return qa_chain


from langchain.output_parsers import StructuredOutputParser, ResponseSchema


def create_pydantic_output_parser(pydantic_object):
    """
    创建一个用于解析 Pydantic 对象输出的解析器。

    参数:
    pydantic_object (BaseModel): 需要用于输出解析的 Pydantic 模型类。

    返回:
    PydanticOutputParser: Pydantic 输出解析器实例。
    """
    from langchain.output_parsers import PydanticOutputParser

    # # 定义我们想要接收的数据格式
    # from pydantic import BaseModel, Field
    # class FlowerDescription(BaseModel):
    #     flower_type: str = Field(description="鲜花的种类")
    #     price: int = Field(description="鲜花的价格")
    #     description: str = Field(description="鲜花的描述文案")
    #     reason: str = Field(description="为什么要这样写这个文案")

    # 创建 PydanticOutputParser 实例，并传入 Pydantic 模型类
    return PydanticOutputParser(pydantic_object=pydantic_object)


def create_structured_output_parser(response_schemas: List[ResponseSchema]):
    """
    创建一个结构化输出解析器对象，根据提供的响应模式列表生成解析器。

    Args:
    - response_schemas (List[ResponseSchema]): 包含响应模式的列表，每个模式定义了一个响应字段的名称、类型和描述。

    Returns:
    - StructuredOutputParser: 根据响应模式生成的结构化输出解析器对象。
    """

    # response_schemas = [
    #     ResponseSchema(
    #         name="name",
    #         type="string",
    #         description="鲜花的名字",
    #     ),
    #     ResponseSchema(
    #         name="price",
    #         type="decimal",
    #         description="鲜花的价格",
    #     ),

    # ]
    return StructuredOutputParser.from_response_schemas(response_schemas)


from langchain_core.output_parsers import BaseOutputParser


def create_prompt_from_template(
    template: str,
    format_instructions=None,
    output_parser: BaseOutputParser | None = None,
):
    """
    从模板字符串创建 PromptTemplate 实例。

    参数：
    - template (str): 用于生成提示的模板字符串。
    - format_instructions (可选): 用于格式化的说明字符串，默认为 None。
    - output_parser (BaseOutputParser | None): 用于解析输出的解析器实例，默认为 None。

    返回：
    - PromptTemplate: 创建的 PromptTemplate 实例。
    """
    if format_instructions is None:
        return PromptTemplate(template=template, output_parser=output_parser)

    return PromptTemplate(
        template=template,
        output_parser=output_parser,
        partial_variables={"format_instructions": format_instructions},
    )


from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
)


def isNullOrEmpty(value: str):
    """
    检查值是否为 None 或空字符串。

    Args:
    - value: 要检查的值。

    Returns:
    - bool: 如果值为 None 或空字符串，则返回 True；否则返回 False。
    """
    return (
        value is None or value == "" or value.strip() == "" or len(value.strip()) == 0
    )


def create_chat_prompt_template_from_template(
    system_template: str = None,
    human_template: str = None,
    ai_template: str = None,
    cot_template: str = None,
):
    """
    根据提供的模板创建聊天提示模板。

    Args:
    - system_template (str): 系统消息的模板，，可选。
    - human_template (str): 人类消息的模板，可选。
    - ai_template (str): AI 消息的模板，可选。
    - cot_template: 系统 COT（Chain of Thought）消息模板，可选。

    Returns:
    - 一个 ChatPromptTemplate 对象，包含从模板生成的消息。
    """

    messages = []

    if not isNullOrEmpty(system_template):
        system_message_prompt = SystemMessagePromptTemplate.from_template(
            system_template
        )
        if system_message_prompt:
            messages.append(system_message_prompt)
    if not isNullOrEmpty(cot_template):
        system_cot_prompt = SystemMessagePromptTemplate.from_template(cot_template)
        if system_cot_prompt:
            messages.append(system_cot_prompt)
    if not isNullOrEmpty(human_template):
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
        if human_message_prompt:
            messages.append(human_message_prompt)

    if not isNullOrEmpty(ai_template):
        ai_message_prompt = AIMessagePromptTemplate.from_template(ai_template)
        if ai_message_prompt:
            messages.append(ai_message_prompt)

    return ChatPromptTemplate.from_messages(messages)


# 创建一个 FewShotPromptTemplate对象
from langchain.prompts import PromptTemplate


def create_fewshot_prompt_template_by_samples(
    example_prompt_template: PromptTemplate,
    suffix: str,
    input_variables: List[str],
    examples=None,
):
    """
    创建一个包含少量示例(FewShotPromptTemplate)的提示模板。

    Args:
    - example_prompt_template (PromptTemplate): 用于生成示例的提示词模板。
    - suffix (str): 在提示末尾添加的后缀。
    - input_variables (List[str]): 提示中使用的输入变量列表。
    - examples (list, optional): 提供的示例列表，默认为空列表。

    Returns:
    - FewShotPromptTemplate: 创建的 FewShotPromptTemplate 实例。

    示例:
    example_prompt_template = PromptTemplate.from_template(
        template="你是一位专业的鲜花店文案撰写员。对于售价为 {price} 元的 {flower_name} ，你能提供一个吸引人的简短描述吗？"
    )
    few_shot_template = create_fewshot_prompt_template(
        example_prompt=example_prompt_template,
        suffix="这是一个例子：",
        input_variables=["price", "flower_name"],
        examples=[{"price": 100, "flower_name": "月季花"}],
    )
    print(few_shot_template)
    """
    from langchain.prompts import FewShotPromptTemplate

    if examples is None:
        examples = []
    return FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt_template,
        suffix=suffix,
        input_variables=input_variables,
    )


from langchain.prompts.example_selector import (
    SemanticSimilarityExampleSelector,
)

from langchain_community.vectorstores import Chroma


def create_fewshot_prompt_template_by_example_selector(
    examples: Optional[List[dict]],
    prefix: str,
    suffix: str,
    input_variables: List[str],
    example_prompt_template: str,
    example_input_variables: List[str],
):
    """
    初始化一个基于语义相似度的示例选择器并创建 FewShotPromptTemplate 对象。

    Args:
    - examples (List[Dict]): 示例列表，每个示例是一个字典，包含提示和完成。
    - prefix (str): 提示前缀。
    - suffix (str): 提示后缀。
    - input_variables (List[str]): 输入变量。
    - example_prompt_template (str): 示例提示模板。
    - example_input_variables (List[str]): 示例输入变量。

    Returns:
    - FewShotPromptTemplate: 初始化的 FewShotPromptTemplate 对象。
    """
    from langchain.prompts import FewShotPromptTemplate

    embeddings = OpenAIEmbeddings(openai_api_type=load_openai_api_key())
    example_selector = SemanticSimilarityExampleSelector.from_examples(
        examples=examples, embeddings=embeddings, vectorstore_cls=Chroma
    )

    example_prompt = create_prompt_from_template(
        template=example_prompt_template,
        format_instructions=example_input_variables,
    )
    return FewShotPromptTemplate(
        example_selector=example_selector,
        prefix=prefix,
        suffix=suffix,
        input_variables=input_variables,
        example_prompt=example_prompt,
    )


import requests
from langchain.llms import BaseLLM
from langchain_core.messages import BaseMessage


class create_local_llm(BaseLLM):
    """
    一个自定义的 LLM 类，用于通过 API 与本地部署的 Llama3 模型交互。
    """

    def __init__(self, api_url, api_key):
        """
        初始化 create_local_llm 实例。

        参数:
        api_url (str): 本地 API 端点的 URL。
        api_key (str): 认证所需的 API 密钥。
        """
        self.api_url = api_url
        self.api_key = api_key

    def _call(
        self,
        prompt: List[BaseMessage],
        model: str = "lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF",
        stop: list = None,
        temperature: float = 0.7,
    ) -> str:
        """
        向本地 Llama3 API 发送请求并返回响应。

        参数:
        prompt (List[BaseMessage]): 模型的提示消息。
        model (str): 模型标识符。
        stop (list, optional): 停止词列表。
        temperature (float, optional): 采样温度。

        返回:
        str: 模型的响应。
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "stop": stop,
            "temperature": temperature,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }
        response = requests.post(self.api_url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()
