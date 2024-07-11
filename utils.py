import os
from typing import List
from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEndpoint


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


def create_llm(openai_api_key=None, model="gpt-3.5-turbo-instruct", temperature=0.3):
    """Create a LlamaIndex language model."""
    if openai_api_key is None:
        openai_api_key = load_openai_api_key()
    return OpenAI(
        model=model,
        temperature=temperature,
        openai_api_key=openai_api_key,
    )


def create_chat_model(
    openai_api_key=None, model_name="gpt-4", temperature=0, max_tokens=200
):
    """
    创建并返回一个 OpenAI Chat 模型实例。

    参数：
    openai_api_key (str): OpenAI API 密钥。如果未提供，将调用 `load_openai_api_key()` 加载。
    model_name (str): 模型名称。默认值为 "gpt-4"。
    temperature (float): 用于生成文本的温度值。默认值为 0。较高的温度值会使生成的文本更具创造性和随机性，较低的温度值会使生成的文本更具确定性。
    max_tokens (int): 生成的最大 token 数。默认值为 200。

    返回值：
    ChatOpenAI: 配置好的 OpenAI Chat 模型实例。
    """
    if openai_api_key is None:
        openai_api_key = load_openai_api_key()
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=openai_api_key,
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
        llm = create_chat_model(model_name="gpt-4o")
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


def create_output_parser(response_schemas: List[ResponseSchema]):
    """
    创建一个输出解析器对象，根据提供的响应模式列表生成解析器。

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


def create_prompt_from_template(template: str, format_instructions=None):
    """
    根据模板和格式说明创建一个提示模板对象。

    参数：
        template (str): 提示模板字符串。
        format_instructions (str, optional): 包含格式说明的字符串。如果提供，则会添加到提示模板中。

    返回：
        PromptTemplate: 创建的提示模板对象。
    """
    from langchain.prompts import PromptTemplate

    if format_instructions is None:
        return PromptTemplate.from_template(template=template)
    else:
        return PromptTemplate.from_template(
            template=template,
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
    system_template: str = "你是一个友好的AI助手",
    human_template: str = None,
    ai_template: str = None,
):
    """
    根据提供的模板创建聊天提示模板。

    Args:
    - system_template (str): 系统消息的模板，默认为 "你是一个友好的AI助手"。
    - human_template (str): 人类消息的模板，可选。
    - ai_template (str): AI 消息的模板，可选。

    Returns:
    - ChatPromptTemplate: 创建的聊天提示模板实例。
    """

    messages = []

    if not isNullOrEmpty(system_template):
        system_message_prompt = SystemMessagePromptTemplate.from_template(
            system_template
        )
        if system_message_prompt:
            messages.append(system_message_prompt)

    if not isNullOrEmpty(human_template):
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
        if human_message_prompt:
            messages.append(human_message_prompt)

    if not isNullOrEmpty(ai_template):
        ai_message_prompt = AIMessagePromptTemplate.from_template(ai_template)
        if ai_message_prompt:
            messages.append(ai_message_prompt)

    return ChatPromptTemplate.from_messages(messages)
