import datetime
from dotenv import load_dotenv

load_dotenv()

from langchain_core.output_parsers import (
    JsonOutputKeyToolsParser,
    PydanticOutputParser,
)
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama

