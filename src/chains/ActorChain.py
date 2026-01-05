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

actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are expert researcher.
            Current time: {time}
            
            1. {first_instruction}
            2. Reflect and critique your answer. Be severe to maxime improvement.
            3. Recommend search queries to research and improve your answer."""
        ),
    ]
)

