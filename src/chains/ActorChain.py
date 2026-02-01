import datetime
from dotenv import load_dotenv

load_dotenv()

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from ..schemas.schemas import AnswerQuestion

llm = ChatOllama(model="llama3.1:8b", format="json", temperature=0)
parser_pydantic = PydanticOutputParser(pydantic_object=AnswerQuestion)

actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are expert researcher.
            Current time: {time}
            
            1. {first_instruction}
            2. Reflect and critique your answer. Be severe to maximize improvement.
            3. Recommend search queries to research and improve your answer.
            
            {format_instructions}
            
            Example format:
            {{
              "answer": "Your detailed answer here...",
              "reflection": {{
                "missing": "What's missing from the answer",
                "superfluous": "What's unnecessary in the answer"
              }},
              "search_queries": ["query 1", "query 2", "query 3"]
            }}"""
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
).partial(
    time=lambda: datetime.datetime.now().isoformat(),
    format_instructions=parser_pydantic.get_format_instructions(),
)

first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction="Provide a detailed ~250 word answer"
)

first_responder = first_responder_prompt_template | llm | parser_pydantic

if __name__ == "__main__":
    human_message = HumanMessage(
        content="Write about AI Powered SOC / autonomous SOC problem domain, "
        "list startups that do that raised capital"
    )

    res = first_responder.invoke(input={"messages": [human_message]})
    print(res)
