# rag/router.py
from typing import List, Literal, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


#from langchain_openai import ChatOpenAI, OpenAIEmbeddings


# 1. Pydantic Schema

"""
User: "Was besagt § 622 BGB zur Kündigungsfrist?"
{
  "route": "specific_norm",
  "search_query": "Kündigungsfristen Arbeitsverhältnis",
  "filters": ["622"]
}
User: "Ich wurde gekündigt wie in § 622 beschrieben, aber mein Chef hat mir den Resturlaub gestrichen, was nun?"
{
  "route": "concept_search",
  "search_query": "Kündigung Resturlaub gestrichen Anspruch",
  "filters": [] 
}
"""


class RouterDecision(BaseModel):
    route: Literal["specific_norm", "concept_search"] = Field(
        description=(
            "Choose 'specific_norm' if the user explicitly wants to know the content or wording of a specific paragraph. "
            "Choose 'concept_search' if the user describes a case, a legal problem, an interpretation, or a general area of law."
        )
    )
    search_query: str = Field(
        description="The search phrase optimized for retrieval (keywords extracted, stop words removed)."
    )
    paragraph_filters: Optional[List[str]] = Field(
    default=None,
    description=(
        "List of paragraph numbers without the '§' sign (e.g., ['535', '566']). "
        "ONLY used if route='specific_norm'. Otherwise None or empty. "
        "CRITICAL: If a paragraph contains a letter suffix (e.g., '556g' or '556d'), "
        "ONLY extract the full string with the letter ('556g'). Do NOT additionally extract the base number ('556')."
    )
    )





# 2. Router Chain Builder
# rag/router.py


def build_llm_router(llm: ChatOpenAI):
    """
    Creates the router chain using the passed LLM instance.
    The prompt is in English and secured with few-shot examples for German legal text.
    """
    structured_llm = llm.with_structured_output(RouterDecision)

    system_prompt = (
        "You are a specialized legal search router for German tenancy law (Deutsches Mietrecht).\n"
        "Your task is to analyze the user's question and choose the optimal routing strategy.\n\n"
        
        "Routing Rules:\n"
        "- 'specific_norm': Choose this if the user explicitly wants to see the exact content, wording, or definition of a specific paragraph (e.g., 'Was steht in § 535?').\n"
        "- 'concept_search': Choose this if the user describes a factual scenario, asks a general legal question, or asks for an evaluation/interpretation – even if paragraphs are mentioned as context.\n\n"
        
        "Critical Extraction Rules for 'paragraph_filters':\n"
        "1. Do NOT include the '§' symbol or words like 'Paragraph'. Only extract the identifier.\n"
        "2. If a paragraph has a letter suffix (e.g., '556g', '573cc'), extract ONLY the full string with the letter. "
        "Do NOT additionally extract the base number (e.g., from '§ 556g' extract ONLY '556g', NOT '556').\n\n"
        
        "Examples for Training:\n"
        "Question: 'Ich möchte § 556g und § 556d sehen'\n"
        "Expected Output: route='specific_norm', paragraph_filters=['556g', '556d']\n\n"
        
        "Question: 'Darf der Vermieter die Miete um 20% erhöhen laut § 558?'\n"
        "Expected Output: route='concept_search', paragraph_filters=None (or empty)\n\n"
        
        "User Question (German): 'Was muss ich bei einer Eigenbedarfskündigung nach § 573 beachten?'\n"
        "Expected Output: route='concept_search', paragraph_filters=None\n\n"
        
        "Question: 'Zeige mir den Wortlaut von Paragraph 573 BGB.'\n"
        "Expected Output: route='specific_norm', paragraph_filters=['573']"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])

    return prompt | structured_llm






    