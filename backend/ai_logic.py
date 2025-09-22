# backend/ai_logic.py
import pandas as pd
from typing import Optional
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from core import get_llm, get_vector_store # Note the simple import

def get_sql_with_rag(user_question: str, session_context: Optional[str]):
    llm, vector_store = get_llm(), get_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={'k': 1})
    
    persistent_context = "\n\n".join([doc.page_content for doc in retriever.invoke(user_question)])
    full_context = persistent_context
    if session_context:
        full_context += "\n\n" + session_context
    
    prompt_template = """
    Your task is to write a single, valid PostgreSQL query. Only output the raw SQL query.
    Use the provided context and schema. When joining, MUST use aliases ('p' for profiles, 'm' for measurements).
    Context: {context}
    Schema with Aliases:
    - p.float_wmo_id (INT), p.profile_id (INT), p.latitude (FLOAT), p.longitude (FLOAT), p.date (TIMESTAMP)
    - m.float_wmo_id (INT), m.profile_id (INT), m."PRES" (FLOAT), m."TEMP" (FLOAT), m."PSAL" (FLOAT)
    Note: "PRES", "TEMP", "PSAL" columns from 'm' must be in double quotes.
    User Question: {question}
    SQL Query:
    """
    prompt = PromptTemplate.from_template(prompt_template)
    sql_chain = prompt | llm | StrOutputParser()
    raw_sql = sql_chain.invoke({"question": user_question, "context": full_context})
    return raw_sql.replace("```sql", "").replace("```", "").strip()

def get_summary_from_ai(user_question: str, result_df: pd.DataFrame):
    if result_df.empty: return "I couldn't find any data that matched your query."
    llm = get_llm()
    data_string = result_df.to_string(index=False, max_rows=5)
    prompt_template = """
    You are a friendly oceanographer's assistant. Provide a short, conversational summary of the retrieved data.
    Original Question: "{question}"
    Retrieved Data: {data}
    Summary:
    """
    prompt = PromptTemplate.from_template(prompt_template)
    summary_chain = prompt | llm | StrOutputParser()
    return summary_chain.invoke({"question": user_question, "data": data_string})

def get_chart_type_from_ai(user_question: str, df_columns: list) -> str:
    llm = get_llm()
    prompt_template = """
    You are a data visualization expert. What is the single best chart type for the user's question?
    User Question: "{question}"
    Available Columns: {columns}
    Your answer MUST be one of: 'depth_time_plot', 'profile_comparison', 'map', 'table'.
    Chart Type:
    """
    prompt = PromptTemplate.from_template(prompt_template)
    chart_chain = prompt | llm | StrOutputParser()
    return chart_chain.invoke({"question": user_question, "columns": df_columns}).strip().lower()