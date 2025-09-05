import os
import streamlit as st
import pandas as pd
import sqlite3

from langchain_community.utilities import SQLDatabase
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from typing_extensions import TypedDict, Annotated
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from sqlalchemy import create_engine

# -----------------------------
# Setup
# -----------------------------
st.set_page_config(page_title="Text-to-SQL", layout="wide")

# Cloud secret handling
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# DB connection (Chinook.db must be in the working dir)
db = SQLDatabase.from_uri("sqlite:///Chinook.db")
conn = sqlite3.connect("Chinook.db")
engine = create_engine("sqlite:///Chinook.db")

# LLM init (Gemini)
llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

# -----------------------------
# Prompt template (same as notebook)
# -----------------------------
system_message = """
Given an input question, create a syntactically correct {dialect} query to
run to help find the answer. Unless the user specifies in his question a
specific number of examples they wish to obtain, always limit your query to
at most {top_k} results. You can order the results by a relevant column to
return the most interesting examples in the database.

Never query for all the columns from a specific table, only ask for a the
few relevant columns given the question.

Pay attention to use only the column names that you can see in the schema
description. Be careful to not query for columns that do not exist. Also,
pay attention to which column is in which table.

Only use the following tables:
{table_info}
"""

user_prompt = "Question: {input}"

query_prompt_template = ChatPromptTemplate(
    [("system", system_message), ("user", user_prompt)]
)

# -----------------------------
# State classes (as in notebook)
# -----------------------------
class State(TypedDict):
    question: str
    query: str
    result: str
    answer: str

class QueryOutput(TypedDict):
    """Generated SQL query."""
    query: Annotated[str, ..., "Syntactically valid SQL query."]

# -----------------------------
# Functions (from notebook)
# -----------------------------
def write_query(state: State):
    """Generate SQL query to fetch information."""
    prompt = query_prompt_template.invoke(
        {
            "dialect": db.dialect,
            "top_k": 10,
            "table_info": db.get_table_info(),
            "input": state["question"],
        }
    )
    structured_llm = llm.with_structured_output(QueryOutput)
    result = structured_llm.invoke(prompt)
    return {"query": result["query"]}

def execute_query(state: State):
    """Execute SQL query."""
    execute_query_tool = QuerySQLDatabaseTool(db=db)
    return {"result": execute_query_tool.invoke(state["query"])}

def print_df(state: State):
    """Convert executed SQL results to DataFrame."""
    result = state.get("result")
    query = state.get("query")

    if isinstance(result, list):
        # Result is already a list of tuples from an executed query.
        # Use a database cursor to get the column names.
        if not query:
            raise ValueError("Query must be provided to determine column names.")
        
        # Get column names by executing a 'dry run' of the query.
        # This approach is database-specific. Here's an example using SQLAlchemy.
        with conn.begin() as transaction:
            cursor = transaction.connection.connection.cursor()
            cursor.execute(f"EXPLAIN {query}")  # Or a similar dry-run command
            column_names = [desc[0] for desc in cursor.description]
            cursor.close()

        return pd.DataFrame(result, columns=column_names)
    
    elif isinstance(query, str):
        # A raw SQL string was passed. Let pandas handle the query execution.
        if conn:
            return pd.read_sql_query(query, conn)
        else:
            raise ValueError("Connection object (`conn`) must be provided for raw SQL query.")
    
    else:
        raise TypeError("Invalid input: 'result' must be a list of tuples or 'query' must be a string.")



# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🧠 Text-to-SQL App")

question = st.text_input("Ask a question:", placeholder="e.g., list of songs and artists")

if st.button("Run"):
    if question.strip():
        with st.spinner("Executing SQL..."):
            sql_query = write_query({"question": question})["query"]
            result = execute_query({"query": sql_query})["result"]

        st.subheader("Generated SQL Query")
        st.code(sql_query, language="sql")

        df = print_df({"query": sql_query, "result": result})

        st.subheader("Query Results")
        st.dataframe(df, use_container_width=True)
