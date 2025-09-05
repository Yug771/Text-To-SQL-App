# 🧠 Text-to-SQL App

This Streamlit application allows users to interact with a SQL database using natural language. Powered by Google's Gemini LLM, it translates user questions into SQL queries, executes them against a `Chinook.db` database, and displays the results in an interactive table.

## ✨ Features

*   **Natural Language to SQL:** Convert plain English questions into valid SQL queries.
*   **Gemini LLM Integration:** Utilizes Google's `gemini-2.5-flash` model for intelligent query generation.
*   **Chinook Database:** Comes pre-configured to work with the popular `Chinook.db` sample database (must be present locally).
*   **Query Execution & Display:** Executes the generated SQL and displays the results as a Pandas DataFrame.
*   **Streamlit UI:** User-friendly web interface for seamless interaction.

## 🚀 Getting Started

Follow these steps to set up and run the application locally.

### Prerequisites

*   Python 3.8+
*   `pip` (Python package installer)

### 1. Clone the Repository (or save the code)

First, save the provided Python code as `app.py` in a new directory.

### 2. Install Dependencies

Navigate to your project directory in the terminal and install the required Python packages:

```bash
pip install -r requirements.txt