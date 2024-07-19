import streamlit as st
import json
import psycopg2
import os
from dotenv import load_dotenv
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

# Load environment variables from credentials.env
load_dotenv('credentials.env')

# Function to establish connection with PostgreSQL database using psycopg2
def connect_to_db():
    try:
        conn = psycopg2.connect(
            host=os.getenv('DB_HOST'),
            port=os.getenv('DB_PORT'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            database=os.getenv('DB_NAME')
        )
        return conn
    except Exception as e:
        st.write(f"Error connecting to database: {str(e)}")
        return None

# Function to fetch table and column information from the database schema
def get_db_schema(conn, table_name):
    cursor = conn.cursor()
    cursor.execute(f"""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema='public' AND table_name='{table_name}'
    """)
    schema_info = [row[0] for row in cursor.fetchall()]
    cursor.close()
    return schema_info

# Function to create SQL chain with quoted column names
def create_sql_chain(conn, target_table, question):
    schema_info = get_db_schema(conn, target_table)
    quoted_schema_info = [f'"{col}"' if col in ['A', 'B', 'C', 'D', 'E', 'F', 'G'] else col for col in schema_info]

    template = f"""
        Based on the table schema of table '{target_table}', write a SQL query to answer the question.
        If the question is on the dataset, the SQL query must take all the columns into consideration.
        Only provide the SQL query, without any additional text or characters.
        Ensure the query includes the table name {target_table} in the FROM clause.

        If user asks a question related to correlation between 2 columns and if either column is non-numeric, display an appropriate message instead of calculating the correlation.
        If the question is related to the column names in the dataset, please refer the schema {quoted_schema_info} to identify the column names.

        If the question involves complex conditions or calculations, consider using SQL subqueries, CASE statements, and CTEs (Common Table Expressions) to ensure the query is efficient and clear.
        Provide subqueries for nested queries when needed to filter or aggregate data before the main query.
        Use CTEs for better readability and organization of complex queries, especially when multiple steps are involved.
        Apply CASE statements to handle conditional logic directly within the query results.

        Examples of SQL constructs to use when appropriate:
        - Subqueries: SELECT * FROM (SELECT ... FROM ...) AS subquery;
        - CTEs: WITH cte_name AS (SELECT ... FROM ...) SELECT * FROM cte_name;
        - CASE statements: SELECT CASE WHEN condition THEN result ELSE other_result END AS column_name;

        Specific columns to refer to based on the question context:
        - Coverage level: column 'A'
        - Smoking type: column 'B'
        - Car type: column 'C'
        - Purpose of the vehicle: column 'D'
        - Safety features: column 'E'
        - Driver's historic record: column 'F'
        - Area where the user will drive the car (rural, urban, suburban or hazardous): column 'G'
        - Previous car type: column 'c_previous'
        
        Week days:
        - 0: Monday
        - 1: Tuesday
        - 2: Wednesday
        - 3: Thursday
        - 4: Friday
        - 5: Saturday
        - 6: Sunday
        
        Car values ranges:
        a - Below $10,001
        b - $10,001 - $20,000
        c - $20,001 - $30,000
        d - $30,001 - $40,000
        e - $40,001 - $50,000
        f - $50,001 - $75,000
        g - $75,001 - $1,00,000
        h - $1,00,000 - $1,50,000
        i = $1,50,001 and above

        If there comes a question which is unrelated to the table {target_table} and is based on general knowledge, answer the question based on your knowledge rather than throwing errors.
        If you are unable to solve very complex questions or it is not possible to get answers based on SQL queries then please answer politely rather than giving errors.
        
        Table schema: {quoted_schema_info}
        Question: {question}

        SQL Query:
    """

    prompt = ChatPromptTemplate.from_template(template=template)
    llm = ChatGroq(model="llama3-8b-8192", temperature=0.2, groq_api_key=os.getenv('InfoQuest_API_KEY'))

    chain = (
        RunnablePassthrough(assignments={"schema": quoted_schema_info, "question": question})
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# Function to execute SQL query and fetch results
def execute_query(conn, query):
    try:
        cursor = conn.cursor()
        cursor.execute(query)
        colnames = [desc[0] for desc in cursor.description]  # Get column names
        result = cursor.fetchall()
        cursor.close()
        return colnames, result
    except Exception as e:
        return "This is out of my knowledge. Please feel free to ask me if you have any other questions related to the dataset!"
        #return None, f"Error: {str(e)}"

# Function to create natural language response based on SQL query results
def create_nlp_answer(sql_query, results, question):
    results_str = "\n".join([str(row) for row in results])

    template = f"""
        Based on the results of the SQL query '{sql_query}', write a natural language response.
        Consider the initial {question} while generating the output in natural language.
        Do not write "Based on the SQL query results" or "Therefore, the natural language response would be:" in the response.
        The response should not contain the entire question asked.
        If you are unable to solve very complex questions or it is not possible to get answers based on SQL queries then please answer politely rather than giving errors.

        Query Results:
        {results_str}
    """
    prompt = ChatPromptTemplate.from_template(template=template)
    llm = ChatGroq(model="llama3-8b-8192", temperature=0.2, groq_api_key=os.getenv('InfoQuest_API_KEY'))

    return (
        RunnablePassthrough(assignments={"sql_query": sql_query, "results": results_str, "question": question})
        | prompt
        | llm
        | StrOutputParser()
    )

# Function to send query to PostgreSQL database and retrieve response
def send_query(question, history):
    try:
        conn = connect_to_db()
        if conn:
            target_table = os.getenv('TABLE_NAME')
            sql_chain = create_sql_chain(conn, target_table, question)
            sql_query_response = sql_chain.invoke({})
            sql_query = sql_query_response.strip()

            colnames, results = execute_query(conn, sql_query)
            if colnames and results:
                nlp_chain = create_nlp_answer(sql_query, results, question)
                nlp_response = nlp_chain.invoke({})

                history.append((question, "User"))
                history.append((nlp_response, "Bot"))

                save_session_history(history)
                conn.close()

                return {'response': nlp_response, 'results': {'colnames': colnames, 'data': results}}
            else:
                conn.close()
                return {'response': "I do not have enough data to answer your question. Please try a different question."}
        else:
            return {'response': "I am unable to connect to the database."}
    except Exception as e:
        return {'response': "Oops! Something went wrong. Please try again later."}

# Function to save session history
def save_session_history(history):
    with open('session_history.json', 'w') as f:
        json.dump(history, f)

# Function to load session history
def load_session_history():
    if os.path.exists('session_history.json'):
        with open('session_history.json', 'r') as f:
            return json.load(f)
    return []

# Main function to run the Streamlit app
def main():
    st.set_page_config(layout="wide")  # Set wide layout

    st.title("InfoQuest")
    st.markdown(
        """
        <style>
        body { background-color: offwhite; }
        .message-container { width: 110%; margin-bottom: 10px; }
        .user-message { width: 60%; background-color: #EBECF0; padding: 10px; border-radius: 10px; text-align: right; color: black; }
        .bot-message { width: 60%; background-color: #EBECF0; padding: 10px; border-radius: 10px; text-align: left; color: black; }
        .label { width: 60%; font-weight: bold; margin-bottom: 5px; }
        .sidebar-left { width: 25%; position: fixed; left: 0; top: 0; height: 100%; background-color: #F8F9FA; padding: 10px; }
        .main-content { margin-left: 25%; margin-right: 25%; padding: 10px; }
        .sidebar-right { width: 25%; position: fixed; right: 0; top: 0; height: 100%; background-color: #F8F9FA; padding: 0px; text-align: centre; }
        </style>
        """,
        unsafe_allow_html=True
    )

    if 'session_history' not in st.session_state:
        st.session_state.session_history = load_session_history()

    history = st.session_state.session_history

    # Left sidebar for Chat History
    with st.sidebar:
        st.title("Chat History")
        questions = [msg for msg, sender in history if sender == "User"]

        selected_question = None
        for i, question in enumerate(questions):
            if st.button(question, key=f"history_{i}"):
                selected_question = i

    # Main content for input and responses
    st.markdown('<div class="main-content">', unsafe_allow_html=True)

    if 'response' in locals():  # Check if response is defined
        if 'response' in response:
            st.markdown(f'<div class="label">Bot:</div><div class="message-container"><div class="bot-message">{response["response"]}</div></div>', unsafe_allow_html=True)
        else:
            st.write(f"Response received: {response}")
    # else:
    #     st.write("No response to display yet.")

    if selected_question is not None:
        selected_history = history[selected_question*2:]  # Each user message has a bot response, so step by 2
        for i, (msg, sender) in enumerate(selected_history):
            if sender == "User":
                st.markdown(f'<div class="label">You:</div><div class="message-container"><div class="user-message">{msg}</div></div>', unsafe_allow_html=True)
            elif sender == "Bot":
                st.markdown(f'<div class="label">Bot:</div><div class="message-container"><div class="bot-message">{msg}</div></div>', unsafe_allow_html=True)

    question = st.text_input("Type your question here:")

    if st.button("Send"):
        if question:
            response = send_query(question, st.session_state.session_history)
            if isinstance(response, dict):
                if 'response' in response:
                    st.markdown(f'<div class="label">You:</div><div class="message-container"><div class="user-message">{question}</div></div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="label">Bot:</div><div class="message-container"><div class="bot-message">{response["response"]}</div></div>', unsafe_allow_html=True)
                else:
                    st.write(f"Response: {response}")
            else:
                st.write("Unexpected response format.")

    # Display all chat history
    for i, (msg, sender) in enumerate(st.session_state.session_history):
        if sender == "User":
            st.markdown(f'<div class="label">You:</div><div class="message-container"><div class="user-message">{msg}</div></div>', unsafe_allow_html=True)
        elif sender == "Bot":
            st.markdown(f'<div class="label">Bot:</div><div class="message-container"><div class="bot-message">{msg}</div></div>', unsafe_allow_html=True)

    save_session_history(st.session_state.session_history)  # Save history at the end of each session

    st.markdown('</div>', unsafe_allow_html=True)

    # Right sidebar for Dataset Description
    st.markdown(
        """
        <div class="sidebar-right">
        <h2>Dataset Description</h2>
        <b>Customer ID</b> - A unique identifier for the customer<br>
        <b>Shopping Point</b> - Unique identifier for the shopping point of the customer<br>
        <b>Record Type</b> - The record type (0: shopping point, 1: purchase point)<br>
        <b>Day</b> - Day of the week when the shopping point was created<br>
        <b>Time</b> - Time of the day when the shopping point was created<br>
        <b>State</b> - The state where the shopping point was created<br>
        <b>Location Coordinate</b> - The location where the shopping point was created<br>
        <b>Group Size</b> - The size of the group the customer is shopping for<br>
        <b>Homeowner</b> - Whether the customer is a homeowner or not<br>
        <b>Car Age</b> - The age of the customer's car<br>
        <b>Car Value</b> - Value of the customer's car at purchase time<br>
        <b>Risk Factor</b> - The risk factor assigned to the customer<br>
        <b>Age Oldest</b> - Age of the oldest person in the customer's group<br>
        <b>Age Youngest</b> - Age of the youngest person in the customer's group<br>
        <b>Married Couple</b> - Indicates whether the group includes a married couple<br>
        <b>C Previous</b> - Previous Car Type<br>
        <b>Duration Previous</b> - The duration of the customer's previous insurance policy<br>
        <b>A</b> - Coverage level<br>
        <b>B</b> Smoking type<br>
        <b>C</b> - Car type<br>
        <b>D</b> - Purpose of the vehicle<br>
        <b>E</b> - Safety features<br>
        <b>F</b> - Driver's historic record<br>
        <b>G</b> - Area where the user will drive the car (rural, urban, suburban or hazardous)<br>      
        <b>Cost</b> - The cost of the insurance policy
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
