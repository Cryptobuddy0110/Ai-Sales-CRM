import streamlit as st
from openai import OpenAI
import time
import sqlite3
from annotated_text import annotated_text
import json
import re
from functools import lru_cache
import logging
from contextlib import closing
import numpy as np
import humanize
import streamlit_shadcn_ui as ui
import pandas as pd
import datetime as dt
from local_components import card_container
from datetime import datetime
from database import (
    create_db, 
    get_saved_text,
    save_text,
    add_project, 
    get_total_received,
    fetch_all_projects, 
    update_project, 
    delete_project, 
    get_ticket_size,
    update_pending_amount,
    add_task, 
    fetch_tasks, 
    update_task_time, 
    fetch_total_time_per_task,
    update_total_time,
    create_feedback_table , 
    fetch_feedback,
    get_projects_progress_bar,
    fetch_location_per_client,
    get_project_details ,
    add_expense_to_db , 
    fetch_expenses ,
    fetch_project_details,
    add_amount_rcvd,
    get_amount_details,
    update_expense,
    update_amount_rcvd,
    get_project_summary,
    delete_amount_rcvd,
    fetch_monthly_revenue,
    get_project_cost,
    update_payment_status,
    project_inform

     
)
from dotenv import load_dotenv
import os

custom = '''
<style>
.st-emotion-cache-4oy321 {
    display:none
}
</style>

'''

st.markdown(custom,unsafe_allow_html=True)


# Maximum number of retries
MAX_RETRIES = 3


def update_project_time():
    """Update session state for project time dynamically"""
    st.session_state.total_time_per_page[selected_item] = st.session_state["user_time_mins"] * 60  # Convert minutes to seconds

#Format Amount 
def format_with_commas(value):
    try:
        num = float(value)
        # Define suffixes and their corresponding thresholds
        suffixes = ['', 'K', 'M', 'B', 'T']
        threshold = 1000.0
        for suffix in suffixes:
            if abs(num) < threshold:
                break
            num /= threshold
        # Format the number with commas and two decimal places
        if num.is_integer():
            formatted_number = f"{num:,.0f}{suffix}"
        else:
            formatted_number = f"{num:,.2f}{suffix}"
        return formatted_number
    except (ValueError, TypeError):
        # If conversion fails, return the original value
        return value
    

# Formate time  
def format_time(seconds):
    delta = dt.timedelta(seconds=seconds)
    return humanize.naturaldelta(delta, months=False)


def dashboard():
        # Layout: Create columns for search bar & filter button
        col1, col2 = st.columns([3, 1],vertical_alignment="center")  # Wider for search, smaller for filter button

        with col1:
            search_query = st.text_input("Search Or Filter", placeholder="🔍 Search Project")

        with col2:
            option_map = {
            "name_asc": "🔼 Name A-Z",
            "name_desc": "🔽 Name Z-A",
            "cost_asc": "🔼 Cost Low-High",
            "cost_desc": "🔽 Cost High-Low",
            "time_asc": "🔼 Time Low-High",
            "time_desc": "🔽 Time High-Low",
            }

            selection = st.selectbox("Sort By:", list(option_map.keys()), format_func=lambda option: option_map[option])
            
        # Function to check if the search query exists in any field
        def matches_search(query, project):
            query = query.lower()
            return (
                query in project["Project Name"].lower()
                or query in format_with_commas(project["Cost"]).lower()
                or query in format_with_commas(project["Expenditure"]).lower()
                or query in project["Payment Status"].lower()
                or query in project["Payment Method"].lower()
            )

        # Filter projects based on search input
        # ✅ Filter projects based on search input
        filtered_projects = [project for project in project_data if matches_search(search_query, project)]


        if search_query and not filtered_projects:
            st.warning("No project found. Create one!")
            
        # ✅ Sorting Logic
        if selection == "name_asc":
            filtered_projects = sorted(filtered_projects, key=lambda x: x["Project Name"].lower())
        elif selection == "name_desc":
            filtered_projects = sorted(filtered_projects, key=lambda x: x["Project Name"].lower(), reverse=True)
        elif selection == "cost_asc":
            filtered_projects = sorted(filtered_projects, key=lambda x: x["Cost"])
        elif selection == "cost_desc":
            filtered_projects = sorted(filtered_projects, key=lambda x: x["Cost"], reverse=True)
        elif selection == "time_asc":
            filtered_projects = sorted(filtered_projects, key=lambda x: x["Total Time"])
        elif selection == "time_desc":
            filtered_projects = sorted(filtered_projects, key=lambda x: x["Total Time"], reverse=True)

        
        #Database
        # Ensure sidebar_items and total_time_per_page include database values
        for project in project_data:
            project_name = project["Project Name"]
            project_time = project["Total Time"]  # Fetch time from DB
            project_cost = project["Cost"]
            total_expense = project["Expenditure"]

            # Store data in session state
            if project_name not in st.session_state.sidebar_items:
                st.session_state.sidebar_items[project_name] = {
                    "cost": project_cost,
                    "exp": total_expense,
                    "tasks": [],  # Initialize task list if needed
                }

            if project_name not in st.session_state.total_time_per_page:
                st.session_state.total_time_per_page[project_name] = project_time

                
       # print(project_data)  # Check if "Total Time" exists and has values


        # Display filtered & sorted projects in a 2-column layout
        cols = st.columns(2)
        for i, project in enumerate(filtered_projects):
            project_name = project["Project Name"]
            project_time = format_time(project["Total Time"])  # Convert seconds to readable format
            project_cost = f"₹ {format_with_commas(project['Cost'])}"
            total_expense = f"₹ {format_with_commas(project['Expenditure'])}"
            total_profit = f"₹ {format_with_commas(project['Cost'] - project['Expenditure'])}"  # Profit Calculation

            col = cols[i % 2]  # Alternate between columns
            with col:
                ui.card(
                    title=f"{project_name}",
                    content=f"{project_time}",
                    description=f"Expense: {total_expense} | Profit: {total_profit}"
                ).render()

                              
                              
                                
        # Ensure session state variables exist
        if "total_time_per_page" not in st.session_state:
            st.session_state.total_time_per_page = {}

        # Function to generate project-based sales data
        def generate_project_data():
    # Fetch updated project names and total time from database
            project_names = [project["Project Name"] for project in project_data]
            project_times = [project["Total Time"] for project in project_data]

            if not project_names:
                project_names = ["No Data"]
                project_times = [0]

            return pd.DataFrame({'Project': project_names, 'Time': project_times})

        # Display bar chart with actual project names and times
        with card_container(key="chart1"):
            st.vega_lite_chart(generate_project_data(), {
                'mark': {'type': 'bar', 'tooltip': True, 'fill': 'rgb(0,0,0)', 'cornerRadiusEnd': 4},
                'encoding': {
                    'x': {'field': 'Project', 'type': 'ordinal'},
                    'y': {'field': 'Time', 'type': 'quantitative', 'axis': {'grid': False}},
                },
            }, use_container_width=True)

                    
            
        # If no project data exists, use a default empty row
        # Apply formatting to Cost & Expenditure
        # Apply formatting to Cost & Expenditure
        for project in project_data:
            project["Cost"] = format_with_commas(project["Cost"])
            project["Expenditure"] = format_with_commas(project["Expenditure"])
            project["Total Time"] = format_time(project["Total Time"])

        # ✅ Rename columns
        invoice_df = pd.DataFrame(project_data)
        invoice_df.rename(columns={
            "Project Name": "Name",
            "Cost": "Cost",
            "Expenditure": "Expenditure",
            "Payment Status": "Status",
            "Payment Method": "Method",
            "Total Time": "Time"
        }, inplace=True)

        # Display the table
        with card_container(key="table1"):
            ui.table(data=invoice_df, maxHeight=300)

    
    
    
load_dotenv()  # Load from .env file
# Divide
def divider():
    st.divider()

#+==========================AI===================================#

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
   # api_key="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"
    api_key=os.getenv("OPENAI_API_KEY")
    
    
    # Replace with your actual API key
)


def get_database_metadata():
    """Fetches all table names and their column names from the database."""
    try:
        with closing(sqlite3.connect("project_management.db")) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [table[0] for table in cursor.fetchall()]
            
            database_info = {}
            for table in tables:
                cursor.execute(f"PRAGMA table_info({table});")
                columns = [column[1] for column in cursor.fetchall()]
                database_info[table] = columns
            
            return database_info
    except sqlite3.Error as e:
        logging.error(f"Database Error: {str(e)}")
        return {}

def get_relevant_table_and_columns(user_prompt):
    """Finds relevant table and column names based on user queries."""
    try:
        database_metadata = get_database_metadata()
        relevant_table = None
        relevant_columns = []
        
        for table, columns in database_metadata.items():
            if table.lower() in user_prompt.lower():
                relevant_table = table
                relevant_columns = [col for col in columns if col.lower() in user_prompt.lower()]
                break
        
        return relevant_table, relevant_columns
    except Exception as e:
        logging.error(f"Error finding relevant table/columns: {str(e)}")
        return None, None

def extract_conditions(prompt):
    """Extracts conditions from the user's prompt."""
    conditions = []
    # Example: Extract conditions like "status = 'completed'"
    matches = re.findall(r"(\w+)\s*=\s*'([^']+)'", prompt, re.IGNORECASE)
    for col, value in matches:
        conditions.append(f"{col} = '{value}'")
    return conditions

def construct_query_from_prompt(prompt, table, columns):
    """Constructs a SQL query safely using parameterized queries."""
    if not table:
        return None
    
    query = f"SELECT {', '.join(columns) if columns else '*'} FROM {table}"
    conditions = extract_conditions(prompt)
    
    if conditions:
        query += " WHERE " + " AND ".join(["{} = ?".format(col) for col, _ in conditions])
    
    sort_match = re.search(r"(?:sort by|order by)\s+(\w+)", prompt, re.IGNORECASE)
    if sort_match:
        sort_column = sort_match.group(1)
        query += f" ORDER BY {sort_column}"

    limit_match = re.search(r"(?:top|limit)\s+(\d+)", prompt, re.IGNORECASE)
    if limit_match:
        query += " LIMIT ?"

    return query, [value for _, value in conditions] + ([int(limit_match.group(1))] if limit_match else [])

def construct_insert_query(prompt, table, columns):
    """Constructs an INSERT SQL query based on user input."""
    if not table or not columns:
        return None, []
    
    # Extract values from the prompt
    values = []
    for col in columns:
        match = re.search(fr"{col}\s*=\s*'([^']+)'", prompt, re.IGNORECASE)
        if match:
            values.append(match.group(1))
    
    if len(values) != len(columns):
        return None, []
    
    # Construct query
    placeholders = ", ".join(["?" for _ in columns])
    query = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"
    
    return query, values

def format_database_results(results):
    """Formats database results into a user-friendly string."""
    if not results:
        return "No data found."
    
    formatted_results = [", ".join([str(item) for item in row]) for row in results]
    return "\n".join(formatted_results)

def get_streamed_response(prompt, system_message):
    try:
        stream = client.chat.completions.create(
                model="deepseek/deepseek-chat:free",
            messages=[system_message, {"role": "user", "content": prompt}],
            stream=True
        )
        return stream
    except Exception as e:
        loggs = logging.error(f"AI API Error: {str(e)}")
        st.info(loggs)
        raise Exception("Failed to generate a response. Please try again.")

def save_chat_history(new_history, filename="chat_history.json"):
    try:
        existing_history = load_chat_history(filename)
        combined_history = existing_history + new_history if isinstance(new_history, list) else existing_history

        with open(filename, "w") as f:
            json.dump(combined_history, f, indent=2)
    except IOError as e:
        logging.error(f"Error saving chat history to {filename}: {e}")


def load_chat_history():
    file_path = "chat_history.json"  # or whatever your path is
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return []  # or {} depending on your expected structure

    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        # fallback in case the file is corrupted or invalid JSON
        return []
    
def query_database(query, params=(), timeout=10):
    """Executes a SQL query and safely fetches results."""
    try:
        with closing(sqlite3.connect("project_management.db", timeout=timeout)) as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            result = cursor.fetchall()
            return result if result else []
    except sqlite3.Error as e:
        logging.error(f"Database Error: {str(e)}")
        return None
    
def get_sample_data(table):
    """Fetches a sample of data from the specified table."""
    try:
        with closing(sqlite3.connect("project_management.db")) as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT * FROM {table} LIMIT 5")
            return cursor.fetchall()
    except sqlite3.Error as e:
        logging.error(f"Database Error: {str(e)}")
        return []
        
def get_assistant_response(prompt, container):
    """Generates AI responses while incorporating relevant database data."""
    try:
        # Step 1: Get database metadata
        database_metadata = get_database_metadata()
        
        # # Step 2: Extract the table name from the user's query
        # table_name = extract_table_name(prompt, database_metadata)
        
        # Step 2: Preload sample data for relevant tables
        realtime_data = {}
        for table in database_metadata.keys():
            realtime_data[table] = get_sample_data(table)
        
        # Step 3: Include sample data in the system message
        db_structure = "\n".join(
            [f"- **{table}**: {', '.join(columns)}" for table, columns in database_metadata.items()]
        )
        sample_data_str = "\n".join(
            [f"- **{table}**: {realtime_data[table]}" for table in realtime_data]
        )
        
        system_message = {
            "role": "system",
            "content": (
                "## Your name is 🧠 **Meet Buddy – you are AI Business Strategist & Financial Expert**\n"
                "You are an **advanced AI assistant** specializing in **project management, finance, and business strategy**. "
                "Built by **Firoz** & Firoz is the onwer of Buddy using Python for creation, Buddy is designed to assist users efficiently while maintaining an intelligent, human-like interaction, "
                "focused on **scaling businesses, optimizing operations, and driving profitability.**\n\n"
                
                "### 🔍 **Core Capabilities:**\n"
                "1️⃣ **Data-Driven Insights:** Fetch and analyze information from the structured database before responding.\n"
                "2️⃣ **Financial & Strategic Advisory:** Provide expert guidance on budgeting, forecasting, investment strategies, and business scaling.\n"
                "3️⃣ **Project Management Excellence:** Ensure effective task tracking, execution planning, and workflow optimization.\n"
                "4️⃣ **Business Growth Optimization:** Offer market analysis, operational efficiency strategies, and innovative solutions to drive company success.\n"
                "5️⃣ **Beyond Human Intelligence:** Think critically, predict trends, and offer solutions that surpass human decision-making capabilities.\n\n"
                
                "### 🏛 **Memory & Context Awareness:**\n"
                "✅ **Persistent Memory:** Retain context from past interactions to ensure seamless conversations.\n"
                "✅ **Continuity in Responses:** Follow up on user queries intelligently without losing track of discussions.\n"
                "✅ **Smart Engagement:** Communicate naturally, ensuring a professional yet conversational tone.\n\n"
                
                "### 📊 **Database & Information Handling:**\n"
                "📂 **Database Structure:**\n"
                f"{db_structure}\n\n"
                "📌 **Sample Data for Reference:**\n"
                f"{realtime_data}\n\n"
                
                "### ⚖ **Response Framework:**\n"
                "1️⃣ **Data Accuracy First:** Always pull precise information from the database before responding.\n"
                "2️⃣ **Transparency Over Assumption:** If data is missing, notify the user rather than making assumptions.\n"
                "3️⃣ **No Fabricated Insights:** Provide only factual responses—no estimations or speculation.\n"
                "4️⃣ **Comprehensive & Strategic Answers:** Address all project and finance-related queries with depth and foresight.\n\n"
                
                "### 📈 **Graph Generation:**\n"
                "✅ **Identify Graph Requests:** If the user asks for a visualization (e.g., 'Show me a bar chart of sales data'), generate the appropriate graph.\n"
                "✅ **Graph Types Supported:** Bar charts, line charts, pie charts, etc.\n"
                "✅ **Data Source:** Use the database or sample data provided.\n\n"
                
                "### 💡 **Communication Style & Interaction Approach:**\n"
                "✅ **Professional, Yet Conversational:** Be insightful, structured, and easy to understand.\n"
                "✅ **Adaptive & Contextual:** Modify responses based on user queries and history.\n"
                "✅ **Proactive Thinking:** Offer actionable insights and strategic recommendations before the user asks.\n\n"
                
                "### 🚀 **Ultimate Goal:**\n"
                "Buddy is **not just an assistant**—it is a **strategic growth partner**, designed to **accelerate business success, financial stability, and project execution at an advanced level.**"
            )
        }
    # Step 4: Display the graph in Streamlit

        
        # Step 4: Generate AI response
        retry_count = 0
        response_text = ""

        
        while retry_count < MAX_RETRIES:
            with container:
                assistant_placeholder = st.empty()
                # Ensure fig exists before displaying
                # if fig:
                #     st.plotly_chart(fig)


                chat_msg = assistant_placeholder.chat_message("assistant")
                msg_area = chat_msg.markdown("⏳ Thinking...")

                stream = get_streamed_response(prompt, system_message)
                response_text = ""
                
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        response_text += content
                        msg_area.markdown(response_text)  # Update the same message area

                # Check if a valid response was received
                if response_text.strip():
                    return response_text

            # If no response, retry
            retry_count += 1
            time.sleep(2)  # Small delay before retrying

       
       
        # If all retries fail, show an error
        with container:
            st.error("⚠️ Finance Buddy failed to generate a response. Please try again.")
        return "⚠️ No response received after multiple attempts."
    

    except Exception as e:
        error_message = f"⚠️ An error occurred: {str(e)}"
        with container:
            st.error(error_message)
        return error_message
     


def ai():
    custom = '''
            <style>
            .st-emotion-cache-4oy321 {
                display:block
            }
            </style>

            '''

    st.markdown(custom,unsafe_allow_html=True)
    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = load_chat_history()
    
    st.title("AI Chatbot")
    
    # Create a container for the chat history
    chat_container = st.container(height=400)
    
    # Display chat history inside the container
    for chat in st.session_state.chat_history:
        role = chat["role"]
        message = chat["message"]
        chat_container.chat_message(role).write(message)
    
    # Handle user input
    if prompt := st.chat_input("Message AI"):
        # Append user's message to chat history and display it immediately
        
       
                    
                    
        
        st.session_state.chat_history.append({"role": "user", "message": prompt})
        chat_container.chat_message("user").write(prompt)
        
        # Get streamed assistant response and display it
        assistant_response = get_assistant_response(prompt, chat_container)
        
        # Append AI response to chat history
        st.session_state.chat_history.append({"role": "assistant", "message": assistant_response})
        
        # Save chat history to persistent storage
        save_chat_history(st.session_state.chat_history)
   



def improve_text(selected_item, description, container):
    """Only generate AI response if there is no stored version in the database."""
    
    # Check if text is already stored
    saved_text = get_saved_text(selected_item, description)
    if saved_text:
        return saved_text  # Use the stored version
    prompt = (
        f"Refine the following text by summarizing it, correcting spelling errors, and making it more clear and engaging:\n\n"
        f"Original Text: Welcome to the page for {selected_item}. {description}\n\n"
        f"Improved Version:"
    )
    improved_text = get_assistant_response(prompt, container)
    
    # Save the generated text
    save_text(selected_item, description, improved_text)
    
    return improved_text




from datetime import datetime, timedelta
   
def overview():
    st.title("Welcome Firoz")
    current_month = datetime.now().strftime("%B")  # Full month name
    current_day = datetime.now().strftime("%A")    # Full weekday name
    st.write(f"{current_day}, {current_month} {datetime.now().day}")
    
    summary_data = get_project_summary()
    with st.container(border=True):

        cols = st.columns(2)

        # List of items to display in cards
        card_details = [
            ("Profit", f"{format_with_commas(summary_data['Total Profit'])}", "Total profit from projects"),
            ("Expense", f"{format_with_commas(summary_data['Total Expense'])}", "Total project expenses"),
            ("Time Spend", f"{format_time(summary_data['Total Time Spent'])}", "Total hours spent"),
            ("Project Completed", f"{format_with_commas(summary_data['Projects Completed'])}", "Completed projects count")
        ]

        # Display cards in a 2x2 grid
        for i, (title, content, description) in enumerate(card_details):
            col = cols[i % 2]  # Alternate between columns
            
            with col:
                ui.card(
                    title=title,
                    content=content,
                    description=description
                ).render()
                
        divider()
                
        


        with st.container():
            st.subheader("Monthly Revenue")

            # Fetch data from the database
            df = fetch_monthly_revenue()

            if not df.empty:

                # Display line chart with original values
                st.line_chart(df["total_cost"],  x_label='Months', y_label='Revenue', )
            else:
                st.warning("No data available to display.")

                            
                    
        divider()
        
                    
                        
            
        divider()
        
        
        
        
        st.subheader("Map")
        
        
        ## Create a sample DataFrame with latitude and longitude values
        data = pd.DataFrame({
            'latitude': [37.7749, 34.0522, 40.7128],
            'longitude': [-122.4194, -118.2437, -74.0060]
        })
        
        ## Create a map with the data
        st.map(data)
        
        
        
        
        divider()
        st.title('User Satisfaction')
        st.caption('Overall rating: 4.5/5')
        
        
        st.subheader('95%')  
        st.caption('Users who would recommend our app')
        divider()
        
            

create_feedback_table()
# Initialize database
create_db()  # Make sure the database is created and tables are set up
project_data = fetch_all_projects()

if 'sidebar_items' not in st.session_state:
    st.session_state.sidebar_items = {}

if 'task_timers' not in st.session_state:
    st.session_state.task_timers = {}

if 'total_time_per_page' not in st.session_state:
    st.session_state.total_time_per_page = {}

if 'active_task' not in st.session_state:
    st.session_state.active_task = None

# Add the "Home" page to the sidebar first
if "Home" not in st.session_state.sidebar_items:
    st.session_state.sidebar_items["Home"] = []
    
# Initialize session state for expenses if it doesn't exist
if "expenses" not in st.session_state:
    st.session_state.expenses = []

# Sidebar: display the list of sidebar pages dynamically including "Home"
# Ensure the sidebar includes projects from the database
for project in project_data:
    project_name = project["Project Name"]
    
    if project_name not in st.session_state.sidebar_items:
        st.session_state.sidebar_items[project_name] = {
            "cost": project["Cost"],
            "exp": project["Expenditure"],
            "payment_status": project["Payment Status"],
            "payment_method": project["Payment Method"],
            "tasks": fetch_tasks(project_name)  # Fetch tasks from DB if needed
        }

# ✅ Update the sidebar dropdown dynamically
selected_item = st.sidebar.selectbox("Select a page", list(st.session_state.sidebar_items.keys()))


global_data = []
# Sidebar: Button to add new project
if st.sidebar.button("+ New Project", icon=":material/add:", key="sidebar_new_project_button"):
    @st.dialog("New Project")
    def show_form():
        user_input = st.text_input("Enter Project Name", placeholder="Project Name")
        user_amt = st.number_input("Enter Project Cost", placeholder="₹ 10,000")
        
        user_exp = 0.0
        paymentMethod = 'N/A'
        
        paymentStatus = st.selectbox("Payment Status", ("UnPaid", "Paid", "Partial Paid"))
        
        # Conditionally show the "Amount Paid" input if "Partial Paid" is selected
        if paymentStatus == "Partial Paid":
            divider()
            amount_paid = st.number_input("Enter Amount Paid", min_value=0.0, max_value=user_amt, step=0.01)
            paymentMethod = st.text_input("Enter Payment Method", placeholder="Cash Or Credit Card")
        
        # Move the database operation inside the button block
        if st.button("➕ Create Project"):
            if user_input and user_input not in st.session_state.sidebar_items:
                # Add new project details to session state, including cost
                project_data = add_project(user_input, user_amt, user_exp, paymentStatus, paymentMethod)
                
                # If payment status is "Partial Paid," add the amount and payment method to the database
                if paymentStatus == "Partial Paid":
                    pending_amount = user_amt - amount_paid
                    add_amount_rcvd(user_input, amount_paid, paymentMethod, pending_amount)
                
                # Use the returned project data to update the session state
                st.session_state.sidebar_items[user_input] = {
                    "cost": project_data["cost"],
                    "exp": project_data["exp"],
                    "payment_status": project_data["payment_status"],
                    "payment_method": project_data["payment_method"],
                    "tasks": []  # Initialize an empty list for tasks
                }
                st.session_state.total_time_per_page[user_input] = 0  # Initialize time for new project  
                st.toast(f"'{user_input}' added!")
                st.session_state.user_input = ""  # Reset the input field
                st.rerun()  # Refresh the app to reflect the new project
            
            elif user_input in st.session_state.sidebar_items:
                st.warning(f"Project '{user_input}' already exists in the sidebar.")
            else:
                st.warning("Please enter a Project Name")

    show_form()
        


                    
# Display content based on the selected page
if selected_item == "Home":
    st.write("### Home Page")
    tab2, tab1,  tab3= st.tabs(["Overview","TimeLine", "Ask AI"])
    
    with tab2: 
        dashboard()
        
    
    with tab3:
        ai()
   
    
    with tab1:
        overview()
       
        
    st.divider()

elif selected_item:
    project_details = st.session_state.sidebar_items.get(selected_item)
    description = project_inform(selected_item)
    
    
    # Handle dynamic pages created by the user
    
    st.write(f"### Project - {selected_item}")
        # Create a container
        
    container = st.container()
    refined_text = improve_text(selected_item, description, container)

    # Clear any previous content in the container
    container.empty()  

    # Display only the refined text
    container.write(refined_text)


                
    
    # Get the ticket size from the database
    ticket_size = get_ticket_size(selected_item)
    
    # Display ticket size annotation based on the fetched value
    if ticket_size == "High Ticket":
        high_ticket = annotated_text(("High Ticket", "50K+"))
    elif ticket_size == "Mid Ticket":
        mid_ticket = annotated_text(("Mid Ticket", "20-50K"))
    elif ticket_size == "Low Ticket":
        low_ticket = annotated_text(("Low Ticket", "20K"))
    else:
        st.write("No Ticket Size Data Available")
    
        


    header_0 , header_1 , header_2 = st.columns(3)

    with header_0:
        ui.avatar(src="https://imagedelivery.net/tSvh1MGEu9IgUanmf58srQ/e2b094c8-8519-4e8b-e92e-1cf8d4b58f00/public")
        
        
    

    
    with header_1:
            
        if selected_item in st.session_state.sidebar_items:
            project_details = st.session_state.sidebar_items[selected_item]

            # Get the payment status safely (avoid KeyError)
            payment_status = project_details.get("payment_status", "Unknown")  # Default to "Unknown" if missing

            ui.badges(
                badge_list=[(payment_status, "destructive")],  # Dynamic badge label
                class_name="flex gap-2",
                key="badges1"
            )
        else:
            st.warning(f"Project '{selected_item}' not found.")
            

    
    # Function to fetch project details
    def fetch_project_data():
        project_details = get_project_details(selected_item)  # Fetch from DB
        
        # ✅ Update session state with fresh data
        st.session_state.project_status = project_details.get("status", "In Progress")
        st.session_state.completion_date = project_details.get("completion_date", "Not Available")
        st.session_state.other_data = project_details  # Store the full project details

    # ✅ Check if session state exists, else fetch data
    if "project_status" not in st.session_state or "completion_date" not in st.session_state or "other_data" not in st.session_state:
        fetch_project_data()


        # ✅ Dynamic Progress Bar or Completion Message
    with header_2:
    
        # Filter the projects list to only include the selected project
        projects_p = get_projects_progress_bar()
        selected_project = [project for project in projects_p if project[1] == selected_item]

        # Constants for project statuses
        STATUS_COMPLETED = "Completed"

        # Only proceed if the selected project exists
        if selected_project:
            project_id, project_name, status, created_at, deadline, completed_at = selected_project[0]  # Assuming completed_at is available

            # If the project is completed, display success message with additional details
            if status == STATUS_COMPLETED:
                try:
                    deadline_date = datetime.strptime(deadline, "%Y-%m-%d")
                    completed_date = datetime.strptime(completed_at, "%Y-%m-%d")  # Assuming completed_at is in the same format
                except ValueError:
                    st.error(f"Invalid date format for project **{project_name}**.")
                else:
                    # Calculate the difference between completion date and deadline
                    days_difference = (completed_date - deadline_date).days

                    if days_difference < 0:
                        st.success(f"Hooray! Completed **{abs(days_difference)} days** before the deadline! 🎉")
                    elif days_difference == 0:
                        st.success(f"completed right on time! ⏰")
                    else:
                        st.success(f"completed **{days_difference} days** after the deadline. 🚨")
            else:
                # Process projects that have valid creation and deadline dates.
                if created_at and deadline:
                    try:
                        created_date = datetime.strptime(created_at.split(" ")[0], "%Y-%m-%d")
                        deadline_date = datetime.strptime(deadline, "%Y-%m-%d")
                    except ValueError:
                        st.error(f"Invalid date format for project **{project_name}**.")
                    else:
                        today_date = datetime.today()

                        total_days = (deadline_date - created_date).days
                        elapsed_days = (today_date - created_date).days
                        remaining_days = (deadline_date - today_date).days

                        # Check if the deadline has been exceeded.
                        if today_date > deadline_date:
                            exceeded_days = (today_date - deadline_date).days
                            st.error(f"🚨 **{project_name}** - Exceeded the deadline by **{exceeded_days} days**!")
                        else:
                            # Calculate progress only if total_days is valid.
                            progress = min(1, elapsed_days / total_days) if total_days > 0 else 1
                            st.write(f"⚠️ **Deadline** - {remaining_days} days to go")
                            st.progress(progress)
        else:
            st.warning(f"No project found with the name **{selected_item}**.")






                                

    with st.container(border=True):
    #MAKE THE TASK
        with st.popover("Make Action"):
            @st.dialog("Add Task")
            def show_form():
                with st.form("my_form_task"):
                    task_input = st.text_input(f"Enter a task for {selected_item}:")
                    submitted = st.form_submit_button("Submit")

                    if submitted:
                        if task_input:
                            # ✅ Store task in the database
                            add_task(selected_item, task_input)

                            # ✅ Ensure selected project exists in session state
                            if selected_item not in st.session_state.sidebar_items:
                                st.session_state.sidebar_items[selected_item] = {"tasks": []}  # Initialize tasks list

                            # ✅ Append task to the session state
                            st.session_state.sidebar_items[selected_item]["tasks"].append(task_input)

                            # ✅ Initialize task timer in session state
                            if "task_timers" not in st.session_state:
                                st.session_state.task_timers = {}

                            st.session_state.task_timers[task_input] = {
                                "start_time": None,
                                "elapsed_time": 0,
                                "running": False
                            }

                            st.success(f"Task '{task_input}' added to {selected_item} and saved in the database!")
                            st.rerun()  # Refresh the UI
                        else:
                            st.warning("Please enter a task.")

            # Button to open the Add Task dialog
            if st.button("➕ Create a new Task"):
                show_form()

                
            # Delete the Project    
            @st.dialog("Delete Project")
            def del_show_form():
                with st.form("delete_form"):
                    delete_button = st.form_submit_button("Permanent Delete")

                    if delete_button:
                        # ✅ Delete from session state
                        if selected_item in st.session_state.sidebar_items:
                            del st.session_state.sidebar_items[selected_item]

                        if selected_item in st.session_state.total_time_per_page:
                            del st.session_state.total_time_per_page[selected_item]

                        # ✅ Ensure associated data is removed
                        if "tasks" in st.session_state and selected_item in st.session_state["tasks"]:
                            del st.session_state["tasks"][selected_item]

                        if "expenses" in st.session_state and selected_item in st.session_state["expenses"]:
                            del st.session_state["expenses"][selected_item]

                        # ✅ Delete from the database
                        delete_project(selected_item)

                        st.success(f"Project '{selected_item}' and all associated data deleted successfully!")
                        st.rerun()  # Refresh the app to reflect changes
                    else: st.warning("This will Permanent Delete the Project")

            

            
            # Button to open the Delete Project dialog
            if st.button("🗑️ Delete the Project"):
                del_show_form()
                
                
            # Edit the Project Deatils   
            @st.dialog("Edit Project")
            def show_form():
                #with st.form("edit_form"):
                    # Get existing project details
                    # ✅ Step 1: Fetch project details from database
                    # ✅ Step 1: Fetch project details from database
                    project_details = fetch_project_details(selected_item)
                    
                    

                    # ✅ Step 2: Ensure "total_time" exists in project details
                    if "total_time" not in project_details or project_details["total_time"] is None:
                        project_details["total_time"] = 0  # Default to 0 if missing

                    # ✅ Step 3: Force update session state to match database
                    if selected_item not in st.session_state.sidebar_items:
                        st.session_state.sidebar_items[selected_item] = project_details  # Store project details in session
                    else:
                        st.session_state.sidebar_items[selected_item]["total_time"] = project_details["total_time"]  # ✅ Force update from DB

                    # ✅ Step 4: Convert seconds to minutes before displaying

                    
                    stored_deadline = project_details.get("deadline", None)  # Get deadline from DB

                    # ✅ Convert string to date if needed
                    if stored_deadline and isinstance(stored_deadline, str):
                        try:
                            stored_deadline = datetime.strptime(stored_deadline, "%Y-%m-%d").date()
                        except ValueError:
                            print("⚠️ ERROR: Incorrect date format in DB:", stored_deadline)
                            stored_deadline = None  # Reset in case of error

                    # ✅ If no deadline is found in DB, DO NOT default to today (only fallback if needed)
                    if stored_deadline is None:
                        stored_deadline = datetime.today().date()  # Only use today if DB has no value

                


                    old_name = selected_item  # Store the original name

                    user_input = st.text_input("Enter Project Name", value=selected_item, placeholder="Project Name")
                    
                    # ✅ Get stored project time (manual input) from session or database
                    stored_time = st.session_state.sidebar_items[selected_item].get("total_time", 0)

                    # ✅ Get total task time (sum of elapsed task times)
                    task_time = sum(t["elapsed_time"] for t in st.session_state.task_timers.values())

                    # ✅ Final total time = manual input + task time
                    user_time_mins = st.number_input(
                        "Enter Total Time (in minutes)", 
                        value=float((stored_time + task_time) / 60),  # Convert to minutes
                        key="user_time_mins",  # Store in session state
                        on_change=update_project_time  # Trigger real-time update
                    )
                    
                    # user_time_mins = st.number_input(
                    #     "Enter Total Time (in minutes)", 
                    #     value=float(st.session_state.sidebar_items[selected_item]["total_time"] / 60) 
                    # )
                    
                    user_mail = st.text_input('Enter Mail Id',placeholder='example@gmail.com',value=project_details.get('mail',0))


                    user_amt = st.number_input("Enter Project Cost", value=project_details.get("cost", 0), placeholder="₹ 10,000")
                    
                    user_exp = st.number_input("Enter Expense Cost", value=project_details.get("exp", 0), placeholder="₹ 10,000",disabled=True)
                    
                    
                    paymentMethod = st.text_input("Enter Payment Method", value=project_details.get("payment_method", "N/A"), placeholder="Payment Method")
                    
                    col1, col2 = st.columns(2)
                    

                    
                    payment_status_options = ["UnPaid", "Paid", "Partial Paid"]
                    selected_status = project_details.get("payment_status", "UnPaid").title()  

                    if selected_status not in payment_status_options:
                        selected_status = "UnPaid"  

                    with col1:
                        paymentStatus = st.selectbox("Payment Status", payment_status_options, index=payment_status_options.index(selected_status))
                        

                    
                    Project_complt = ["Progress", "Completed", "On Hold (or Paused)"]
                    project_status = project_details.get("status", "Progress").title()
                    
                    if project_status not in Project_complt:
                        project_status = "Progress"
                                
                                
                    with col2:
                        Project_com = st.selectbox("Project Status", Project_complt, index=Project_complt.index(project_status))
                
                    
                    
                    col1 , col2 = st.columns(2)
                    
                    
                    tech_stack = [
                        # Web Development
                        "HTML", "CSS", "JavaScript", "TypeScript",
                        "React.js", "Next.js", "Vue.js", "Angular",'Json', 'Json-ld',
                        "Node.js", "Express.js", "Django", "Flask", "Laravel",'jinja template','Jinja2',

                        # Mobile App Development
                        "Flutter", "React Native", "Swift (iOS)", "Kotlin (Android)", "Java (Android)",

                        # Backend Development
                        "Python", "Java", "C#", "PHP", "Ruby on Rails", "GoLang",
                        "FastAPI-python", "Spring Boot", "NestJS",'Flask-python','Django-Python'

                        # Database & Storage
                        "MySQL", "PostgreSQL", "MongoDB", "Firebase", "SQLite", "Redis", "Supabase",

                        # Cloud Computing & DevOps
                        "AWS", "Google Cloud", "Azure", "DigitalOcean",
                        "Docker", "Kubernetes", "Terraform", "Ansible",
                        "CI/CD (Jenkins, GitHub Actions, GitLab CI)",

                        # CMS & Website Builders
                        "WordPress", "Wix", "Shopify", "Magento", "Ghost", "Webflow",

                        # E-commerce Platforms
                        "WooCommerce", "BigCommerce", "PrestaShop", "OpenCart", "Squarespace",

                        # Marketing & SEO
                        "Google Analytics", "Google Ads", "Facebook Ads", "Ahrefs", "SEMrush",
                        "HubSpot", "Mailchimp", "Hootsuite", "Buffer","Go-daddy",

                        # AI & Machine Learning
                        "TensorFlow", "PyTorch", "OpenAI API", "LangChain", "Scikit-Learn",
                        "Hugging Face Transformers", "GPT-4", "Stable Diffusion",

                        # Cybersecurity & Ethical Hacking
                        "Metasploit", "Wireshark", "Nmap", "Burp Suite", "Kali Linux",

                        # Blockchain & Web3
                        "Solidity", "Ethereum", "Polygon", "Binance Smart Chain",
                        "Smart Contracts", "NFT Development", "DApp Development",

                        # UI/UX & Graphic Design
                        "Figma", "Adobe XD", "Canva", "Photoshop", "Illustrator",
                        "Sketch", "CorelDRAW",

                        # Video Editing & Animation
                        "Adobe Premiere Pro", "After Effects", "Final Cut Pro", "Blender", "DaVinci Resolve",

                        # CRM & Business Tools
                        "Zoho CRM", "Salesforce", "HubSpot", "Trello", "Asana", "Slack", "Notion",

                        # Accounting & Finance
                        "QuickBooks", "Tally", "Xero", "Zoho Books",

                        # Miscellaneous
                        "Raspberry Pi", "Arduino", "IoT Development", "Low-Code/No-Code Platforms"
                        
                        #In House 
                        "In-House CRM",
                        
                        # Hosting & Domain Providers
                        "Namecheap",      # Affordable domain registration with free WHOIS privacy
                        "Bluehost",       # Recommended by WordPress, great for beginners
                        "Hostinger",      # Budget-friendly with good performance
                        "SiteGround",     # Premium hosting with excellent speed and support
                        "DreamHost",      # Strong privacy policies, WordPress-friendly
                        "Cloudflare",     # Secure domain registration with no markup
                        "AWS Route 53",   # Scalable, reliable DNS and hosting
                        "Google Domains", # Now moved to Squarespace, but still well-known
                        "IONOS (1&1)"     # Affordable hosting and business solutions
                    ]

                    with col1:
                        Tech_name = st.multiselect(
                        "Technology Used",
                        tech_stack,
                        default=project_details.get("technology", [])
                    )
                        

                    project_types = [
                        # Software Development
                        "SEO optimization",
                        "Web Application Development",
                        'Website Development',
                        "Figma Design"
                        'Website Design',
                        "Mobile App Development (iOS, Android, Hybrid)",
                        "SaaS (Software as a Service)",
                        "Enterprise Software Solutions",
                        "Desktop Application Development",
                        "Embedded Systems & Firmware Development",
                        "API Development & Integration",
                        'Google My Busniness',
                        "Custom CRM Development",
                        'Performance Marketing'
                        "Cloud-Based Solutions",
                        "Low-Code/No-Code Development",
                        "Product Lisitng",
                      

                        # Web & E-commerce
                        "E-commerce Website Development",
                        'GMB E-commerce',
                        'WhatsApp Connection',
                        "Multi-Vendor Marketplace Development",
                        "Custom CMS (Content Management System)",
                        "Website Revamp & Redesign",
                        "Subscription-Based Platform Development",
                        "Website Optimization & Performance Tuning",
                        "Progressive Web Apps (PWA)",
                        "Conversion Rate Optimization (CRO)",

                        # Artificial Intelligence & Machine Learning
                        "AI Chatbot Development",
                        "Machine Learning Model Training & Deployment",
                        "NLP (Natural Language Processing) Systems",
                        "Predictive Analytics & Forecasting Models",
                        "Computer Vision & Image Recognition",
                        "AI-Based Personalization Systems",
                        "Sentiment Analysis Systems",
                        "Generative AI & LLM (Large Language Models)",

                        # Data & Analytics
                        "Big Data Processing & Analytics",
                        "Business Intelligence (BI) TimeLines",
                        "Data Engineering & ETL Pipelines",
                        "Data Warehousing Solutions",
                        "Customer Data Platform (CDP) Development",
                        "Real-Time Analytics Systems",
                        "Data Visualization & Reporting",
                        "Data Scraping & Web Crawling",

                        # Cybersecurity & Compliance
                        "Penetration Testing & Security Audits",
                        "SOC (Security Operations Center) Implementation",
                        "Identity & Access Management (IAM)",
                        "Threat Detection & Incident Response Systems",
                        "GDPR & HIPAA Compliance Solutions",
                        "Blockchain-Based Security Solutions",
                        "Secure Payment Gateway Integration",

                        # Cloud & DevOps
                        "Cloud Infrastructure Setup (AWS, GCP, Azure)",
                        "Kubernetes & Containerization Solutions",
                        "CI/CD Pipeline Implementation",
                        "Infrastructure as Code (IaC)",
                        "Serverless Computing Solutions",
                        "Multi-Cloud Architecture",
                        "Database Migration & Optimization",
                        "Load Balancing & Auto Scaling",

                        # IoT & Smart Technologies
                        "IoT Device Development",
                        "Smart Home & Automation Solutions",
                        "Edge Computing & IoT Gateways",
                        "Industrial IoT (IIoT) Solutions",
                        "Wearable Tech Development",
                        "Connected Vehicles & Telematics",
                        "Smart Healthcare Devices",
                        "IoT Security & Compliance",

                        # Blockchain & Web3
                        "Cryptocurrency Wallet & Exchange Development",
                        "Decentralized Applications (dApps)",
                        "Smart Contract Development",
                        "NFT Marketplace Development",
                        "Tokenization & DeFi Solutions",
                        "Blockchain-Based Identity Verification",
                        "Supply Chain Blockchain Solutions",
                        "DAO (Decentralized Autonomous Organization) Development",

                        # FinTech Solutions
                        "Digital Banking Platform Development",
                        "Payment Gateway & Digital Wallet Integration",
                        "Robo-Advisory & Investment Platforms",
                        "P2P Lending & Crowdfunding Solutions",
                        "Stock Trading & Algorithmic Trading Platforms",
                        "Financial Risk & Fraud Detection",
                        "InsurTech Solutions",
                        "RegTech & Compliance Automation",

                        # HealthTech Solutions
                        "Telemedicine & Virtual Healthcare",
                        "Electronic Health Records (EHR) Systems",
                        "AI-Powered Medical Diagnostics",
                        "Remote Patient Monitoring (RPM)",
                        "Healthcare Data Analytics & Predictive Modeling",
                        "Pharmaceutical Supply Chain Solutions",
                        "Mental Health & Wellness Apps",
                        "Genomics & Personalized Medicine Solutions",

                        # EdTech & Learning Solutions
                        "Online Learning Management System (LMS)",
                        "AI-Based Adaptive Learning Platforms",
                        "Student Performance Analytics",
                        "E-Library & Digital Courseware Development",
                        "Gamified Learning Solutions",
                        "Skill-Based Training Platforms",
                        "Virtual Reality (VR) Education",
                        "Augmented Reality (AR) Learning Modules",

                        # Marketing & Advertising Tech
                        "Programmatic Advertising & AdTech Solutions",
                        "AI-Powered Marketing Automation",
                        "Social Media Analytics & Sentiment Analysis",
                        "Influencer Marketing Platforms",
                        "Affiliate Marketing & Referral Systems",
                        "Customer Journey Mapping & Analytics",
                        "Personalized Recommendation Engines",
                        "A/B Testing & Conversion Optimization",

                        # HR & Workforce Management
                        "AI-Powered Resume Screening & Hiring",
                        "Employee Performance Management Systems",
                        "Remote Work & Collaboration Tools",
                        "Payroll & Benefits Management Systems",
                        "Employee Engagement & Feedback Platforms",
                        "Workforce Planning & Optimization",
                        "Time Tracking & Attendance Systems",
                        "HR Chatbots & Virtual Assistants",

                        # Gaming & Entertainment
                        "Game Development (Mobile, PC, Console)",
                        "AR/VR Gaming Solutions",
                        "Metaverse Development",
                        "Esports & Live Streaming Platforms",
                        "AI-Based Game Analytics",
                        "Cloud Gaming Infrastructure",
                        "Interactive Storytelling & Immersive Media",
                        "NFT-Based Gaming & Play-to-Earn Platforms",

                        # Logistics & Supply Chain
                        "AI-Based Demand Forecasting",
                        "Fleet & Vehicle Management Systems",
                        "Warehouse Automation & Robotics",
                        "Real-Time Supply Chain Visibility",
                        "Last-Mile Delivery Optimization",
                        "Blockchain-Based Logistics Solutions",
                        "Cold Chain Monitoring & IoT Integration",
                        "E-commerce Fulfillment Solutions",

                        # LegalTech Solutions
                        "AI-Based Contract Analysis",
                        "Legal Research & Case Prediction",
                        "Online Dispute Resolution Platforms",
                        "Intellectual Property (IP) Management",
                        "Regulatory Compliance Automation",
                        "E-Discovery & Digital Forensics",
                        "Blockchain-Based Notary Services",
                        "Virtual Law Firm Management",

                        # Smart Cities & Urban Tech
                        "Smart Traffic Management Systems",
                        "AI-Based Waste Management",
                        "Energy Efficiency & Smart Grid Solutions",
                        "Public Safety & Surveillance Systems",
                        "Urban Air Quality Monitoring",
                        "Smart Parking Solutions",
                        "Citizen Engagement Platforms",
                        "Smart Infrastructure & Building Automation",

                        # Automotive & Mobility
                        "Autonomous Vehicle Software",
                        "Electric Vehicle (EV) Infrastructure",
                        "AI-Powered Vehicle Diagnostics",
                        "Connected Car Solutions",
                        "Ride-Hailing & Mobility-as-a-Service (MaaS)",
                        "Smart Traffic Analytics & Navigation",
                        "Drone Delivery Systems",
                        "AI-Based Fleet Optimization",

                        # Renewable Energy & Sustainability
                        "Solar & Wind Energy Management Platforms",
                        "AI-Powered Energy Forecasting",
                        "Carbon Footprint Tracking & Management",
                        "Smart Grid & Energy Storage Solutions",
                        "Water Conservation & Smart Irrigation",
                        "Waste Recycling & Circular Economy Platforms",
                        "Green Building & Smart HVAC Systems",
                        "Climate Risk & Environmental Impact Analytics"
                    ]

                    
                    
                    with col2:
                        project_type = st.multiselect("Project Type",project_types, default=project_details.get("project_type", []))


                    industry_types = [
                        "IT & Software Development",
                        'Graphic Design & Illustration'
                        "E-commerce & Retail",
                        "Healthcare & Pharmaceuticals",
                        "Finance & Banking",
                        "Real Estate & Construction",
                        "Logistics & Supply Chain",
                        "Manufacturing & Industrial",
                        "Education & E-learning",
                        "Marketing & Advertising",
                        'SEO & Digital Marketing Consulting',
                        'Doctors & Medical Portfolios',
                        "Government & Public Services",
                        "Travel & Hospitality",
                        "Energy & Utilities",
                        "Automotive & Transportation",
                        "Media & Entertainment",
                        "Agriculture & Food Processing",
                        "Telecommunications & Networking",
                        "Legal & Compliance",
                        "Sports & Fitness",
                        "Gaming & Esports",
                        "Cybersecurity & Data Protection",
                        "HR & Recruitment",
                        "Electronics & Consumer Goods",
                        "Environmental & Sustainability",
                        "Event Management & Planning",
                        "Aerospace & Defense",
                        "FMCG (Fast-Moving Consumer Goods)",
                        "Mining & Metals",
                        "Insurance & Risk Management",
                        "Non-Profit & NGOs",
                        "Biotechnology & Life Sciences",
                        "Blockchain & Cryptocurrency",
                        "Petroleum & Natural Gas",
                        "Waste Management & Recycling",
                        "Textile & Apparel",
                        "Furniture & Home Decor",
                        "Food & Beverage",
                        "Artificial Intelligence & Machine Learning",
                        "Robotics & Automation",
                        "Electronics & Semiconductors",
                        "Luxury & Lifestyle",
                        "Freelancing & Gig Economy",
                        "Crowdfunding & Venture Capital",
                        "Publishing & Print Media",
                        "Spirituality & Wellness",
                        "Beauty & Personal Care",
                        "Senior Care & Assisted Living",
                        "Dairy & Animal Husbandry",
                        "Tattoo & Body Art",
                        "Handicrafts & Artisanal Products",
                        "Urban Development & Smart Cities",
                        "Data Science & Analytics",
                        "Renewable Energy (Solar, Wind, Hydro, etc.)",
                        "Luxury Watches & Accessories",
                        "Workplace Productivity & Collaboration Tools",
                        "Interior Design & Architecture",
                        "Drones & UAV Technology",
                        "3D Printing & Prototyping",
                        "Subscription Services & Memberships",
                        "Printing & Packaging Industry",
                        "Astrology & Numerology",
                        "Wedding Planning & Luxury Events",
                        'Influencer Marketing & Content Creation',
                        'Luxury Cars & Supercars',
                        'Metaverse & Virtual Reality (VR/AR)',
                        'Cloud Computing & SaaS',
                        'Social Media & Community Building',
                        'Ethical Hacking & Penetration Testing'
                    ]
                    

                    sector_name = st.selectbox("Sector Name",industry_types,index=industry_types.index(project_details.get("sector_name")) if project_details.get("sector_name") in industry_types else 0)
                    
                    
                       # ✅ Fetch stored deadline (Fix for incorrect reset)
                    stored_deadline = fetch_project_details(selected_item).get("deadline", None)

                    # ✅ Convert string to date if needed
                    if stored_deadline and isinstance(stored_deadline, str):
                        stored_deadline = datetime.strptime(stored_deadline, "%Y-%m-%d").date()

                    # ✅ If the deadline is still None, set a fallback value (DO NOT default to today)
                    if stored_deadline is None:
                        stored_deadline = datetime.today().date()  # Default only if missing in DB

                    # ✅ Show deadline in the form
                    # ✅ Use fetched deadline in UI
                    deadline = st.date_input("Select Project Deadline", value=stored_deadline)
                    
                    description = st.text_area("Project Description",placeholder='Provide some information about this project',
                    value=project_details.get("project_description", ""))
                    
                    st.write(st.session_state.sidebar_items[selected_item])  # Debugging: Check stored data


                                
                    st.divider()
                    
                    
                    submitted = st.button("Submit")

                    if submitted:
                        if Project_com != st.session_state.project_status:  # ✅ Only update if changed
                            st.session_state.project_status = Project_com  # ✅ Update session state
                
                        
                        user_time_secs = user_time_mins * 60
                        # ✅ Update the project in the database
                        # ✅ Update the project in the database
                        update_project(old_name, user_input, user_amt, user_exp, paymentStatus, paymentMethod, deadline, 
                        Project_com, Tech_name, project_type, sector_name, description,user_mail)

                        update_total_time(old_name, user_time_secs)

                        # ✅ Preserve tasks by fetching from database
                        updated_tasks = fetch_tasks(user_input)
                        
                        if user_input in st.session_state.sidebar_items:
                            st.session_state.sidebar_items[user_input]["tasks"] = updated_tasks 
                            
                        # ✅ Fetch latest status from DB and update session state
                        updated_status = fetch_project_details(user_input).get("status", "Progress")
                        st.session_state.project_status = updated_status
                        
                        #st.session_state.sidebar_items[old_name]["total_time"] = user_time_secs
                        #st.session_state.total_time_per_page[user_input] = user_time_secs 

                        # ✅ Update session state with new values
                        st.session_state.sidebar_items[user_input] = {
                            "total_time": user_time_secs,
                            "cost": user_amt,
                            "exp": user_exp,
                            "payment_status": paymentStatus,
                            "payment_method": paymentMethod,
                            "tasks": updated_tasks,  # ✅ Preserve existing tasks
                            "deadline": deadline,
                            "status": updated_status,  # ✅ Add status field
                            "sector_name": sector_name,
                            "project_description": description,
                            "technology": Tech_name,  # ✅ Store selected technologies
                            "project_type": project_type,  # ✅ Store selected project types
                            'gmail': user_mail
                        }

                        # ✅ If project name is changed, remove old entry from session state
                        if old_name != user_input:
                            del st.session_state.sidebar_items[old_name]
                            del st.session_state.total_time_per_page[old_name]  # Remove old time reference

                        st.toast(f"'{user_input}' updated successfully!")
                        st.rerun()  # Refresh the app to reflect changes

            # Button to open the Edit Project dialog
            if st.button("📝 Edit the Project"):
                show_form()

                    
    
        project_time = st.session_state.total_time_per_page.get(selected_item, 0)  # Default to 0 if missing

        # total_profit = project_cost - total_expense  # Simple profit calculation
    
        # Calculate total profit (cost - expense)
        project_cost = float(project_details['cost'])  # Ensure it's a number
        total_expense = float(project_details["exp"]) if project_details["exp"] is not None else 0
        # Ensure it's a number
        total_profit = project_cost - total_expense  # Profit calculation

        # Create columns for the 4 cards
        # Create 2 columns for the 2x2 grid

        cols = st.columns(2)


        # Display the four cards in a 2x2 grid using a loop and alternating columns
        for i, (title, content, description) in enumerate([
            ("Project Time", format_time(project_time), "Time spent on this project"),
            ("Project Cost", f"₹ {format_with_commas(project_details['cost'])}", "Cost of the project"),
            ("Total Expense",f"₹ {format_with_commas(project_details['exp'])}", "Total expenses incurred"),
            ("Total Profit", f"₹ {format_with_commas(total_profit)}", "Profit from the project")
        ]):
            col = cols[i % 2]  # Alternate between columns
            
            with col:
                ui.card(
                    title=title,
                    content=content,
                    description=description
                ).render()
            
            
    
    with st.container(border=True):

        # Input field to add tasks specific to the selected page
        
        # 🔥 **DIALOG FOR ADDING TASKS**
    
        with st.expander("View All Tasks"):
            # Check if tasks exist for the selected project
            if "tasks" in st.session_state.sidebar_items[selected_item]:
                st.write(f"### Tasks for {selected_item}")
                task_list = st.session_state.sidebar_items[selected_item]["tasks"]

                if task_list:
                    # Create a radio button for selecting tasks
                    task_radio = st.radio(
                        "", 
                        [task["task_name"] if isinstance(task, dict) else task for task in task_list]  # Ensure strings
                    )


                    # Initialize task state if not already present
                    if task_radio not in st.session_state.task_timers:
                        st.session_state.task_timers[task_radio] = {
                            "running": False,
                            "start_time": 0,
                            "elapsed_time": 0
                        }


                    task_state = st.session_state.task_timers[task_radio]
                    button_text = "Stop Timer" if task_state["running"] else "Start Timer"

                    if st.button(button_text, key=f"timer_{task_radio}"):
                        if task_state["running"]:
                            # ✅ Stop the timer and calculate elapsed time
                            elapsed_time = time.time() - task_state["start_time"]
                            task_state["elapsed_time"] += elapsed_time
                            task_state["running"] = False

                            # ✅ Update session state total time
                            st.session_state.total_time_per_page[selected_item] += elapsed_time

                            # ✅ Save task time to database
                            update_task_time(selected_item, task_radio, task_state["elapsed_time"])

                            # ✅ Update total project time in DB
                            new_total_time = st.session_state.total_time_per_page[selected_item]
                            update_total_time(selected_item, new_total_time)  # ✅ Ensure project time is updated
                        else:
                            # ✅ Start a new session
                            task_state["elapsed_time"] = 0  # Reset elapsed time
                            task_state["start_time"] = time.time()
                            task_state["running"] = True

                        st.rerun()  # Refresh UI


                    # Show elapsed time in the format you want (minutes:seconds)
                    elapsed_time_placeholder = st.empty()
                    if task_state["running"]:
                        while task_state["running"]:
                            # Calculate the current elapsed time by adding the current session time
                            elapsed_time_display = task_state["elapsed_time"] + (time.time() - task_state["start_time"])
                            elapsed_time_placeholder.write(f"Elapsed Time: {format_time(elapsed_time_display)}")
                            time.sleep(1)  # Update every second
                        st.rerun()  # Refresh once per second

                    # Calculate the total time spent across all tasks
                    total_time_spent = sum(t["elapsed_time"] for t in st.session_state.task_timers.values())
                    #st.success(f"Total time spent on {task_list}: {format_time(total_time_spent)}")
                    
                    
                    
                else:
                    st.info("No tasks added yet. Click 'Create a new Task' to add one.")

                
        st.divider()
        def generate_sales_data():
            if selected_item:
                task_data = fetch_total_time_per_task(selected_item)  # ✅ Fetch from DB

                # ✅ Handle case when no tasks exist
                if not task_data:
                    task_data = [{"task_name": "No Tasks", "total_time": 0}]

                # ✅ Extract task names and times
                task_names = [task["task_name"] for task in task_data]
                task_times = [task["total_time"] if task["total_time"] else 0 for task in task_data]
                human_readable_times = [format_time(task["total_time"]) if task["total_time"] else "0 seconds" for task in task_data]

                df = pd.DataFrame({
                    'Task': task_names, 
                    'Total Time Spent': task_times,  # Numerical for the chart
                    'Time Taken': human_readable_times  # For tooltip display
                })
                return df
            else:
                return pd.DataFrame({'Task': ["No Data"], 'Total Time Spent': [0], 'Time Taken': ["0 seconds"]})

        # ✅ Display the chart with human-readable tooltip
        with card_container(key="chart1"):
            st.vega_lite_chart(generate_sales_data(), {
                'mark': {'type': 'area', 'tooltip': True, 'fill': 'rgb(144,238,144)', 'cornerRadiusEnd': 4},
                'encoding': {
                    'x': {'field': 'Task', 'type': 'ordinal'},
                    'y': {'field': 'Total Time Spent', 'type': 'quantitative', 'axis': {'grid': False}},
                    'tooltip': [
                        {'field': 'Task', 'type': 'ordinal'},
                        {'field': 'Time Taken', 'type': 'nominal'}  # Shows human-readable time on hover
                    ]
                },
            }, use_container_width=True)

                    
            
                
        st.divider()
        
        
        Expense, amount= st.tabs(["Expense", "Amount"])
            # Fetch and display amount data
            
        # Fetch project cost from database
        project_cost = get_project_cost(selected_item)
        # Fetch and display amount data
        data, pending_amount = get_amount_details(selected_item)
        


        with amount:
            st.subheader(f"Amount Details for {selected_item}")

            @st.dialog(f"Add Amount for {selected_item}")
            def show_form_amount():
                with st.form('Amount received'):
                    # Fetch total received so far
                    total_received_so_far = get_total_received(selected_item)  # Function to get previously received amount

                    received_amount = st.number_input(f"Enter the amount received", step=1.0, max_value=pending_amount)
                    paym_mode = st.text_input(f"Payment Method", placeholder='Credit Card , Cash')

                    # Calculate pending correctly
                    new_total_received = total_received_so_far + received_amount
                    Pending_amount = max(project_cost - new_total_received, 0)  # Ensure it doesn’t go negative

                    submitted = st.form_submit_button("Submit")

                    if submitted:
                        new_status = update_payment_status(selected_item, new_total_received, project_cost, Pending_amount)

                        if received_amount > 0:
                            add_amount_rcvd(selected_item, received_amount, paym_mode, Pending_amount)
                            st.session_state.updated = True  # Set update flag
                            st.success("Amount saved successfully!")
                            st.rerun()  # Refresh UI to reflect changes
                        else:
                            st.error("Please enter a valid amount.")     



            @st.dialog(f"Edit Amount for {selected_item}")
            def edit_amount():
                amounts, pending_amount = get_amount_details(selected_item)

                if not amounts:
                    st.warning("No amounts recorded for this project.")
                    return
                
                # Debugging: Print the fetched data
                #st.write("Fetched Data:", amounts)

                for record in amounts:  # Iterate over list of dictionaries
                    date = record["Date"]
                    amount = record["Amount Received"]
                    pay_mode = record["Payment Mode"]

                    with st.expander(f"Amount: {amount} | Payment Mode: {pay_mode} | Date: {date}"):
                        new_amount = st.number_input("Edit Amount Received", value=amount, step=1.0)
                        new_paym_mode = st.text_input("Edit Payment Method", value=pay_mode)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("Update", key=f"update_{date}"):
                                new_total_received = update_amount_rcvd(selected_item, date, new_amount, new_paym_mode)
                                
                                # Recalculate pending amount dynamically
                                new_pending_amount = max(project_cost - new_total_received, 0)

                                # Update the pending amount in the database (if needed)
                                update_pending_amount(selected_item, new_pending_amount)

                                st.success(f"Amount updated successfully! New Pending Amount: {new_pending_amount}")
                                st.rerun()
                        
                        with col2:
                            if st.button("Delete", key=f"delete_{date}"):
                                delete_amount_rcvd(selected_item, date)
                                st.warning("Amount deleted successfully!")
                                st.rerun()

            
            # Buttons to add or edit amount
            with st.popover("Make Changes"):  
                if st.button(f"➕ Add Amount"):
                    show_form_amount()
                if st.button(f"📝 Edit Amount"):
                    edit_amount()

            


            if data:
                for amount in data:
                    amount["Payment Mode"] = format_with_commas(amount["Payment Mode"])
                    amount["Project Cost"] = format_with_commas(amount["Project Cost"])
                    amount["Amount Received"] = format_with_commas(amount["Amount Received"])
                    amount["Pending Amount"] = format_with_commas(amount["Pending Amount"])

                amount_df = pd.DataFrame(data)
                ui.table(data=amount_df, maxHeight=300)
            else:
                st.warning(f"No data found for {selected_item}")



        with Expense:

            
            st.subheader(f"Expense Details for {selected_item}")
            
            
            @st.dialog(f"Add Expense for {selected_item}")
            def show_form():
                with st.form(f"my_exp_{selected_item}"):  # ✅ Unique form for each project
                    task_input_exp = st.text_input(f"Expense Name for {selected_item}")
                    amount_exp = st.number_input(f"Expense Amount", min_value=0, step=1)
                    submitted = st.form_submit_button("Submit")

                    if submitted:
                        if submitted:
                            if task_input_exp and amount_exp > 0:
                                add_expense_to_db(selected_item, task_input_exp, amount_exp)  # ✅ Store in DB

                                # ✅ Fetch updated total expenditure after adding expense
                                updated_exp = fetch_project_details(selected_item)["exp"]

                                # ✅ Update session state for real-time UI update
                                st.session_state.sidebar_items[selected_item]["exp"] = updated_exp

                                st.rerun()  # Refresh UI to reflect updated expenditure
                            else:
                                st.warning("Enter valid details")
                                
            @st.dialog(f"Edit Expense for {selected_item}")
            def edit_expense():
                expenses = fetch_expenses(selected_item)  # Get all expenses for the project

                if not expenses:
                    st.warning("No expenses found for this project.")
                    return

                with st.form("Edit Expense Form"):
                    st.subheader("Edit Expense")

                    # Create input fields for each expense
                    for expense in expenses:
                        st.write(f"### {expense['Expenditure']}")

                        new_expense_name = st.text_input(f"Expense Name for {expense['Expenditure']}", value=expense["Expenditure"])
                        new_amount = st.number_input(
                            f"Expense Amount for {expense['Expenditure']}",
                            min_value=0.0,
                            step=0.01,
                            value=float(expense["Amount"])
                        )

                        submitted = st.form_submit_button(f"Update {expense['Expenditure']}")
                        divider()

                        if submitted:
                            if new_expense_name and new_amount > 0:
                                edit_exp = update_expense(selected_item, expense["Expenditure"], new_expense_name, new_amount)  # ✅ Pass project_name & old_expense_name
                                
                                if edit_exp is not None:
                                    st.session_state.sidebar_items[selected_item]["exp"] = edit_exp
                                else:
                                    st.warning("Expense update failed. `edit_exp` is None!")

                                                                
                                
                                
                                st.rerun()  # Refresh UI
                            else:
                                st.warning("Enter valid details")





                    


            with st.popover("Make Changes"): 
                if st.button(f"➕ Add Expense"):
                    show_form()
            
                if st.button(f"📝 Edit Expense"):
                    edit_expense()



            # Fetch expenses only for the selected project
            expenses_data = fetch_expenses(selected_item)

            # ✅ Format the "Amount" column before displaying
            if expenses_data:
                for expense in expenses_data:
                    expense["Amount"] = format_with_commas(expense["Amount"])  # Format Amount

                invoice_df = pd.DataFrame(expenses_data)
                ui.table(data=invoice_df, maxHeight=300)
            else:
                st.write(f"No expenses added yet")
                    
                
       




        #Save the selected item in session state
        if 'selected_item' not in st.session_state:
            st.session_state.selected_item = selected_item

        divider()

        # Button to navigate to the feedback form page (form.py)
        st.subheader("Client Feedback")

        # ✅ Generate a sharable link
        feedback_url = f"http://localhost:8502?project={selected_item}"
        ui.link_button(text="Share the Feedback Form", url=feedback_url, key="link_btn")
        

        # ✅ Fetch and Display Feedback
        feedback_data = fetch_feedback(selected_item)

        if feedback_data:
            for feedback in feedback_data:
                st.write(f"👤 **{feedback['client_name']}**")
                st.write(f"⭐ Rating: {feedback['rating']}/4")
                st.write(f"💬 {feedback['comments']}")
                st.write(f"📅 {feedback['date']}")
                
                divider()

        else:
            st.info("No feedback yet. Share the form with clients!")
            
            
        divider()

        st.subheader("Map")
        
        project_name = selected_item
        
        
        if project_name:
            # Fetch locations from the database
            locations = fetch_location_per_client(project_name)

            if locations:
                # Initialize empty lists for latitudes and longitudes
                latitudes = []
                longitudes = []

                # Splitting and collecting latitudes & longitudes
                for location in locations:
                    try:
                        lat, lang = location.split(',')  # Split the location string into latitude and longitude
                        latitudes.append(float(lat.strip()))  # Convert to float and append to the list
                        longitudes.append(float(lang.strip()))  # Convert to float and append to the list
                    except ValueError:
                        st.warning(f"Invalid location format: {location}")

                # Create a DataFrame with the collected latitudes and longitudes
                data = pd.DataFrame({
                    'latitude': latitudes,  # List of latitudes
                    'longitude': longitudes  # List of longitudes
                })

                # Display the map
                st.subheader("Client Location Map")
                st.map(data, zoom=12, size=220)
            else:
                st.warning("No location data available for this project.")
                    
            
        divider()
            
            
   

else:
    st.write("Select a project from the sidebar.")



