import sqlite3 , json 
from datetime import datetime
import pandas as pd
conn = sqlite3.connect('project_management.db')

# Function to create the database and the 'projects' table
def create_db():
    conn = sqlite3.connect('project_management.db')
    cursor = conn.cursor()
    
        # Create the 'projects' table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS projects (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE,
        cost REAL NOT NULL,
        expenditure REAL NOT NULL,
        payment_status TEXT NOT NULL,
        payment_method TEXT NOT NULL,
        total_time REAL DEFAULT 0,  
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        status TEXT DEFAULT 'In Progress',
        deadline TEXT,
        completion_date TIMESTAMP DEFAULT NULL,
        technology TEXT DEFAULT '[]',
        project_type TEXT DEFAULT '[]',
        sector_name TEXT DEFAULT '',
        project_description TEXT DEFAULT '',
        mail TEXT
    );
    ''')

    # Create the 'tasks' table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS tasks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        project_name TEXT NOT NULL,
        task_name TEXT NOT NULL,
        total_time REAL DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (project_name) REFERENCES projects(name) ON DELETE CASCADE
    );
    ''')

    # Create the 'expenses' table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS expenses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        project_name TEXT NOT NULL,
        expense_name TEXT NOT NULL,
        amount REAL NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (project_name) REFERENCES projects(name) ON DELETE CASCADE
    );
    ''')

    # Create the 'feedback' table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        project_name TEXT NOT NULL,
        client_name TEXT NOT NULL,
        rating INTEGER NOT NULL,
        comments TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (project_name) REFERENCES projects(name) ON DELETE CASCADE
    );
    ''')

    # Create the 'feedback_metadata' table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS feedback_metadata (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        project_name TEXT NOT NULL,
        ip_address TEXT,
        location TEXT,
        browser_info TEXT,
        referrer_url TEXT,
        device_info TEXT,
        battery_life TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    ''')

    # Create the 'add_amount' table
    # cursor.execute('''
    # CREATE TABLE IF NOT EXISTS add_amount (
    #     id INTEGER PRIMARY KEY AUTOINCREMENT,
    #     project_name TEXT NOT NULL,
    #     amount REAL NOT NULL,
    #     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    #     pay_mode TEXT
    # );
    # ''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS add_amount  (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        project_name TEXT NOT NULL,
        amount_received  REAL NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        pay_mode TEXT
    ,   pending_amount  REAL);
    ''')
    
   

  
    conn.commit()
    conn.close()
#amount & Pending ammpunt 

# Function to add a new project and return only the new project
def add_project(name, cost, expenditure, payment_status, payment_method):
    conn = sqlite3.connect('project_management.db')
    cursor = conn.cursor()

    cursor.execute('''
    INSERT INTO projects (name, cost, expenditure, payment_status, payment_method, total_time)
    VALUES (?, ?, ?, ?, ?, ?)
    ''', (name, cost, expenditure, payment_status, payment_method, 0))

    conn.commit()
    
    # Fetch the newly added project
    cursor.execute('SELECT * FROM projects WHERE name = ?', (name,))
    new_project = cursor.fetchone()
    
    conn.close()

    return {
        "name": new_project[1],
        "cost": new_project[2],
        "exp": new_project[3],
        "payment_status": new_project[4],
        "payment_method": new_project[5],
        "total_time": new_project[6]
    }

# Function to update total_time in the database safely
def update_total_time(project_name, new_total_time):
    conn = sqlite3.connect('project_management.db')
    cursor = conn.cursor()

    cursor.execute('SELECT COUNT(*) FROM projects WHERE name = ?', (project_name,))
    exists = cursor.fetchone()[0]

    if exists:
        cursor.execute('UPDATE projects SET total_time = ? WHERE name = ?', (new_total_time, project_name))
        conn.commit()
    else:
        print(f"Warning: Project '{project_name}' does not exist. No update made.")

    conn.close()
    
    
# Function to update an existing project in the database
# def update_project(old_name, new_name, cost, expenditure, payment_status, payment_method, deadline):
#     conn = sqlite3.connect('project_management.db')
#     cursor = conn.cursor()

#     # Ensure the project exists before updating
#     cursor.execute('SELECT COUNT(*) FROM projects WHERE name = ?', (old_name,))
#     exists = cursor.fetchone()[0]

#     if exists:
#         cursor.execute('''
#             UPDATE projects
#             SET name = ?, cost = ?, expenditure = ?, payment_status = ?, payment_method = ?, deadline = ?
#             WHERE name = ?
#         ''', (new_name, cost, expenditure, payment_status, payment_method, deadline, old_name))
#         conn.commit()
#     else:
#         print(f"Warning: Project '{old_name}' does not exist. No update made.")

#     conn.close()





def update_project(old_name, new_name=None, cost=None, expenditure=None, 
                    payment_status=None, payment_method=None, deadline=None, 
                    project_status=None, technology=None, project_type=None, 
                    sector_name=None, project_description=None, mail=None):
    
    conn = sqlite3.connect('project_management.db')
    cursor = conn.cursor()

    # ✅ Fetch existing project details
    cursor.execute('''
        SELECT name, cost, expenditure, payment_status, payment_method, deadline, status, 
               technology, project_type, sector_name, project_description, mail
        FROM projects WHERE name = ?
    ''', (old_name,))
    
    project = cursor.fetchone()

    if project:
        (existing_name, existing_cost, existing_expenditure, existing_payment_status, 
         existing_payment_method, existing_deadline, existing_status, 
         existing_technology, existing_project_type, existing_sector_name, 
         existing_project_description, existing_mail) = project

        updated_name = new_name if new_name else existing_name
        updated_cost = cost if cost is not None else existing_cost
        updated_expenditure = expenditure if expenditure is not None else existing_expenditure
        updated_payment_status = payment_status if payment_status else existing_payment_status
        updated_payment_method = payment_method if payment_method else existing_payment_method
        updated_deadline = deadline if deadline else existing_deadline  
        updated_status = project_status if project_status else existing_status
        updated_sector_name = sector_name if sector_name else existing_sector_name
        updated_project_description = project_description if project_description else existing_project_description
        updated_mail = mail if mail else existing_mail

        # ✅ Convert lists to JSON for storage
        updated_technology = json.dumps(technology) if technology else existing_technology
        updated_project_type = json.dumps(project_type) if project_type else existing_project_type

        # ✅ Set completion date only if status is "Completed"
        completion_date = datetime.now().strftime("%Y-%m-%d") if updated_status == "Completed" else None

        # ✅ Perform update
        cursor.execute('''
            UPDATE projects
            SET name = ?, cost = ?, expenditure = ?, payment_status = ?, payment_method = ?, 
                deadline = ?, status = ?, completion_date = ?, 
                technology = ?, project_type = ?, sector_name = ?, project_description = ?, mail = ?
            WHERE name = ?
        ''', (updated_name, updated_cost, updated_expenditure, updated_payment_status, updated_payment_method, 
              updated_deadline, updated_status, completion_date, 
              updated_technology, updated_project_type, updated_sector_name, updated_project_description, 
              updated_mail, old_name))

        conn.commit()
        conn.close()

        return {
            "updated_project": {
                "name": updated_name,
                "cost": updated_cost,
                "expenditure": updated_expenditure,
                "payment_status": updated_payment_status,
                "payment_method": updated_payment_method,
                "deadline": updated_deadline,
                "status": updated_status,
                "completion_date": completion_date,
                "technology": json.loads(updated_technology),  # Convert back from JSON
                "project_type": json.loads(updated_project_type),  # Convert back from JSON
                "sector_name": updated_sector_name,
                "project_description": updated_project_description,
                "mail": updated_mail
            }
        }
    else:
        conn.close()
        return None




# def update_project(old_name, new_name=None, cost=None, expenditure=None, payment_status=None, payment_method=None, deadline=None, time_spent=None):
#     conn = sqlite3.connect('project_management.db')
#     cursor = conn.cursor()

#     # Ensure the project exists before updating
#     cursor.execute('SELECT name, cost, expenditure, payment_status, payment_method, deadline, total_time FROM projects WHERE name = ?', (old_name,))
#     project = cursor.fetchone()

#     if project:
#         # Extract existing values
#         existing_name, existing_cost, existing_expenditure, existing_payment_status, existing_payment_method, existing_deadline, existing_total_time = project

#         # Update fields with new values (if provided), else keep old values
#         updated_name = new_name if new_name else existing_name
#         updated_cost = existing_cost + cost if cost is not None else existing_cost
#         updated_expenditure = existing_expenditure + expenditure if expenditure is not None else existing_expenditure
#         updated_payment_status = payment_status if payment_status else existing_payment_status
#         updated_payment_method = payment_method if payment_method else existing_payment_method
#         updated_deadline = deadline if deadline else existing_deadline
#         updated_total_time = existing_total_time + time_spent if time_spent is not None else existing_total_time  # ✅ Add new time to total_time

#         # Perform update query
#         cursor.execute('''
#             UPDATE projects
#             SET name = ?, cost = ?, expenditure = ?, payment_status = ?, payment_method = ?, deadline = ?, total_time = ?
#             WHERE name = ?
#         ''', (updated_name, updated_cost, updated_expenditure, updated_payment_status, updated_payment_method, updated_deadline, updated_total_time, old_name))

#         conn.commit()
#         print(f"Project '{old_name}' updated successfully!")
#     else:
#         print(f"Warning: Project '{old_name}' does not exist. No update made.")

#     conn.close()



# Function to delete a project from the database
def delete_project(project_name):
    conn = sqlite3.connect('project_management.db')
    cursor = conn.cursor()

    # ✅ Delete associated tasks
    cursor.execute('DELETE FROM tasks WHERE project_name = ?', (project_name,))
    
    # ✅ Delete associated add amount
    cursor.execute('DELETE FROM add_amount WHERE project_name = ?', (project_name,))

    # ✅ Delete associated expenses
    cursor.execute('DELETE FROM expenses WHERE project_name = ?', (project_name,))

    # ✅ Delete associated feedback
    cursor.execute('DELETE FROM feedback WHERE project_name = ?', (project_name,))

    # ✅ Finally, delete the project itself
    cursor.execute('DELETE FROM projects WHERE name = ?', (project_name,))
    
     # ✅ Finally, delete the project metadata itself
    cursor.execute('DELETE FROM feedback_metadata WHERE project_name = ?', (project_name,))


    conn.commit()
    # ✅ Reset auto-increment IDs (Optional, if required)
    tables_to_reset = ["projects", "tasks", "add_amount", "expenses", "feedback", "feedback_metadata"]
    for table in tables_to_reset:
        cursor.execute(f"DELETE FROM sqlite_sequence WHERE name='{table}'")  # Resets auto-increment

    conn.commit()
    conn.close()


def add_task(project_name, task_name):
    conn = sqlite3.connect('project_management.db')
    cursor = conn.cursor()

    cursor.execute('''
    INSERT INTO tasks (project_name, task_name, total_time)
    VALUES (?, ?, ?)
    ''', (project_name, task_name, 0))  # Initial time = 0

    conn.commit()
    conn.close()


def fetch_tasks(project_name):
    conn = sqlite3.connect('project_management.db')
    cursor = conn.cursor()

    cursor.execute('''
    SELECT task_name, total_time FROM tasks WHERE project_name = ?
    ''', (project_name,))
    
    tasks = cursor.fetchall()
    conn.close()

    return [{"task_name": row[0], "total_time": row[1]} for row in tasks]


def update_task_time(project_name, task_name, new_total_time):
    conn = sqlite3.connect('project_management.db')
    cursor = conn.cursor()

    # Fetch existing task time
    cursor.execute('SELECT total_time FROM tasks WHERE project_name = ? AND task_name = ?', (project_name, task_name))
    existing_time = cursor.fetchone()

    if existing_time:
        updated_time = existing_time[0] + new_total_time  # ✅ Accumulate time correctly
        cursor.execute('''
        UPDATE tasks
        SET total_time = ?
        WHERE project_name = ? AND task_name = ?
        ''', (updated_time, project_name, task_name))
        
        # ✅ Update project total time (Sum all task times)
        cursor.execute('''
        UPDATE projects
        SET total_time = (SELECT SUM(total_time) FROM tasks WHERE project_name = ?)
        WHERE name = ?
        ''', (project_name, project_name))

    conn.commit()
    conn.close()
    
def update_total_time(project_name, new_total_time):
    conn = sqlite3.connect('project_management.db')
    cursor = conn.cursor()

    cursor.execute('''
    UPDATE projects 
    SET total_time = ? 
    WHERE name = ?
    ''', (new_total_time, project_name))

    conn.commit()
    conn.close()


def update_total_time(project_name, new_total_time):
    conn = sqlite3.connect('project_management.db')
    cursor = conn.cursor()

    cursor.execute('UPDATE projects SET total_time = ? WHERE name = ?', (new_total_time, project_name))
    conn.commit()
    conn.close()

def fetch_total_time_per_task(project_name):
    conn = sqlite3.connect('project_management.db')
    cursor = conn.cursor()

    cursor.execute('''
    SELECT task_name, total_time FROM tasks WHERE project_name = ?
    ''', (project_name,))

    tasks = cursor.fetchall()
    conn.close()

    return [{"task_name": task[0], "total_time": task[1] if task[1] else 0} for task in tasks]  # ✅ Ensure time is not None



# Function to fetch all projects and store them in session state
def fetch_all_projects():
    conn = sqlite3.connect('project_management.db')
    cursor = conn.cursor()

    # ✅ Ensure fetching latest total_time
    cursor.execute('''
    SELECT name, cost, expenditure, payment_status, payment_method, COALESCE(total_time, 0)
    FROM projects
    ORDER BY created_at DESC
    ''')
    
    projects = cursor.fetchall()
    conn.close()

    project_data = []
    for project in projects:
        project_data.append({
            "Project Name": project[0],
            "Cost": project[1],
            "Expenditure": project[2],
            "Payment Status": project[3],
            "Payment Method": project[4],
            "Total Time": project[5]  # ✅ Ensure this is always fetched
        })

    return project_data


#-----------------------------------------------------#
#Feedback

def create_feedback_table():
    conn = sqlite3.connect('project_management.db')
    cursor = conn.cursor()
    
    # ✅ Create a feedback table (linked to projects)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        project_name TEXT NOT NULL,
        client_name TEXT NOT NULL,
        rating INTEGER NOT NULL,
        comments TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (project_name) REFERENCES projects(name) ON DELETE CASCADE
    )
    ''')
    
    conn.commit()
    conn.close()


# ✅ Function to store feedback in the database
def add_feedback(project_name, client_name, rating, comments):
    conn = sqlite3.connect('project_management.db')
    cursor = conn.cursor()

    cursor.execute('''
    INSERT INTO feedback (project_name, client_name, rating, comments)
    VALUES (?, ?, ?, ?)
    ''', (project_name, client_name, rating, comments))

    conn.commit()
    conn.close()


# ✅ Function to fetch feedback for a project
def fetch_feedback(project_name):
    conn = sqlite3.connect('project_management.db')
    cursor = conn.cursor()

    cursor.execute('''
    SELECT client_name, rating, comments, created_at FROM feedback WHERE project_name = ?
    ''', (project_name,))
    
    feedbacks = cursor.fetchall()
    conn.close()

    return [{"client_name": row[0], "rating": row[1], "comments": row[2], "date": row[3]} for row in feedbacks]


# Function to create the table and store user feedback metadata
def store_user_feedback_metadata(project_name, ip_address, location, browser_info, referrer_url, device_info, battery_life):
    conn = sqlite3.connect("project_management.db")
    cursor = conn.cursor()

    # ✅ Check if this IP already submitted metadata for this project
    cursor.execute("""
        SELECT COUNT(*) FROM feedback_metadata
        WHERE project_name = ? AND ip_address = ?
    """, (project_name, ip_address))
    
    exists = cursor.fetchone()[0]  # If count > 0, data already exists

    if exists == 0:
        # ✅ Insert only if no previous record exists
        cursor.execute("""
            INSERT INTO feedback_metadata (project_name, ip_address, location, browser_info, referrer_url, device_info, battery_life)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (project_name, ip_address, location, browser_info, referrer_url, device_info, battery_life))
        
        conn.commit()

    conn.close()


# Fetch logs from the database
def fetch_feedback_logs():
    conn = sqlite3.connect('project_management.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM feedback_metadata ORDER BY created_at DESC')
    logs = cursor.fetchall()  # Returns a list of tuples
    conn.close()
    return logs



def fetch_location_per_client(project_name):
      # Connect to the SQLite database
    conn = sqlite3.connect('project_management.db')
    cursor = conn.cursor()

    # Query the database for location data
    cursor.execute('SELECT location FROM feedback_metadata WHERE project_name = ?', (project_name,))
    rows = cursor.fetchall()  # Fetch all results

    # Extract the first value from each tuple
    locations = [row[0] for row in rows]

    # Close the connection
    conn.close()

    return locations
  


def project_inform(project_name):
    # Connect to the SQLite database
    conn = sqlite3.connect('project_management.db')
    cursor = conn.cursor()

    # Query the database for project description
    cursor.execute('SELECT project_description FROM projects WHERE  name = ?', (project_name,))
    row = cursor.fetchone()  # Fetch only the first result

    # Close the connection
    conn.close()

    # Return the project description as a string (or None if no result is found)
    return row[0] if row else None
  
  

def get_locations_for_all(project_name):
    """
    Fetches locations for a given project name from the feedback_metadata table.
    """
    conn = sqlite3.connect('project_management.db')
    cursor = conn.cursor()
    cursor.execute('SELECT location FROM feedback_metadata WHERE project_name = ?', (project_name,))
    rows = cursor.fetchall()
    print(f"Query result for project '{project_name}': {rows}")  # Debugging line
    conn.close()
    return [row[0] for row in rows]


#=============DEADLINE=====================#

# def add_deadline_column():
#     conn = sqlite3.connect('project_management.db')
#     cursor = conn.cursor()
    
#     # Check if the column already exists (to avoid errors)
#     cursor.execute("PRAGMA table_info(projects);")
#     columns = [column[1] for column in cursor.fetchall()]
    
#     if "deadline" not in columns:
#         cursor.execute("ALTER TABLE projects ADD COLUMN deadline TEXT;")
#         conn.commit()
#         print("✅ 'deadline' column added successfully!")
#     else:
#         print("⚠️ 'deadline' column already exists.")
    
#     conn.close()

# # Run this function once
# add_deadline_column()




def get_project_details(project_name):
    conn = sqlite3.connect('project_management.db')
    cursor = conn.cursor()

    project_data = {}

    # ✅ Get all table names dynamically
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [table[0] for table in cursor.fetchall()]

    for table in tables:
        try:
            # ✅ Check if the table has a 'project_name' or 'name' column
            cursor.execute(f"PRAGMA table_info({table});")
            columns = [col[1] for col in cursor.fetchall()]

            project_column = "project_name" if "project_name" in columns else "name" if "name" in columns else None
            
            if project_column:
                # ✅ Fetch relevant data for the project
                cursor.execute(f"SELECT * FROM {table} WHERE {project_column} = ?", (project_name,))
                rows = cursor.fetchall()

                if rows:
                    # ✅ Store column names with data for better readability
                    project_data[table] = [dict(zip(columns, row)) for row in rows]  

        except Exception as e:
            print(f"Error fetching data from {table}: {e}")

    conn.close()
    return project_data if project_data else None



# add new expense 
def add_expense_to_db(project_name, expense_name, amount):
    conn = sqlite3.connect('project_management.db')
    cursor = conn.cursor()

    cursor.execute('''
    INSERT INTO expenses (project_name, expense_name, amount)
    VALUES (?, ?, ?)
    ''', (project_name, expense_name, amount))

    conn.commit()
    conn.close()

    # ✅ Update the project's total expenditure in the database
    update_project_expenditure(project_name)


def add_amount_rcvd(project_name, amount_received, paym_mode, pending_amount):
    """
    Add the received amount and payment method for a given project.
    """
    conn = sqlite3.connect('project_management.db')
    cursor = conn.cursor()

    # Insert data including the timestamp and payment method
    cursor.execute('''
    INSERT INTO add_amount (project_name, amount_received, pay_mode ,  pending_amount)
    VALUES (?, ?, ?, ?)
    ''', (project_name, amount_received, paym_mode,  pending_amount))

    conn.commit()
    conn.close()

def get_total_received(selected_item):
    conn = sqlite3.connect("project_management.db")  # Connect to your database
    cursor = conn.cursor()
    
    # Query to sum all received amounts for the selected item
    cursor.execute("SELECT SUM(amount_received) FROM add_amount WHERE project_name = ?", (selected_item,))
    result = cursor.fetchone()[0]  # Fetch the sum

    conn.close()  # Close connection

    return result if result else 0  # Return 0 if no records found
def fetch_expenses(project_name):
    conn = sqlite3.connect('project_management.db')
    cursor = conn.cursor()

    # Fetch expenses with their IDs
    cursor.execute('''
    SELECT id, expense_name, amount, created_at FROM expenses 
    WHERE project_name = ? ORDER BY created_at ASC
    ''', (project_name,))
    
    expenses = cursor.fetchall()
    conn.close()

    # Return expenses as a list of dictionaries, now including 'id' and 'Date'
    return [
        {"Date": row[3] ,  "Expenditure": row[1], "Amount": row[2]}
        for row in expenses
    ]



def update_project_expenditure(project_name):
    conn = sqlite3.connect('project_management.db')
    cursor = conn.cursor()

    # ✅ Calculate total expenses for the selected project ONLY
    cursor.execute('''
    SELECT SUM(amount) FROM expenses WHERE project_name = ?
    ''', (project_name,))
    
    total_exp = cursor.fetchone()[0] or 0  # Default to 0 if no expenses

    # ✅ Update the expenditure column in the `projects` table
    cursor.execute('''
    UPDATE projects SET expenditure = ? WHERE name = ?
    ''', (total_exp, project_name))

    conn.commit()
    conn.close()


# def migrate_expenses_table():
#     conn = sqlite3.connect('project_management.db')
#     cursor = conn.cursor()

#     # Step 1: Create new expenses table with project_name
#     cursor.execute('''
#     CREATE TABLE IF NOT EXISTS expenses_new (
#         id INTEGER PRIMARY KEY AUTOINCREMENT,
#         project_name TEXT NOT NULL,  -- ✅ Project name is required
#         expense_name TEXT NOT NULL,
#         amount REAL NOT NULL,
#         created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
#         FOREIGN KEY (project_name) REFERENCES projects(name) ON DELETE CASCADE
#     )
#     ''')

#     # Step 2: Assign a default project name while copying data
#     cursor.execute('''
#     INSERT INTO expenses_new (id, project_name, expense_name, amount, created_at)
#     SELECT id, 
#            'Unknown' AS project_name,  -- ✅ Assign a default project name
#            expense_name, 
#            amount, 
#            created_at 
#     FROM expenses
#     ''')

#     # Step 3: Drop old table
#     cursor.execute('DROP TABLE expenses')

#     # Step 4: Rename new table to original name
#     cursor.execute('ALTER TABLE expenses_new RENAME TO expenses')

#     conn.commit()
#     conn.close()



def delete_expense(expense_id):

    conn = sqlite3.connect('project_management.db')
    cursor = conn.cursor()
    cursor.execute("DELETE FROM expenses WHERE id = ?", (expense_id,))
    conn.commit()
    conn.close()

def update_expense(project_name, old_name, new_name, new_amount):
    conn = sqlite3.connect('project_management.db')
    cursor = conn.cursor()

    try:
        # Fetch the old expense amount
        cursor.execute(
            "SELECT amount FROM expenses WHERE project_name = ? AND expense_name = ?",
            (project_name, old_name)
        )
        old_expense = cursor.fetchone()
        
        if old_expense is None:
            print("Expense not found!")
            return
        
        old_amount = old_expense[0]  # Extract old amount
        
        # Update the `expenses` table
        cursor.execute(
            "UPDATE expenses SET expense_name = ?, amount = ? WHERE project_name = ? AND expense_name = ?",
            (new_name, new_amount, project_name, old_name)
        )

        # Calculate the difference in expenditure
        amount_difference = new_amount - old_amount

        # Update the `projects` table expenditure
        cursor.execute(
            "UPDATE projects SET expenditure = expenditure + ? WHERE name = ?",
            (amount_difference, project_name)
        )

        conn.commit()
        print("Expense and project expenditure updated successfully!")

    except sqlite3.Error as e:
        print("Error updating expense:", e)
        conn.rollback()
    
    finally:
        conn.close()


# def fetch_expenses(project_name):
#     conn = sqlite3.connect('project_management.db')
#     cursor = conn.cursor()
#     cursor.execute("SELECT id, expense_name, amount FROM expenses WHERE project_name = ? ORDER BY created_at ASC", (project_name,))
#     expenses = cursor.fetchall()
#     conn.close()
#     return [{"ID": row[0], "Expenditure": row[1], "Amount": row[2]} for row in expenses]



def fetch_project_details(project_name):
    """Fetch project details including all fields from the database"""
    conn = sqlite3.connect('project_management.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT name, cost, expenditure, payment_status, payment_method, total_time, deadline, 
               technology, project_type, sector_name, project_description, mail, status
        FROM projects 
        WHERE name = ?
    ''', (project_name,))
    
    project = cursor.fetchone()
    conn.close()

    if project:
        return {
            "name": project[0],
            "cost": project[1],
            "exp": project[2],
            "payment_status": project[3],
            "payment_method": project[4],
            "total_time": project[5] if project[5] is not None else 0,
            "deadline": project[6],
            "technology": json.loads(project[7]) if project[7] else [],  # Convert JSON to list
            "project_type": json.loads(project[8]) if project[8] else [],  # Convert JSON to list
            "sector_name": project[9] if project[9] else "",  # String field
            "project_description": project[10] if project[10] else "",  # String field
            "mail": project[11] if project[11] else "",  # ✅ Added mail field
            "status": project[12] if project[12] else "In Progress"  # ✅ Added status field
        }
    
    return {}


# ✅ Return an empty dictionary if the project is not found
# conn = sqlite3.connect('project_management.db')
# cursor = conn.cursor()

# # ✅ Add status column if not exists
# cursor.execute("ALTER TABLE projects ADD COLUMN status TEXT DEFAULT 'In Progress'")

# conn.commit()
# conn.close()



def update_project_status_in_db(project_name, new_status):
    conn = sqlite3.connect("project_management.db")
    cursor = conn.cursor()

    # ✅ Fetch existing project details
    cursor.execute('SELECT completion_date FROM projects WHERE name = ?', (project_name,))
    stored_completion_date = cursor.fetchone()

    # ✅ Set completion date only if marking as "Completed" and date isn't already set
    if new_status == "Completed" and (not stored_completion_date or stored_completion_date[0] is None):
        completion_date = datetime.today().strftime('%Y-%m-%d')
    else:
        completion_date = stored_completion_date[0] if stored_completion_date else None

    # ✅ Update project status and completion date in the database
    cursor.execute('''
        UPDATE projects 
        SET status = ?, completion_date = ? 
        WHERE name = ?
    ''', (new_status, completion_date, project_name))

    conn.commit()
    conn.close()




def get_amount_details(selected_project):
    conn = sqlite3.connect('project_management.db')
    cursor = conn.cursor()

    # Fetch the project cost from the `projects` table
    cursor.execute('''
    SELECT cost FROM projects WHERE name = ?
    ''', (selected_project,))
    project_cost = cursor.fetchone()

    if not project_cost:
        conn.close()
        return []  # Return an empty list if the project doesn't exist

    project_cost = project_cost[0]  # Extract the cost value

    # Fetch received amounts for the selected project
    cursor.execute('''
    SELECT created_at, amount_received , pay_mode ,  pending_amount FROM add_amount
    WHERE project_name = ?
    ''', (selected_project,))

    rows = cursor.fetchall()
    conn.close()

    # Debugging: Print database rows
    print("Database Rows:", rows)  

    # Sum all received amounts
    total_received = sum(row[1] for row in rows)  

    # Calculate pending amount
    pending_amount = project_cost - total_received  

    # Convert to a list of dictionaries
    data = [
        {
            "Date": row[0],  
            "Project Cost": project_cost,  
            "Amount Received": row[1],  # Ensure this matches the column name  
            'Payment Mode': row[2],  # Corrected index for `Pay Mode`
            "Pending Amount": row[3]  # Keep this for remaining balance
        }
        for row in rows
    ]

    return data , pending_amount






def get_project_cost(selected_project):
    conn = sqlite3.connect('project_management.db')
    cursor = conn.cursor()

    cursor.execute('SELECT cost FROM projects WHERE name = ?', (selected_project,))
    project_cost = cursor.fetchone()

    conn.close()

    # Convert project_cost to float (or int if necessary)
    return float(project_cost[0]) if project_cost else 0.0


def update_amount_rcvd(project_name, date, new_amount, new_paym_mode):
    conn = sqlite3.connect('project_management.db')
    cursor = conn.cursor()

    # Update amount received and payment mode
    cursor.execute('''
    UPDATE add_amount
    SET amount_received = ?, pay_mode = ?
    WHERE project_name = ? AND created_at = ?
    ''', (new_amount, new_paym_mode, project_name, date))

    conn.commit()

    # Fetch the latest total received amount
    cursor.execute('''
    SELECT SUM(amount_received) FROM add_amount WHERE project_name = ?
    ''', (project_name,))
    total_received = cursor.fetchone()[0] or 0.0

    conn.close()

    return total_received
def update_pending_amount(project_name, new_pending_amount):
    conn = sqlite3.connect('project_management.db')
    cursor = conn.cursor()

    cursor.execute('''
    UPDATE add_amount
    SET pending_amount = ?
    WHERE project_name = ?
    ''', (new_pending_amount, project_name))

    conn.commit()
    conn.close()


def delete_amount_rcvd(project_name, date):
    """
    Delete the received amount for a given project and date.
    """
    conn = sqlite3.connect('project_management.db')
    cursor = conn.cursor()

    # Delete the record
    cursor.execute('''
    DELETE FROM add_amount
    WHERE project_name = ? AND created_at = ?
    ''', (project_name, date))

    conn.commit()
    conn.close()

def get_projects_progress_bar():
    """Fetch all projects from the database."""
    conn = sqlite3.connect('project_management.db')
    cursor = conn.cursor()
    cursor.execute("SELECT id, name,status,created_at, deadline,completion_date FROM projects")
    projects = cursor.fetchall()
    conn.close()
    return projects



def get_project_summary():

    conn = sqlite3.connect('project_management.db')
    cursor = conn.cursor()

    # Total Expense from the expenses table
    cursor.execute("SELECT SUM(amount) FROM expenses")
    total_expense = cursor.fetchone()[0] or 0

    # Total Received from the add_amount table
    cursor.execute("SELECT SUM(amount_received) FROM add_amount")
    total_received = cursor.fetchone()[0] or 0

    # Profit is the difference between total received and total expense
    total_profit = total_received - total_expense

    # Total Time Spent from the projects table (assuming it is updated by tasks)
    cursor.execute("SELECT SUM(total_time) FROM projects")
    total_time_spent = cursor.fetchone()[0] or 0

    # Projects Completed (assuming a 'status' column with value 'Completed')
    cursor.execute("SELECT COUNT(*) FROM projects WHERE status = 'Completed'")
    projects_completed = cursor.fetchone()[0] or 0

    conn.close()

    return {
        "Total Profit": total_profit,
        "Total Expense": total_expense,
        "Total Time Spent": total_time_spent,
        "Projects Completed": projects_completed
    }
    

def parse_date(date_str):
    """Parse a date string into a datetime object."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except (ValueError, TypeError):
        return None

def get_project_progress():
    """Fetch project progress data from the database."""
    today = datetime.now()

    # Use context manager to handle database connection safely
    with sqlite3.connect('project_management.db') as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, name, created_at, deadline, status
            FROM projects
        ''')
        projects = cursor.fetchall()

    project_progress = {}
    for project in projects:
        project_id, name, created_at, deadline, status = project

        # Parse dates
        created_at = parse_date(created_at)
        deadline = parse_date(deadline)

        if not created_at or not deadline:
            continue  # Skip projects with missing or invalid dates

        # Calculate progress based on elapsed days
        total_days = max(1, (deadline - created_at).days)  # Avoid division by zero
        elapsed_days = max(0, (today - created_at).days)

        progress = min(100, max(0, (elapsed_days / total_days) * 100))

        # Store project details
        project_progress[project_id] = {
            "name": name,
            "progress": progress,
            "status": status,
            "deadline": deadline
        }

    return project_progress


# Connect to database
def get_db_connection():
    return sqlite3.connect("project_management.db", check_same_thread=False)

# Function to fetch monthly revenue
def fetch_monthly_revenue():
    conn = get_db_connection()
    query = """
    SELECT strftime('%Y-%m', created_at) AS month, SUM(cost) AS total_cost
    FROM projects
    GROUP BY month
    ORDER BY month;
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


import sqlite3
import streamlit as st

# Function to fetch project cost
def get_project_cost(selected_project):
    conn = sqlite3.connect('project_management.db')
    cursor = conn.cursor()
    
    cursor.execute('SELECT cost FROM projects WHERE name = ?', (selected_project,))
    project_cost = cursor.fetchone()

    conn.close()

    return float(project_cost[0]) if project_cost else 0.0  # Ensure it's a float


# Function to update payment status in the database
def update_payment_status(selected_project, received_amount, project_cost, pending_amount):
    # Check the status based on the pending amount
    if pending_amount == 0:
        status = "Paid"  # If no pending amount, set as Paid
    elif received_amount == 0:
        status = "Unpaid"  # If no received amount, set as Unpaid
    else:
        status = "Partial Paid"  # Otherwise, set as Partial Paid

    # Connect to the SQLite database
    conn = sqlite3.connect('project_management.db')
    cursor = conn.cursor()

    # Update the payment status in the database
    cursor.execute('''
        UPDATE projects 
        SET payment_status = ? 
        WHERE name = ?
    ''', (status, selected_project))

    # Commit the changes and close the connection
    conn.commit()
    conn.close()

    return status



def ticket_size():
    conn = sqlite3.connect("project_management.db")
    cursor = conn.cursor()

    # Step 1: Add the 'ticket_size' column if it doesn't exist
    # try:
    #     cursor.execute("ALTER TABLE projects ADD COLUMN ticket_size TEXT;")
    # except sqlite3.OperationalError:
    #     pass  # Ignore if the column already exists


    # Step 2: Function to categorize ticket size
    def categorize_ticket_size(cost):
        """Returns ticket size based on the cost."""
        if cost <= 20000:
            return "Low Ticket"
        elif 20000 < cost <= 50000:
            return "Mid Ticket"
        else:
            return "High Ticket"

    # Step 3: Update existing records
    cursor.execute("SELECT id, cost FROM projects")
    projects = cursor.fetchall()

    for project_id, cost in projects:
        ticket_size = categorize_ticket_size(cost)
        cursor.execute("UPDATE projects SET ticket_size = ? WHERE id = ?", (ticket_size, project_id))

    # Step 4: Create a trigger to automatically update ticket_size for future inserts
    cursor.execute('''
        CREATE TRIGGER IF NOT EXISTS set_ticket_size
        AFTER INSERT ON projects
        FOR EACH ROW
        BEGIN
            UPDATE projects
            SET ticket_size = 
                CASE 
                    WHEN NEW.cost <= 20000 THEN 'Low Ticket'
                    WHEN NEW.cost > 20000 AND NEW.cost <= 50000 THEN 'Mid Ticket'
                    ELSE 'High Ticket'
                END
            WHERE id = NEW.id;
        END;
    ''')

    # Commit changes and close the connection
    conn.commit()
    conn.close()
    
    
# Function to fetch ticket size from the databas


def get_ticket_size(name):
    conn = sqlite3.connect("project_management.db")
    cursor = conn.cursor()
    
    # ✅ Fix: Ensure single-element tuple by adding a comma (name,)
    cursor.execute("SELECT ticket_size FROM projects WHERE name = ?", (name,))  
    result = cursor.fetchone()
    
    conn.close()
    
    return result[0].strip() if result else "No Data"  # Ensure correct formatting

    


# Initialize database connection
conn = sqlite3.connect("ai_responses.db", check_same_thread=False)
cursor = conn.cursor()

# Create table if not exists
cursor.execute("""
CREATE TABLE IF NOT EXISTS text_improvements (
    selected_item TEXT,
    description TEXT,
    improved_text TEXT,
    PRIMARY KEY (selected_item, description)
)
""")
conn.commit()

def get_saved_text(selected_item, description):
    """Check if the improved text already exists in the database."""
    cursor.execute(
        "SELECT improved_text FROM text_improvements WHERE selected_item = ? AND description = ?", 
        (selected_item, description)
    )
    result = cursor.fetchone()
    return result[0] if result else None

def save_text(selected_item, description, improved_text):
    """Save the improved text into the database."""
    cursor.execute(
        "INSERT OR REPLACE INTO text_improvements (selected_item, description, improved_text) VALUES (?, ?, ?)", 
        (selected_item, description, improved_text)
    )
    conn.commit()