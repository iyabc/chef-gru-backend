import sqlite3

def get_database_connection():
    return sqlite3.connect('evaluations.db')

def create_table():
    conn = get_database_connection()
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS evaluations_table (
        id INTEGER PRIMARY KEY,
        contact_number TEXT,
        years_experience INTEE,
        user_input TEXT,
        recipe_output TEXT,
        clarity_rating INTEGER,
        creativity_rating INTEGER,
        suitability_rating INTEGER,
        doability_rating INTEGER,
        likelihood_to_try_rating INTEGER,
        overall_rating INTEGER
    )
    ''')
    conn.commit()
    conn.close()

def view_table(table_name): 
    conn = get_database_connection()
    cursor = conn.cursor()
    cursor.execute(f'SELECT * FROM {table_name}')
    result = cursor.fetchall()
    print(result)
    conn.close()

def insert_row(contact_number, years_experience, user_input, recipe_output, clarity_rating, creativity_rating, suitability_rating, doability_rating, likelihood_to_try_rating, overall_rating):  
    conn = get_database_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO evaluations_table (contact_number, years_experience, user_input, recipe_output, clarity_rating, creativity_rating, suitability_rating, doability_rating, likelihood_to_try_rating, overall_rating) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
               (contact_number, years_experience, user_input, recipe_output, clarity_rating, creativity_rating, suitability_rating, doability_rating, likelihood_to_try_rating, overall_rating))
    conn.commit()
    conn.close()

def delete_table(table_name):
    conn = sqlite3.connect('evaluations.db')
    cursor = conn.cursor()

    cursor.execute(f"DROP TABLE IF EXISTS {table_name}")

    conn.commit()
    conn.close()