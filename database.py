import sqlite3

def get_database_connection():
    return sqlite3.connect('evaluations.db')

def create_table():
    conn = get_database_connection()
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS evaluations_table (
        id INTEGER PRIMARY KEY,
        name TEXT,
        email TEXT,
        contact_number TEXT,
        profession TEXT,
        scores TEXT
    )
    ''')
    conn.commit()
    conn.close()

def view_table(): 
    conn = get_database_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM evaluations_table')
    result = cursor.fetchall()
    print(result)
    conn.close()

def insert_row(name, email, contact_number, profession, scores):  
    conn = get_database_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO evaluations_table (name, email, contact_number, profession, scores) VALUES (?, ?, ?, ?, ?)",
               (name, email, contact_number, profession, str(scores)))
    conn.commit()
    conn.close()

def delete_table(table_name):
    conn = sqlite3.connect('evaluations.db')
    cursor = conn.cursor()

    cursor.execute(f"DROP TABLE IF EXISTS {table_name}")

    conn.commit()
    conn.close()