import mariadb
import sys

def connecting_mariadb():
    conn_params = {
        'user': "user",
        'password': "user",
        'host': "localhost",  
        'port': 3306,
        'database': "avismariadb"
    }
    try:
        connection = mariadb.connect(**conn_params)
        cursor = connection.cursor()
        print("Connected to MariaDB successfully.")
        return connection, cursor
    except mariadb.Error as e:
        print(f"Error connecting to MariaDB: {e}")
        sys.exit(1)

def disconnecting_mariadb(connection):
    """
    Close the connection to the MariaDB database.
    """
    try:
        if connection:
            connection.close()
            print("Disconnected from MariaDB successfully.")
    except mariadb.Error as e:
        print(f"Error disconnecting from MariaDB: {e}")

def show_tables(cursor):
    """
    Show all tables in the connected database.
    """
    try:
        cursor.execute("SHOW TABLES;")
        tables = cursor.fetchall()
        print("Tables in the database:")
        for table in tables:
            print(table[0])
    except mariadb.Error as e:
        print(f"Error showing tables: {e}")

def fetch_subject(cursor):
    """
    Fetch all rows from the `subject` table and display them.
    """
    try:
        cursor.execute("SELECT * FROM `subject`;")
        subjects = cursor.fetchall()
        print("Subjects in the database:")
        for sub_id, sub_name in subjects:
            print(f"ID: {sub_id}, Name: {sub_name}")
    except mariadb.Error as e:
        print(f"Error fetching subjects: {e}")

