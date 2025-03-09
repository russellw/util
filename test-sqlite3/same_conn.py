import sqlite3


def create_test_database(db_name="test.db"):
    # Create a new database connection and cursor
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Create a test table
    cursor.execute("DROP TABLE IF EXISTS test_table")
    cursor.execute("CREATE TABLE test_table (id INTEGER PRIMARY KEY, data TEXT)")
    print("Database and table created.")

    # Insert data without committing
    cursor.execute("INSERT INTO test_table (data) VALUES ('Uncommitted data')")
    print("Data inserted, but not committed.")

    # Attempt to read the uncommitted data
    print("Attempting to read uncommitted data...")
    try:
        cursor2 = conn.cursor()
        cursor2.execute("SELECT * FROM test_table")
        rows = cursor2.fetchall()

        if rows:
            print("Read succeeded. Retrieved rows:", rows)
        else:
            print("Read succeeded, but no rows found.")
    except sqlite3.DatabaseError as e:
        print("Error reading from the database:", e)
    finally:
        # Cleanup
        conn.close()


create_test_database()
