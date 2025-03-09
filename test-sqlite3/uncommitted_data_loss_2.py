import sqlite3


def demonstrate_uncommitted_data_loss(db_name="test.db"):
    # Step 1: Insert data without committing, then close connection
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Create a test table
    cursor.execute("DROP TABLE IF EXISTS test_table")
    cursor.execute("CREATE TABLE test_table (id INTEGER PRIMARY KEY, data TEXT)")
    print("Database and table created.")

    # Insert data but do not commit
    cursor.execute("INSERT INTO test_table (data) VALUES ('Uncommitted data')")
    print("Data inserted, but not committed.")


def demonstrate_uncommitted_data_loss_2(db_name="test.db"):
    # Step 2: Reopen the database in a new connection and attempt to read the data
    conn2 = sqlite3.connect(db_name)
    cursor2 = conn2.cursor()
    print("Reopened connection. Attempting to read data...")

    cursor2.execute("SELECT * FROM test_table")
    rows = cursor2.fetchall()

    if rows:
        print("Read succeeded. Retrieved rows:", rows)
    else:
        print("Read found no rows. Uncommitted data was not written to disk.")

    # Cleanup second connection
    conn2.close()


# Run the demonstration
demonstrate_uncommitted_data_loss()
demonstrate_uncommitted_data_loss_2()
