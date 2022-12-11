import sqlite3
conn=sqlite3.connect('profiles.db')
conn.execute("CREATE TABLE Sheet2(id INTEGER PRIMARY KEY AUTOINCREMENT,FULLNAME TEXT,USID TEXT,PASSWORD TEXT)")
print("table created")