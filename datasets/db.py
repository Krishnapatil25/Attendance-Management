import sqlite3
conn=sqlite3.connect('candidates.db')
#conn.execute("CREATE TABLE product(id INTEGER PRIMARY KEY AUTOINCREMENT,FULLNAME TEXT,USID TEXT,PASSWORD TEXT)")
#print("table created")

#conn=sqlite3.connect('Attendance.db')
#conn.execute("CREATE TABLE Sheet1(id INTEGER PRIMARY KEY AUTOINCREMENT,FULLNAME TEXT,STATUS TEXT,DATE TEXT,TIME TEXT,TEMP TEXT)")
cur=conn.execute("SELECT * FROM Sheet1")
print(cur.fetchall())