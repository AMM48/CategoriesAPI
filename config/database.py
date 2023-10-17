import mysql.connector

def get_db():
    db = mysql.connector.connect(host="localhost", user="root", passwd="", database="budget_tracker")

    return db