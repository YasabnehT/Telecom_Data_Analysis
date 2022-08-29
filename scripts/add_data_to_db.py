#%%
import os
from tkinter import E
import pandas as pd
import mysql.connector as mysql
from mysql.connector import Error

def DBConnector(dbName = None):
    conn = mysql.connect(host = 'localhost', user = 'root', password = 'YasTesh@2123#', database = dbName, buffered =True)
    curs = conn.cursor()
    return conn, curs
def TelecomDB(dbName:str)->None:
    conn,curs = DBConnector(dbName)
    query = f"alter database {dbName} CHARACTER SET = utf8mb4 COLLATE = utf8mb4_unicode_ci;"
    curs.execute(query)
    conn.commit()
def createDB(dbName:str) -> None:
    conn, curs = DBConnector()
    curs.execute(f"create database if not exists {dbName};")
    conn.commit()
    curs.close()
    
def createTables(dbName:str) -> None:
    conn,curs = DBConnector(dbName)
    db_file = open("scripts/db.sql", 'r')
    readSQL = db_file.read()
    db_file.close()
    # db_file = open("db_schema.sql", 'r').read()
    
    sql_commands = readSQL.split(';')
    
    for command in sql_commands:
        try:
            res = curs.execute(command)
        except Exception as ex:
            print('Execution failed: ', command)
            print(ex)
    conn.commit()
    curs.close()
    return 
# def preprocess_df(df:pd.DataFrame) -> pd.DataFrame:
#     columns_2_drop = ['id_str','truncated', 'truncated','Unnamed: 0', 'timestamp', 'sentiment', 'possibly_sensitive', 'original_text']
#     try:
#         df = df.drop(columns=columns_2_drop, axis =1)
#         df = df.fillna(0)
#     except KeyError as e:
#         print('An error occured: ', e)
#     return df
def insert_to_table(dbName:str, df:pd.DataFrame, table_name:str) -> None:
    conn,curs = DBConnector(dbName)
    # df = preprocess_df(df)
    for _,row in df.iterrows():
    
        query = f"""insert into {table_name}
        ('MSISDN_Number','Engagement_Score_Dur, Engagement_Score_Total,
        Engagement_Satisf_Score, Experience_Score_Total_TCP, Experience_Score_Total_RTT,
        Experience_Score_Total_TP, Experience_Satisf_Score)
        values (%d,%s, %s, %s, %s, %s, %s, %s);"""
        data = (row[0], row[1], row[2], row[3], (row[4]), (row[5]), row[6], row[7])
        try:
            curs.execute(query, data)
            conn.commit()
            print("Data insertion successfull")
        except Exception as e:
            conn.rollback()
            print("Error: ", e)
        return
def db_fetch_data(*args, many = False, table_name = '', rdf = True, **kwargs) -> pd.DataFrame:
        conn, curs1 = DBConnector(*args)
        if many:
            curs1.executemany(*args)
        else:
            curs1.execute(*args)
        #columns
        field_names = [i[0] for i in curs1.description]
        # column values
        res = curs1.fetchall()
        nrows = curs1.rowcount()
        if table_name:
            print(f"{nrows} records fetched from {table_name} table")
        curs1.close()
        conn.close()
            
        if rdf: # row as dataframe
            return pd.DataFrame(res, columns=field_names)
        else:
            return res
        
        
if __name__ == "__main__":
    # createDB(dbName='TweetsDB')
    TelecomDB(dbName='TelecomDB')
    createTables(dbName = 'TelecomDB')
    
    df = pd.read_csv('data/User_Scores.csv')
    
    insert_to_table(dbName='TelecomDB', df = df, 
                    table_name='Scores')

# %%
