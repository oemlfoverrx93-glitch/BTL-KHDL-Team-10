import pandas as pd
from sqlalchemy import create_engine
import urllib

server = r'DESKTOP-GNEK8B3\MSSQLSERVER01'  # Thay bằng tên Server của bạn trong SSMS (VD: '.' hoặc 'localhost')
database = 'Dudoangianha'    # Thay bằng tên Database bạn muốn import vào

connection_string = f"Driver={{ODBC Driver 17 for SQL Server}};Server={server};Database={database};Trusted_Connection=yes;"

params = urllib.parse.quote_plus(connection_string)
engine = create_engine(f"mssql+pyodbc:///?odbc_connect={params}")

print("Đang đọc file CSV...")
df_train = pd.read_csv('train.csv', keep_default_na=False)
df_test = pd.read_csv('test.csv', keep_default_na=False)

df_train.replace("", None, inplace=True)
df_test.replace("", None, inplace=True)

print("Đang import dữ liệu vào bảng ames_train...")

df_train.to_sql('ames_train', con=engine, if_exists='replace', index=False)
print("Import thành công ames_train!")

print("Đang import dữ liệu vào bảng ames_test...")
df_test.to_sql('ames_test', con=engine, if_exists='replace', index=False)
print("Import thành công ames_test!")

print("HOÀN TẤT TOÀN BỘ QUÁ TRÌNH!")