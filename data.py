import pymysql
from openpyxl import Workbook
from openpyxl import load_workbook

con = pymysql.connect(host='localhost', user='root', password='epdlxj', db='testdb', charset='utf8')
cur = con.cursor()

sql = "select * from sabjill"

cur.execute(sql)
for i in cur:
    print(i)