# 2. 使用 mysql-connector-python 驱动连接数据库

import mysql.connector
from mysql.connector import Error

try:
    # 创建数据库连接
    db = mysql.connector.connect(
        host="localhost",  # MySQL服务器地址
        user="root",       # 用户名
        password="Qw242015",  # 密码
        database="nb1"     # 数据库名称
    )

    # 检查数据库连接是否成功
    if db.is_connected():
        print("成功连接到数据库")

    # 创建游标对象，用于执行SQL查询
    cursor = db.cursor()

except Error as e:
    print(f"数据库连接错误: {e}")

