# 用 pymysql 连接mysql 数据库，并进行基本操作


''' 步骤 1：连接 MySQL 数据库 '''

import pymysql
 
try:
    # 连接 MySQL 数据库
    conn = pymysql.connect(
        host='localhost',  # 主机名
        port=3306,             # 端口号，MySQL默认为3306
        user='root',           # 用户名
        password='Qw242015',       # 密码
        database='nb1',         # 数据库名称
        charset='utf8mb4'      # 指定字符集
    )
    print("数据库连接成功\n")
except pymysql.MySQLError as e:
    print(f"数据库连接失败：{e}\n")


''' 步骤2: 查询所有用户信息'''
try:
    with conn.cursor() as cursor:
        # SQL 查询语句
        table_name = "user1"
        sql = f"SELECT * FROM {table_name}"
        cursor.execute(sql)
        # 获取所有记录列表
        results = cursor.fetchall()
        print(f"查询当前{table_name}表用户信息：")
        for row in results:
            print(row)
finally:
    conn.close()