import pymysql
import pandas as pd
from pymysql import OperationalError, connect


def read_table_data(file_path, column_mapping):
    df = pd.read_excel(file_path)
    valid_columns = [col for col in df.columns if col in column_mapping]
    df = df[valid_columns].rename(columns=column_mapping)

    # 新增NaN值处理逻辑
    def handle_nan(value, column_type):
        if pd.isna(value):
            if column_type in ['int', 'bigint', 'smallint']:  # 数值型字段填充0
                return 0
            elif column_type in ['varchar', 'text', 'datetime']:  # 字符串/时间字段填充空值
                return '' if column_type != 'datetime' else None  # datetime字段填充None
            else:  # 其他类型默认填充None（需根据数据库表结构调整）
                return None
        return value

    # 假设已知数据库表各列的数据类型（需根据实际表结构修改）
    column_types = {
        'job_title': 'varchar',
        'location': 'varchar',
        'company_name': 'varchar',
        'degree': 'varchar',
        'experience': 'varchar',
        'skill': 'varchar',
        'salary_range': 'varchar',  # 假设薪资是字符串类型，若为数值型需调整
        'welfare': 'varchar',
        'description': 'text',
        'industry': 'varchar'
    }

    # 按列类型处理NaN值
    for col in df.columns:
        col_type = column_types.get(col, 'varchar')  # 获取列类型，默认按字符串处理
        df[col] = df[col].apply(lambda x: handle_nan(x, col_type))

    return df


def connect_mysql():
    connection = connect(
        host='localhost',
        port=3306,
        user='root',
        password='TYH041113',
        database='jobrec',
        charset='utf8mb4',
        connect_timeout=10,
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=False
    )
    connection.ping(reconnect=True)
    return connection


def insert_data_to_mysql(dataframe, connection, table_name, batch_size=500):
    if dataframe is None or connection is None:
        return

    db_columns = dataframe.columns.tolist()
    sql = f"INSERT INTO {table_name} ({', '.join(db_columns)}) VALUES ({', '.join(['%s'] * len(db_columns))})"

    cursor = connection.cursor()
    total_rows = len(dataframe)

    for i in range(0, total_rows, batch_size):
        batch = dataframe.iloc[i:i + batch_size]
        values = [tuple(row[col] for col in db_columns) for _, row in batch.iterrows()]

        try:
            cursor.executemany(sql, values)
            connection.commit()
            print(f"成功插入第 {i + 1}-{min(i + batch_size, total_rows)} 条数据")
        except OperationalError as e:
            if "2013" in str(e):
                print(f"检测到连接丢失，尝试重新连接...")
                connection = connect_mysql()
                cursor = connection.cursor()
                cursor.executemany(sql, values)
                connection.commit()
            else:
                print(f"插入错误：{e}，当前批次数据量：{len(values)}")
                connection.rollback()
        except Exception as e:
            print(f"未知错误：{e}，错误数据样例：{values[0] if values else '空批次'}")
            connection.rollback()

    cursor.close()


if __name__ == "__main__":
    file_path = 'bossData.xlsx'
    table_name = 'jobSys'
    column_mapping = {
        '职位': 'job_title',
        '城市': 'location',
        '公司': 'company_name',
        '学历': 'degree',
        '经验': 'experience',
        '技能': 'skill',
        '薪资': 'salary_range',
        '福利': 'welfare',
        '工作描述': 'description',
        '行业': 'industry'
    }

    df = read_table_data(file_path, column_mapping)
    conn = connect_mysql()

    if df is not None and conn is not None:
        insert_data_to_mysql(df, conn, table_name)
        conn.close()
        print("数据插入完成！")
