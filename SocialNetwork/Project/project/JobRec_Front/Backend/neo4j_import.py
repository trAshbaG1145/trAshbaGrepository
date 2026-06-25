"""
Neo4j 知识图谱数据导入脚本
从 MySQL jobSys 表读取岗位数据，导入到 Neo4j 图数据库
"""
import pymysql
from kg_config import get_graph
import re
import sys


def connect_mysql():
    """连接 MySQL 数据库（Docker 映射端口 3413）"""
    connection = pymysql.connect(
        host='localhost',
        port=3413,
        user='root',
        password='MySQL@999999',
        database='jobrec',
        charset='utf8mb4',
        connect_timeout=10,
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True
    )
    connection.ping(reconnect=True)
    return connection


def split_skills(skill_str):
    """将技能字符串拆分为列表"""
    if not skill_str:
        return []
    # 按逗号、顿号、空格等分隔
    skills = re.split(r'[,，、\s/]+', skill_str)
    return [s.strip() for s in skills if s.strip() and len(s.strip()) > 1]


def import_to_neo4j():
    """从 MySQL 读取数据并导入 Neo4j"""
    print("Connecting to MySQL...")
    mysql_conn = connect_mysql()

    print("Connecting to Neo4j...")
    graph = get_graph()

    # 清空现有数据
    print("Clearing existing Neo4j data...")
    graph.run("MATCH (n) DETACH DELETE n")

    # 读取所有岗位数据
    print("Reading job data from MySQL...")
    cursor = mysql_conn.cursor()
    cursor.execute("SELECT * FROM jobSys WHERE status = '1'")
    jobs = cursor.fetchall()
    print(f"Found {len(jobs)} job records")

    # 创建约束和索引
    print("Creating constraints...")
    try:
        graph.run("CREATE CONSTRAINT IF NOT EXISTS FOR (j:Job) REQUIRE j.name IS UNIQUE")
        graph.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Company) REQUIRE c.name IS UNIQUE")
        graph.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:City) REQUIRE c.name IS UNIQUE")
        graph.run("CREATE CONSTRAINT IF NOT EXISTS FOR (s:Skill) REQUIRE s.name IS UNIQUE")
        graph.run("CREATE CONSTRAINT IF NOT EXISTS FOR (d:Degree) REQUIRE d.name IS UNIQUE")
        graph.run("CREATE CONSTRAINT IF NOT EXISTS FOR (e:Experience) REQUIRE e.name IS UNIQUE")
    except Exception as e:
        print(f"Constraint creation note: {e}")

    # 批量导入
    batch_size = 100
    for i in range(0, len(jobs), batch_size):
        batch = jobs[i:i + batch_size]
        print(f"Importing batch {i // batch_size + 1}/{(len(jobs) - 1) // batch_size + 1} ({i + 1}-{min(i + batch_size, len(jobs))})...")

        for job in batch:
            job_name = job.get('job_title', '').strip()
            company_name = job.get('company_name', '').strip()
            location = job.get('location', '').strip()
            degree = job.get('degree', '').strip()
            experience = job.get('experience', '').strip()
            skill_str = job.get('skill', '')
            salary = job.get('salary_range', '').strip()
            industry = job.get('industry', '').strip()
            description = job.get('description', '')

            if not job_name:
                continue

            # 创建 Job 节点
            try:
                graph.run("""
                    MERGE (j:Job {name: $name})
                    SET j.salary = $salary,
                        j.industry = $industry,
                        j.description = $description
                """, name=job_name, salary=salary, industry=industry, description=description[:500] if description else '')
            except Exception as e:
                print(f"  Error creating Job '{job_name}': {e}")
                continue

            # 创建 Company 节点和关系
            if company_name:
                try:
                    graph.run("""
                        MERGE (c:Company {name: $name})
                        WITH c
                        MATCH (j:Job {name: $job_name})
                        MERGE (j)-[:BELONGS_TO]->(c)
                    """, name=company_name, job_name=job_name)
                except Exception as e:
                    print(f"  Error creating Company '{company_name}': {e}")

                # 创建 City 节点和关系
                if location:
                    try:
                        graph.run("""
                            MERGE (city:City {name: $name})
                            WITH city
                            MATCH (c:Company {name: $company_name})
                            MERGE (c)-[:LOCATED_IN]->(city)
                        """, name=location, company_name=company_name)
                    except Exception as e:
                        print(f"  Error creating City '{location}': {e}")

            # 创建 Skill 节点和关系
            skills = split_skills(skill_str)
            for skill in skills:
                try:
                    graph.run("""
                        MERGE (s:Skill {name: $name})
                        WITH s
                        MATCH (j:Job {name: $job_name})
                        MERGE (j)-[:REQUIRES_SKILL]->(s)
                    """, name=skill, job_name=job_name)
                except Exception as e:
                    pass  # 单个技能失败不影响整体

            # 创建 Degree 节点和关系
            if degree:
                try:
                    graph.run("""
                        MERGE (d:Degree {name: $name})
                        WITH d
                        MATCH (j:Job {name: $job_name})
                        MERGE (j)-[:REQUIRES_DEGREE]->(d)
                    """, name=degree, job_name=job_name)
                except Exception as e:
                    print(f"  Error creating Degree '{degree}': {e}")

            # 创建 Experience 节点和关系
            if experience:
                try:
                    graph.run("""
                        MERGE (e:Experience {name: $name})
                        WITH e
                        MATCH (j:Job {name: $job_name})
                        MERGE (j)-[:REQUIRES_EXPERIENCE]->(e)
                    """, name=experience, job_name=job_name)
                except Exception as e:
                    print(f"  Error creating Experience '{experience}': {e}")

    # 统计导入结果
    print("\n=== Import Statistics ===")
    for label in ['Job', 'Company', 'City', 'Skill', 'Degree', 'Experience']:
        result = graph.run(f"MATCH (n:{label}) RETURN count(n) as count").data()
        print(f"  {label}: {result[0]['count']}")

    cursor.close()
    mysql_conn.close()
    print("\nImport completed!")


if __name__ == "__main__":
    import_to_neo4j()
