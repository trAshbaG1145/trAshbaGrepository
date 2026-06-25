from kg_config import get_graph

graph = get_graph()


def search_jobs(job_name=None, skills=None, education=None, experience=None):
    print("Input parameters - Job Name:", job_name, "Skills:", skills, "Education:",
          education, "Experience:", experience)

    query = """
    MATCH (j:Job)
    OPTIONAL MATCH (j)-[:BELONGS_TO]->(c:Company)
    OPTIONAL MATCH (c)-[:LOCATED_IN]->(city:City)
    OPTIONAL MATCH (j)-[:REQUIRES_SKILL]->(s:Skill)
    OPTIONAL MATCH (j)-[:REQUIRES_DEGREE]->(d:Degree)
    OPTIONAL MATCH (j)-[:REQUIRES_EXPERIENCE]->(e:Experience)
    WHERE 1=1
    """

    conditions = []
    params = {
        "job_name": job_name,
        "skills": skills if skills else [],
        "education": education,
        "experience": experience,
    }

    if job_name:
        conditions.append("j.name = $job_name")

    if skills:
        skill_conditions = []
        for idx, skill in enumerate(skills):
            param_name = f"skill_{idx}"
            skill_conditions.append(f"s.name = ${param_name}")
            params[param_name] = skill
        conditions.append(f"({' OR '.join(skill_conditions)})")

    if education:
        conditions.append("d.name = $education")

    if experience:
        conditions.append("e.name = $experience")
    else:
        conditions.append("e.name = '经验不限'")

    if conditions:
        query += "\nAND " + "\nAND ".join(conditions)

    query += """
    WITH 
        j, c, city,
        collect(DISTINCT s.name) as skills_collected,
        collect(DISTINCT d.name) as degrees_collected,
        collect(DISTINCT e.name) as experiences_collected
    RETURN 
        j.name as job_name,
        j.salary as salary,
        c.name as company_name,
        city.name as city_name,
        skills_collected as skills,
        degrees_collected as degrees,
        experiences_collected as experiences,
        size([skill IN skills_collected WHERE skill IN $skills]) +
        (CASE WHEN $education IN degrees_collected THEN 1 ELSE 0 END) +
        (CASE WHEN $experience IN experiences_collected THEN 1 ELSE 0 END) AS match_count
    ORDER BY match_count DESC
    LIMIT 10
    """

    print("Generated query:", query)
    print("Query parameters:", params)

    try:
        result = graph.run(query, params)
    except Exception as e:
        print("Query execution failed:", e)
        return [], {"nodes": [], "links": [], "categories": []}

    top_jobs = []
    echart_data = {
        "nodes": [],
        "links": [],
        "categories": [
            {"name": "职位", "itemStyle": {"color": "#4CAF50"}},
            {"name": "技能", "itemStyle": {"color": "#2196F3"}},
            {"name": "学历", "itemStyle": {"color": "#FF9800"}},
            {"name": "经验", "itemStyle": {"color": "#9C27B0"}},
            {"name": "公司", "itemStyle": {"color": "#FF5722"}},
            {"name": "城市", "itemStyle": {"color": "#795548"}}
        ]
    }

    node_ids = set()

    for record in result:
        print("Record:", record)
        job_name = record["job_name"]
        salary = record["salary"]
        company = record.get("company_name", "未知公司")
        city = record.get("city_name", "未知城市")

        # 处理职位节点
        if job_name not in node_ids:
            echart_data["nodes"].append({
                "id": job_name,
                "name": f"{job_name}\n{company}|{city}\n({salary})",
                "category": 0
            })
            node_ids.add(job_name)

        # 处理公司节点和链接
        if company:
            if company not in node_ids:
                echart_data["nodes"].append({
                    "id": company,
                    "name": company,
                    "category": 4
                })
                node_ids.add(company)
            echart_data["links"].append({
                "source": job_name,
                "target": company,
                "label": "所属公司"
            })

            # 处理城市节点和链接
            if city:
                if city not in node_ids:
                    echart_data["nodes"].append({
                        "id": city,
                        "name": city,
                        "category": 5
                    })
                    node_ids.add(city)
                echart_data["links"].append({
                    "source": company,
                    "target": city,
                    "label": "公司所在地"
                })

        # 处理技能、学历和经验的关系
        for skill in record["skills"]:
            if skill not in node_ids:
                echart_data["nodes"].append({
                    "id": skill,
                    "name": skill,
                    "category": 1
                })
                node_ids.add(skill)
            echart_data["links"].append({
                "source": job_name,
                "target": skill,
                "label": "要求技能"
            })

        for degree in record["degrees"]:
            if degree not in node_ids:
                echart_data["nodes"].append({
                    "id": degree,
                    "name": degree,
                    "category": 2
                })
                node_ids.add(degree)
            echart_data["links"].append({
                "source": job_name,
                "target": degree,
                "label": "要求学历"
            })

        for exp in record["experiences"]:
            if exp not in node_ids:
                echart_data["nodes"].append({
                    "id": exp,
                    "name": exp,
                    "category": 3
                })
                node_ids.add(exp)
            echart_data["links"].append({
                "source": job_name,
                "target": exp,
                "label": "需要经验"
            })

        top_jobs.append({
            "职位": job_name,
            "薪资": salary,
            "匹配技能": record["skills"],
            "要求学历": record["degrees"],
            "经验要求": experience,
            "公司": company,
            "城市": city
        })

    return top_jobs, echart_data


if __name__ == "__main__":
    # 示例输入
    job_name = "软件工程师"
    skills = ["Python", "Java"]
    education = "本科"
    experience = "3-5年"

    top_jobs, echart_data = search_jobs(job_name, skills, education, experience)
    print("Top Jobs:", top_jobs)
    print("ECharts Data:", echart_data)
