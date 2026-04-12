from dashscope import MultiModalConversation

# 导入你刚刚写好的 RAG 检索函数 (假设你的上一个文件叫 test_rag.py)
from test_rag import build_and_search_rag

# 你的 API KEY
MY_ALIYUN_API_KEY = "sk-281cb6a56a954bfea647a2d1e6e0ee49" 

def generate_health_report():
    print("\n" + "="*50)
    print("🔄 开始执行核心工作流：多源数据融合与生成")
    print("="*50)

    # --------------------------------------------------
    # 第一步：获取用户的个人数据 (未来这些数据从 SQLite 数据库查)
    # --------------------------------------------------
    user_profile = {
        "height": 175,
        "weight": 80,
        "daily_steps": 8000,
        "step_calories": 300
    }

    # --------------------------------------------------
    # 第二步：获取多模态视觉模型的识别结果 (未来这里接收 APP 传来的图片)
    # --------------------------------------------------
    food_name = "炸鸡"
    food_calories = 1900

    # --------------------------------------------------
    # 第三步：触发 RAG 检索 (去翻本地指南)
    # --------------------------------------------------
    print("1️⃣ 正在呼叫 RAG 查阅医学指南...")
    # 我们用提取出来的食物名字作为检索词，更精准
    search_query = f"{food_name}吃多了，高脂高热量饮食后如何补救？"
    medical_context = build_and_search_rag() 
    # (注意：为了不重复建库，你可以在 test_rag.py 里把建库的代码注释掉，直接调取本地库，或者由它重新建一次也很快)

    # --------------------------------------------------
    # 第四步：数据大融合（组装终极 Prompt）
    # --------------------------------------------------
    print("\n2️⃣ 正在进行数据融合，组装超级 Prompt...")
    final_prompt = f"""
    【系统身份】：你是一位严厉且专业的私人营养师。
    
    【用户身体数据】：身高{user_profile['height']}cm，体重{user_profile['weight']}kg。
    【今日运动数据】：今日已步行{user_profile['daily_steps']}步，约消耗{user_profile['step_calories']}大卡。
    【今日超标饮食】：用户刚才吃了一顿{food_name}，摄入了高达{food_calories}大卡的热量。
    
    【权威医学参考资料】：
    {medical_context}
    
    【你的任务】：
    1. 评估用户今天的热量盈亏情况（结合他的运动消耗）。
    2. **严格依据上述提供的[权威医学参考资料]**，为用户制定明天的饮食和运动补救计划。不要编造资料中没有的医学原理。
    3. 语气要像专业的私人教练，适当严厉，分点给出建议。
    """

    # --------------------------------------------------
    # 第五步：呼叫千问 Plus 生成最终报告
    # --------------------------------------------------
    print("3️⃣ 正在呼叫千问 Plus (qwen-plus) 生成个性化报告...\n")
    response = MultiModalConversation.call(
        api_key=MY_ALIYUN_API_KEY,
        model='qwen3.5-plus',
        messages=[
            {'role': 'system', 'content': 'You are a senior nutritionist.'},
            {'role': 'user', 'content': final_prompt}
        ],
    )

    # 打印最终结果
    if response.status_code == 200:
        final_answer = response.output.choices[0].message.content[0]['text']
        return final_answer
    else:
        return f"❌ 调用大模型失败：{response.code} - {response.message}"

if __name__ == "__main__":
    generate_health_report()
    print(generate_health_report())