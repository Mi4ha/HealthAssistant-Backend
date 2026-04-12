import dashscope
from dashscope import MultiModalConversation

# 各地域配置不同，请根据实际地域修改
# dashscope.base_http_api_url = "https://dashscope-intl.aliyuncs.com/api/v1"

# 假设这些数据是前端传过来的，或者是数据库里查出来的
user_height = 175
user_weight = 80
today_steps = 8000
step_calories = 300 # 假设算出来的消耗
food_name = "炸鸡"   # 这是 QVQ 刚才提取出来的
food_calories = 1900 # 这是 QVQ 刚才提取出来的

# 使用 f-string 动态生成 Prompt
prompt = f"""
用户基本信息：身高{user_height}cm，体重{user_weight}kg。
今日运动数据：已步行{today_steps}步（消耗约{step_calories}大卡）。
刚才饮食识别结果：摄入了 {food_name}，约 {food_calories}大卡。

请结合营养学知识，指出他今日的热量盈亏情况，并给出明天严厉的运动与饮食补救建议。
"""

messages = [
    {'role': 'system', 'content': 'You are a senior nutritionist.'},
    {
        "role": "user",
        "content": [
            {"text": f"{prompt}"},
        ],
    }
]
response = MultiModalConversation.call(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    # 各地域的API Key不同。获取API Key：https://help.aliyun.com/zh/model-studio/get-api-key
    api_key="sk-281cb6a56a954bfea647a2d1e6e0ee49", 
    model='qwen3.5-plus',   # 可按需更换为其它多模态模型，并修改相应的 messages
    messages=messages)
print(response.output.choices[0].message.content[0]['text'])