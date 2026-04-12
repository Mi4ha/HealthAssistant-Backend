from dashscope import MultiModalConversation
import dashscope 

prompt = """
请仔细观察图片并推算食物份量。
请严格按照以下JSON格式输出，不要包含任何Markdown标记（如```json），不要输出任何其他解释性文字！
{
"food_names": ["鸡肉", "面粉"],
"weight_g": 600,
"calories": 1900
}
"""
messages = [
    {'role': 'system', 'content': 'You are a senior nutritionist.'},
    {
        'role': 'user',
        "content": [
            {"image": "file://C:/Users/Mitsuha/Desktop/毕设/chicken.jpg"},
            {"text": f"{prompt}"}
        ]
    }
]
response = MultiModalConversation.call(
    # 若没有配置环境变量，请用阿里云百炼API Key将下行替换为：api_key = "sk-xxx",
    api_key="sk-281cb6a56a954bfea647a2d1e6e0ee49", 
    model="qvq-max",   # 模型列表：https://help.aliyun.com/model-studio/getting-started/models
    messages=messages,
    stream=True,
)

# 定义完整思考过程
reasoning_content = ""
# 定义完整回复
answer_content = ""
# 判断是否结束思考过程并开始回复
is_answering = False

print("=" * 20 + "思考过程" + "=" * 20)

for chunk in response:
    # 如果思考过程与回复皆为空，则忽略
    message = chunk.output.choices[0].message
    reasoning_content_chunk = message.get("reasoning_content", None)
    if (chunk.output.choices[0].message.content == [] and
        reasoning_content_chunk == ""):
        pass
    else:
        # 如果当前为思考过程
        if reasoning_content_chunk != None and chunk.output.choices[0].message.content == []:
            print(chunk.output.choices[0].message.reasoning_content, end="")
            reasoning_content += chunk.output.choices[0].message.reasoning_content
        # 如果当前为回复
        elif chunk.output.choices[0].message.content != []:
            if not is_answering:
                print("\n" + "=" * 20 + "完整回复" + "=" * 20)
                is_answering = True
            print(chunk.output.choices[0].message.content[0]["text"], end="")
            answer_content += chunk.output.choices[0].message.content[0]["text"]

# 如果您需要打印完整思考过程与完整回复，请将以下代码解除注释后运行
# print("=" * 20 + "完整思考过程" + "=" * 20 + "\n")
# print(f"{reasoning_content}")
# print("=" * 20 + "完整回复" + "=" * 20 + "\n")
# print(f"{answer_content}")