
# 
 
import os
from openai import OpenAI

def stream_chat():
    client = OpenAI(
        api_key="sk-178e130a121445659860893fdfae1e7d",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    
    # 创建一个对话历史列表
    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'}
    ]
    
    print("欢迎使用通义千问聊天机器人！输入 'quit' 退出对话。")
    
    while True:
        # 获取用户输入
        user_input = input("\n用户: ")
        if user_input.lower() == 'quit':
            print("再见！")
            break
        
        # 将用户输入添加到对话历史
        messages.append({'role': 'user', 'content': user_input})
        
        try:
            # 创建流式响应
            stream = client.chat.completions.create(
                model="qwen-turbo",
                messages=messages,
                stream=True  # 启用流式输出
            )
            
            print("\n助手: ", end="", flush=True)
            assistant_message = ""
            
            # 逐字输出响应
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    print(content, end="", flush=True)
                    assistant_message += content
            
            # 将助手的完整回复添加到对话历史
            messages.append({'role': 'assistant', 'content': assistant_message})
            
        except Exception as e:
            print(f"\n错误信息：{e}")
            print("请参考文档：https://help.aliyun.com/zh/model-studio/developer-reference/error-code")

if __name__ == "__main__":
    stream_chat()
