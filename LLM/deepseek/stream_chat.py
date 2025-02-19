from openai import OpenAI
import os

class DeepSeekChat:
    def __init__(self, api_key):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
        
    def chat_stream(self, user_input, system_prompt="You are a helpful assistant"):
        try:
            # 创建流式对话
            stream = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input}
                ],
                stream=True  # 启用流式输出
            )
            
            # 逐步获取并打印响应
            print("\nDeepSeek Assistant: ", end="", flush=True)
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    print(chunk.choices[0].delta.content, end="", flush=True)
            print("\n")
            
        except Exception as e:
            print(f"Error occurred: {str(e)}")

def main():
    # 从环境变量获取API密钥，或者直接使用你的API密钥
    api_key = "sk-ba656c564e2148009618ad3a2231c002"  # 建议使用环境变量存储
    
    # 创建聊天实例
    chat = DeepSeekChat(api_key)
    
    print("欢迎使用DeepSeek聊天助手！(输入'quit'退出)")
    
    while True:
        user_input = input("\n你: ")
        if user_input.lower() == 'quit':
            print("再见！")
            break
        
        chat.chat_stream(user_input)

if __name__ == "__main__":
    main()
