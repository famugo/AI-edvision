# 使用access_token鉴权，实现了流式对话的功能
# 使用SDK实现流式对话

import os
import qianfan
import signal
import sys

# 替换为你的 API_Key 和 Secret_Key
os.environ["QIANFAN_AK"] = "zTHgzj10IC19lwqwvgPVTLvc"
os.environ["QIANFAN_SK"] = "SUxBxmpaVlqWS86mm7zreYN5BSGZOcR6"

# 初始化 ChatCompletion
chat_comp = qianfan.ChatCompletion()

# 初始化对话上下文
messages = [{"role": "assistant", "content": "欢迎使用AI助手，请提问！"}]  # 默认首条消息

# 信号处理器：支持按 Ctrl+C 结束程序
def signal_handler(sig, frame):
    print("\n正在结束对话...")
    sys.exit(0)

# 验证输入内容是否有效
def is_valid_input(user_input):
    return user_input.strip() != ""

# 主程序：实现流式对话
def main():
    signal.signal(signal.SIGINT, signal_handler)  # 捕获 Ctrl+C 信号
    print("开始对话，输入你的问题（按 Ctrl+C 退出）：")
    
    while True:
        try:
            # 获取用户输入
            user_input = input("你：")
            if not is_valid_input(user_input):
                print("输入不能为空，请重新输入。")
                continue

            # 动态处理首次消息
            if len(messages) == 1 and messages[0]["role"] == "assistant":
                messages.append({"role": "user", "content": user_input})  # 首次用户消息
            else:
                messages.append({"role": "user", "content": user_input})  # 普通用户消息
            
            # 调用 API
            response = chat_comp.do(model="ERNIE-Tiny-8K", messages=messages)

            # 检查返回内容
            if "body" not in response or not isinstance(response["body"], dict):
                print("AI 回复格式错误或为空，请检查响应内容。")
                continue

            # 提取 AI 回复内容
            ai_reply = response["body"].get("result", "未能获取 AI 回复")
            print("AI：" + ai_reply)

            # 保存 AI 回复到对话上下文
            messages.append({"role": "assistant", "content": ai_reply})

        except Exception as e:
            print("发生错误：", str(e))
            continue

# 启动程序
if __name__ == "__main__":
    main()
