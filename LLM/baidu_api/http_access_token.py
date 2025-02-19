# 使用access_token鉴权，实现了流式对话的功能，用户输入一句话，AI回复一句话，直到用户按Ctrl+C终止。
# 使用 Http 请求实现流式对话
import requests  
import json  
import signal
import sys

API_KEY = "zTHgzj10IC19lwqwvgPVTLvc"  
SECRET_KEY = "SUxBxmpaVlqWS86mm7zreYN5BSGZOcR6"  
 
payload = {  
    "user_id": "python",  
    "messages": [],  
    "system": "这里是AI设定，不用可以删掉",  
    "disable_search": False,  
    "enable_citation": False  
    }
 
def add_message(role, content):  
    message = {  
        "role": role,  
        "content": content  
    }
    payload["messages"].append(message)   

def signal_handler(sig, frame):
    """
    处理Ctrl+C信号的处理函数
    """
    print("\n正在终止程序...")
    sys.exit(0)

def is_valid_input(input_str):
    """
    验证用户输入是否有效
    去除空白字符后检查输入是否为空
    """
    return input_str.strip() != ""

def main():
    # 注册 Ctrl+C 信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    
    url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions?access_token=" + get_access_token()
    headers = {'Content-Type': 'application/json'}
    
    try:
        while True:
            user_input = input("你：")
            if not is_valid_input(user_input):
                print("有什么需要帮忙的吗？")
                continue

            add_message("user", user_input)
            json_payload = json.dumps(payload)

            try:
                response = requests.post(url, headers=headers, data=json_payload)
                response.raise_for_status()  # 检查 HTTP 状态码
                response_text = response.text
                print("API 响应：", response_text)  # 打印原始响应内容
                result = response.json().get("result")
            except requests.exceptions.RequestException as e:
                print("请求失败，错误信息：", e)
                continue

            if result is None:
                result = "未能获取 AI 回复"
            print("AI: " + result)

            add_message("assistant", result)

    except KeyboardInterrupt:
        print("\n正在终止程序...")
        sys.exit(0)

      
def get_access_token():  # 获取token
    """  
    使用 AK，SK 生成鉴权签名（Access Token）  
    :return: access_token，或是None(如果错误)  
    """  
    url = "https://aip.baidubce.com/oauth/2.0/token"  
    params = {"grant_type": "client_credentials", "client_id": API_KEY, "client_secret": SECRET_KEY}  
    return str(requests.post(url, params=params).json().get("access_token"))  
  
if __name__ == '__main__':  
    main()