"""
1、实现了
    智能聊天系统，用于与用户进行教育相关的对话
    包含对话历史记录管理
    集成了上下文感知能力
2、主要技术：
使用 OpenAI API（通义千问模型）进行对话生成
采用异步流式处理技术（Generator）
使用 typing 模块进行类型提示
实现了上下文管理和历史记录维护
"""

from typing import List, Dict, Optional, Generator
from openai import OpenAI

class ChatSystem:
    # 初始化聊天系统
    def __init__(self):
        self.history: List[Dict] = []
        # 初始化通义千问客户端
        self.client = OpenAI(
            api_key = "sk-178e130a121445659860893fdfae1e7d",
            base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        # 添加系统角色设置
        self.history.append({
            "role": "system",
            "content": "你是一个教育助手。你会为用户提供安全，有帮助，准确的回答。同时，你会拒绝一切涉及恐怖主义，种族歧视，黄色暴力等问题的回答。"
        })

    # 处理用户消息并返回响应（流式）
    def chat(self, message: str, context: Optional[str] = None) -> Generator[str, None, None]:
        try:
            # 准备发送到API的消息
            messages = self._prepare_messages(message, context)
            
            # 创建流式对话
            stream = self.client.chat.completions.create(
                model="qwen-turbo",
                messages=messages,
                temperature=0.3,
                stream=True  # 启用流式输出
            )
            
            # 用于收集完整的回答
            full_response = ""
            
            # 逐个词语返回回答
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    yield content
            
            # 添加助手响应到历史记录
            self.history.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            error_msg = f"对话生成出错: {str(e)}"
            self.history.append({"role": "assistant", "content": error_msg})
            yield error_msg
    
    # 准备发送给API的消息列表
    def _prepare_messages(self, message: str, context: Optional[str]) -> List[Dict]:
        messages = []
        
        # 添加系统消息
        if context:
            messages.append({
                "role": "system",
                "content": f"你是一个教育助手。请基于以下视频内容回答问题：\n\n{context}"
            })
        else:
            # 使用初始化时设置的系统消息
            messages.append(self.history[0])
        
        # 添加历史对话
        messages.extend(self.history[1:])  # 排除系统消息后的所有历史记录
        
        # 添加当前消息
        messages.append({"role": "user", "content": message})
        
        return messages
    
    # 获取对话历史
    def get_history(self) -> List[Dict]:
        return self.history
    
    # 清空对话历史
    def clear_history(self) -> None:
        # 保留系统角色设置
        system_message = self.history[0]
        self.history = [system_message]
