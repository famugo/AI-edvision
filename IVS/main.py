import warnings
import torchvision
torchvision.disable_beta_transforms_warning()
warnings.filterwarnings('ignore', category=UserWarning)

import streamlit as st
import tempfile
import os
from openai import OpenAI
from Process_video import process_video
from download_video import download_and_play_video
from note_system import NoteSystem, Note, NoteImportance, NoteTemplate
from rag_system import RAGSystem
from learning_path import KnowledgeGraph, LearningPathPlanner, LearningProgressTracker
import logging
from datetime import datetime
import time
import random

# 设置页面配置
st.set_page_config(
    page_title="AI-EdVision  视频分析系统",
    page_icon="🎓",
    layout="wide"
)

# 配置日志
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_file = os.path.join(log_dir, f"streamlit_{datetime.now().strftime('%Y%m%d')}.log")
logging.basicConfig(
    filename=log_file,
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 通义千问 API配置
API_KEY = "sk-178e130a121445659860893fdfae1e7d"  # 建议使用环境变量

# 笔记模板名称映射
NOTE_TEMPLATE_NAMES = {
    "无模板": "无模板",
    NoteTemplate.CONCEPT.value: "概念笔记",
    NoteTemplate.QUESTION.value: "问题笔记",
    NoteTemplate.SUMMARY.value: "总结笔记",
    NoteTemplate.REVIEW.value: "复习笔记"
}

# 通义千问对话API类
class QwenChatAPI:
    def __init__(self):
        self.client = OpenAI(
            api_key=API_KEY,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        # 分别存储两种模式的消息历史
        self.video_qa_messages = [
            {"role": "system", "content": "你是一个教育助手。你会为用户提供安全，有帮助，准确的回答。同时，你会拒绝一切涉及恐怖主义，种族歧视，黄色暴力等问题的回答。"}
        ]
        self.free_chat_messages = [
            {"role": "system", "content": "你是一个教育助手。你会为用户提供安全，有帮助，准确的回答。同时，你会拒绝一切涉及恐怖主义，种族歧视，黄色暴力等问题的回答。"}
        ]
        # 添加请求计数器和最后请求时间
        self.request_count = 0
        self.last_request_time = 0
        self.max_retries = 3
        self.retry_delay = 1.5

    def _wait_for_rate_limit(self):
        """等待以符合速率限制"""
        current_time = time.time()
        time_diff = current_time - self.last_request_time
        
        # 如果距离上次请求不到1秒
        if time_diff < 1:
            # 添加一个随机延迟，避免所有请求同时发送
            sleep_time = 1 - time_diff + random.uniform(0, 0.5)
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()

    def _make_request(self, messages, retry_count=0):
        """发送API请求，包含重试逻辑"""
        try:
            self._wait_for_rate_limit()
            
            stream_response = self.client.chat.completions.create(
                model="qwen-turbo",
                messages=messages,
                temperature=0.3,
                stream=True
            )
            return stream_response
            
        except Exception as e:
            if "rate_limit_reached_error" in str(e) and retry_count < self.max_retries:
                # 如果是速率限制错误且未超过最大重试次数
                retry_sleep = self.retry_delay * (retry_count + 1)
                time.sleep(retry_sleep)
                return self._make_request(messages, retry_count + 1)
            else:
                raise e

    def _judge_question_type(self, question: str) -> bool:
        """判断是否需要分析整个视频内容"""
        judge_prompt = f"""请判断以下问题是否需要分析整个视频内容来回答：

问题：{question}

判断标准：
1. 如果问题涉及视频的整体内容、主题、总结等（如"视频讲了什么"、"视频的主要内容是什么"、"总结一下视频内容"等），返回"需要全文分析"
2. 如果问题是具体的、针对特定内容的提问，返回"使用RAG检索"

请只返回"需要全文分析"或"使用RAG检索"这两个短语之一。"""

        try:
            messages = [{"role": "user", "content": judge_prompt}]
            self._wait_for_rate_limit()  # 添加速率限制等待
            response = self.client.chat.completions.create(
                model="qwen-turbo",
                messages=messages,
                temperature=0.3,
            )
            result = response.choices[0].message.content.strip()
            return result == "需要全文分析"
        except Exception as e:
            logging.error(f"Question type judgment error: {str(e)}")
            return False

    def chat(self, user_input, mode="free_chat", context=None, full_transcript=None, stream=True):
        """流式对话函数"""
        # 选择对应模式的消息历史
        messages = self.video_qa_messages if mode == "video_qa" else self.free_chat_messages
        
        try:
            if mode == "video_qa":
                # 判断是否需要分析整个视频内容
                needs_full_analysis = self._judge_question_type(user_input)
                
                if needs_full_analysis and full_transcript:
                    # 使用完整字幕进行回答
                    prompt = f"""请基于以下完整的视频字幕回答用户的问题。

用户问题：{user_input}

完整视频字幕：
{full_transcript}

请给出全面、详细的回答。回答要求：
1. 分条列点说明
2. 使用markdown格式
3. 突出重点内容"""
                else:
                    # 使用相关字幕片段进行回答
                    prompt = f"""请基于以下视频片段回答用户的问题。

用户问题：{user_input}

{context}

请给出准确、相关的回答。回答要求：
1. 分条列点说明
2. 使用markdown格式
3. 突出重点内容
4. 如果提供的视频片段无法完全回答问题，请说明"""
            else:
                # 自由对话模式
                prompt = user_input

            # 添加用户消息
            messages.append({"role": "user", "content": prompt})
            
            # 创建流式对话（使用重试机制）
            stream_response = self._make_request(messages)
            
            # 用于收集完整的回答
            full_response = ""
            
            # 逐个词语返回回答
            for chunk in stream_response:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    yield content
            
            # 添加助手回答到历史记录
            messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            error_msg = f"对话生成出错: {str(e)}"
            logging.error(f"Chat API Error: {str(e)}")
            yield error_msg

def submit_chat():
    if st.session_state.chat_input.strip():  # 确保输入不是空白
        st.session_state.submit_chat = True

# 处理本地上传的视频
def process_uploaded_video(uploaded_file, whisper_model_size, video_language):
    """处理上传的视频文件"""
    if uploaded_file is not None:
        try:
            # 保存视频数据到session state
            video_data = uploaded_file.getvalue()
            st.session_state.video_data = video_data
            
            # 创建临时文件用于处理
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(video_data)
                video_path = tmp_file.name

            # 处理视频
            with st.spinner('处理视频中...'):
                process_video(
                    video_path,
                    original_filename=uploaded_file.name,
                    whisper_model=whisper_model_size,
                    st_session_state=st.session_state
                )
                
            st.session_state.processed_video = True
            return True

        except Exception as e:
            st.error(f'处理视频时出错: {str(e)}')
            return False
        finally:
            # 保留视频文件，不再删除
            pass
    return False

def process_video_link(video_link, whisper_model_size, video_language):
    """处理视频链接"""
    if video_link:
        try:
            # 创建一个可展开的状态信息区域
            status_expander = st.expander("📝 处理状态信息", expanded=True)
            
            # 下载视频
            with status_expander:
                with st.spinner('下载视频中...'):
                    result = download_and_play_video(video_link)
                    if result is None or result[0] is None:  # 检查返回值
                        status_expander.error('❌ 下载视频失败')
                        return False
                    
                    video_path, video_title = result
                    if not os.path.exists(video_path):  # 确保文件存在
                        status_expander.error(f'❌ 视频文件不存在: {video_path}')
                        return False
                    
                    status_expander.success(f"✅ 视频 '{video_title}' 已下载")
                    status_expander.info(f"📂 保存位置: {video_path}")
                    
                    # 读取视频数据
                    with open(video_path, 'rb') as f:
                        video_data = f.read()
                    st.session_state.video_data = video_data

            # 处理视频
            with status_expander:
                with st.spinner('处理视频中...'):
                    process_video(
                        video_path,
                        whisper_model=whisper_model_size,
                        st_session_state=st.session_state
                    )
                    
            st.session_state.processed_video = True
            status_expander.success("✅ 视频处理完成")
            # 将处理成功的状态信息保存到session state中
            if 'status_expander' not in st.session_state:
                st.session_state.status_expander = status_expander
            return True

        except Exception as e:
            if 'status_expander' in locals():
                status_expander.error(f'❌ 处理视频时出错: {str(e)}')
            else:
                st.error(f'❌ 处理视频时出错: {str(e)}')
            return False
        finally:
            # 保留视频文件，不再删除
            pass
    return False

def handle_video_tab():
    """处理视频上传和显示标签页"""
    st.header("📹 视频处理")
    
    # 初始化session state
    if 'processed_video' not in st.session_state:
        st.session_state.processed_video = False
    if 'video_data' not in st.session_state:
        st.session_state.video_data = None
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = "upload"  # 默认显示上传标签页
        
    # 创建两列布局
    col1, col2 = st.columns(2)
    
    # 左列：语言选择
    with col1:
        video_language = st.selectbox(
            "选择视频语言", 
            ["Other", "English"], 
            index=0,
            help="""
            🌏 选择Other：
            - 适用于包含非英语内容的视频（如中文、日语等）
            - 系统将使用多语言模型进行处理
            
            🎯 选择English：
            - 适用于主要是英语内容的视频
            - 系统将使用英语专用模型，可获得更好的识别效果
            """
        )
    
    # 右列：模型选择
    with col2:
        if video_language == "English":
            whisper_model_size = st.selectbox(
                "选择Whisper模型大小",
                ["tiny.en", "base.en", "small.en", "medium.en"],
                index=1,
                help="""
                🤖 模型大小说明：
                - tiny.en：最快速但准确度较低
                - base.en：平衡速度和准确度（推荐）
                - small.en：准确度较高但速度较慢
                - medium.en：最高准确度但速度最慢
                """
            )
        else:
            whisper_model_size = st.selectbox(
                "选择Whisper模型大小",
                ["tiny", "base", "small", "medium", "large", "turbo"],
                index=5,
                help="""
                🤖 模型大小说明：
                - tiny/base：速度快但准确度较低
                - small/medium：平衡速度和准确度
                - large：最高准确度但速度较慢
                - turbo：OpenAI最新模型，速度快且准确度高（推荐）
                """
            )

    # 创建选项卡
    tab1, tab2 = st.tabs(["📤 本地视频上传", "🔗 视频链接输入"])
    
    # 本地视频上传标签页
    with tab1:
        if not st.session_state.processed_video:
            uploaded_file = st.file_uploader("选择视频文件", type=["mp4", "mov", "avi"])
            if uploaded_file:
                if st.button("处理本地视频"):
                    process_uploaded_video(uploaded_file, whisper_model_size, video_language)
        else:
            # 创建或获取状态信息expander
            if 'status_expander' not in st.session_state:
                status_expander = st.expander("📝 处理状态信息", expanded=True)
                status_expander.info("已有视频正在处理中。如需处理新视频，请刷新页面。")
                st.session_state.status_expander = status_expander
            else:
                st.session_state.status_expander.info("已有视频正在处理中。如需处理新视频，请刷新页面。")

    # 视频链接输入标签页
    with tab2:
        if not st.session_state.processed_video:
            video_link = st.text_input("输入视频链接（支持B站、YouTube等平台）", key="video_link_input")
            # 始终显示按钮，但根据是否有输入来决定是否禁用
            if st.button("处理在线视频", disabled=not bool(video_link), key="process_video_button"):
                process_video_link(video_link, whisper_model_size, video_language)
        else:
            # 创建或获取状态信息expander
            if 'status_expander' not in st.session_state:
                status_expander = st.expander("📝 处理状态信息", expanded=True)
                status_expander.info("已有视频正在处理中。如需处理新视频，请刷新页面。")
                st.session_state.status_expander = status_expander
            else:
                st.session_state.status_expander.info("已有视频正在处理中。如需处理新视频，请刷新页面。")

    # 显示视频
    if st.session_state.video_data is not None:
        with st.expander("📺 播放视频", expanded=True):
            st.video(st.session_state.video_data)
            
    # 添加重置按钮
    if st.session_state.processed_video:
        # 在状态信息expander中显示成功信息
        if 'status_expander' not in st.session_state:
            status_expander = st.expander("📝 处理状态信息", expanded=True)
            status_expander.success("✅ 视频已成功处理")
            st.session_state.status_expander = status_expander
        else:
            st.session_state.status_expander.success("✅ 视频已成功处理")
            
        if st.button("处理新视频"):
            st.session_state.processed_video = False
            st.session_state.video_data = None
            if 'status_expander' in st.session_state:
                del st.session_state.status_expander
            st.rerun()

def merge_subtitle_segments(transcript):
    """将字幕合并成有意义的段落，并添加标点符号"""
    if not transcript:
        return ""
    
    lines = transcript.strip().split('\n\n')
    merged_paragraphs = []
    current_paragraph = []
    current_text = ""
    start_time = ""
    end_time = ""
    
    for line in lines:
        if not line.strip():
            continue
            
        # 提取时间戳和文本
        try:
            # 提取时间戳 [HH:MM:SS.sss --> HH:MM:SS.sss]
            time_parts = line[1:line.find(']')].split(' --> ')
            if not start_time:
                start_time = time_parts[0]
            end_time = time_parts[1]
            
            # 提取文本并添加标点
            text = line[line.find(']') + 1:].strip()
            if text:
                if current_text:
                    # 检查最后一个字符是否已经有标点
                    if not current_text[-1] in '。，！？':
                        current_text += '，'
                current_text += text
                
                # 如果文本较长或以句号结尾，创建新段落
                if len(current_text) > 50 or text[-1] in '。！？':
                    if not current_text[-1] in '。，！？':
                        current_text += '。'
                    merged_paragraphs.append(
                        f"[{start_time} --> {end_time}]\n{current_text}\n\n---\n\n"
                    )
                    current_text = ""
                    start_time = ""
                    end_time = ""
        except Exception as e:
            print(f"处理字幕行时出错: {str(e)}")
            continue
    
    # 处理最后一个段落
    if current_text:
        if not current_text[-1] in '。，！？':
            current_text += '。'
        merged_paragraphs.append(
            f"[{start_time} --> {end_time}]\n{current_text}\n\n---\n\n"
        )
    
    return ''.join(merged_paragraphs)

def parse_timestamp(time_str):
    """将时间字符串转换为秒数"""
    try:
        # 处理 HH:MM:SS.mmm 格式
        if '.' in time_str:
            main_time, ms = time_str.split('.')
            hours, minutes, seconds = map(int, main_time.split(':'))
            return hours * 3600 + minutes * 60 + seconds + float(f"0.{ms}")
        # 处理 HH:MM:SS 格式
        else:
            hours, minutes, seconds = map(int, time_str.split(':'))
            return hours * 3600 + minutes * 60 + seconds
    except Exception as e:
        raise ValueError(f"无法解析时间戳: {time_str}")

def format_timestamp(seconds):
    """将秒数转换为时:分:秒格式"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    else:
        return f"{minutes:02d}:{seconds:02d}"

def handle_subtitle_tab():
    if st.session_state.video_transcript:
        # 创建process_captions目录
        process_captions_dir = os.path.join('../IVS/captions/process_captions')
        os.makedirs(process_captions_dir, exist_ok=True)
        
        # 获取当前视频文件名
        if 'current_video_name' in st.session_state:
            base_filename = os.path.splitext(st.session_state.current_video_name)[0]
            processed_filepath = os.path.join(process_captions_dir, f"{base_filename}_processed.txt")
            
            # 获取合并后的字幕段落
            merged_transcript = merge_subtitle_segments(st.session_state.video_transcript)
            
            # 保存处理后的字幕文件
            try:
                with open(processed_filepath, 'w', encoding='utf-8') as f:
                    f.write(merged_transcript)
                st.success(f"已保存处理后的字幕文件到: {processed_filepath}")
            except Exception as e:
                st.error(f"保存字幕文件失败: {str(e)}")

        # 创建HTML内容
        html_content = """
        <style>
        .subtitle-container {
            height: 400px;
            overflow-y: auto;
            padding: 20px;
            font-family: sans-serif;
            margin-top: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        .subtitle-segment {
            margin-bottom: 20px;
            padding: 10px;
            border-bottom: 1px solid #ddd;
        }
        .selected-timestamp {
            background-color: #f0f2f6;
            border-left: 4px solid #ff4b4b;
            padding: 10px;
            border-radius: 5px;
        }
        </style>
        
        <script>
        function scrollToSegment(id) {
            const element = document.getElementById(id);
            if (element) {
                element.scrollIntoView({behavior: 'smooth', block: 'center'});
            }
        }
        </script>
        
        <div class="subtitle-container" id="subtitle-container">
        """
        
        # 获取合并后的字幕段落
        merged_transcript = merge_subtitle_segments(st.session_state.video_transcript)
        paragraphs = []
        
        for paragraph in merged_transcript.split('---\n\n'):
            if not paragraph.strip():
                continue
            try:
                # 提取时间戳 [HH:MM:SS.sss --> HH:MM:SS.sss]
                time_str = paragraph[paragraph.find('[')+1:paragraph.find(']')]
                text_part = paragraph[paragraph.find(']')+1:].strip()
                paragraphs.append((time_str, text_part))
            except Exception as e:
                continue
                
        # 为每个段落添加HTML内容
        selected_index = st.session_state.get('note_timestamp', None)
        for i, (time_str, text) in enumerate(paragraphs):
            segment_class = "selected-timestamp" if i == selected_index else "subtitle-segment"
            html_content += f"""
            <div id="segment_{i}" class="{segment_class}">
                [{time_str}]<br>{text}
            </div>
            """
        
        html_content += "</div>"
        
        # 添加自动滚动脚本
        if selected_index is not None:
            html_content += f"""
            <script>
                setTimeout(function() {{
                    scrollToSegment('segment_{selected_index}');
                }}, 100);
            </script>
            """
        
        # 使用st.components.html显示内容
        st.components.v1.html(html_content, height=450)

def handle_qa_tab():
    st.markdown("### 💡 智能问答")
    
    # 初始化两种模式的聊天历史
    if 'video_qa_messages' not in st.session_state:
        st.session_state.video_qa_messages = []
    if 'free_chat_messages' not in st.session_state:
        st.session_state.free_chat_messages = []
    
    # 初始化通义千问 API
    if 'qwen_api' not in st.session_state:
        st.session_state.qwen_api = QwenChatAPI()
    
    # 初始化输入框的key
    if 'qa_input_key' not in st.session_state:
        st.session_state.qa_input_key = 0

    # 添加选择器，让用户选择是否基于视频内容进行问答
    use_video_content = st.radio(
        "选择问答模式",
        ["基于视频内容的智能问答", "自由对话模式"],
        index=0,  # 默认选择基于视频内容
        help="基于视频内容：分析视频内容回答问题\n自由对话：可以询问任何问题"
    )

    # 根据当前模式选择对应的消息列表
    current_messages = (st.session_state.video_qa_messages 
                       if use_video_content == "基于视频内容的智能问答" 
                       else st.session_state.free_chat_messages)

    # 添加清除聊天记录按钮
    col1, col2, col3 = st.columns([6, 2, 2])
    with col2:
        if st.button("清除聊天记录", use_container_width=True):
            if use_video_content == "基于视频内容的智能问答":
                st.session_state.video_qa_messages = []
            else:
                st.session_state.free_chat_messages = []
            st.rerun()

    if use_video_content == "基于视频内容的智能问答":
        # 检查是否有视频数据和转录文本
        if not st.session_state.get("video_data") or not st.session_state.get("video_transcript"):
            st.warning("请先上传并处理视频")
            return
    
    # 用户输入区域（固定高度）
    user_input = st.text_area(
        "在这里输入你的问题",
        key=f"qa_input_{st.session_state.qa_input_key}",
        height=100,
        placeholder="请输入你的问题...",
    )

    # 创建两列布局用于按钮
    col1, col2 = st.columns([4, 1])
    with col1:
        if use_video_content == "基于视频内容的智能问答":
            st.markdown("*提示：系统将基于视频内容为您解答问题*")
        else:
            st.markdown("*提示：您可以询问任何问题*")
    with col2:
        send_button = st.button("发送", use_container_width=True)

    # 设置消息显示容器的样式
    st.markdown("""
        <style>
        .stChatMessage {
            padding: 10px;
            margin: 5px 0;
            border-radius: 5px;
            background-color: #f9f9f9;
        }
        .stChatMessage:hover {
            background-color: #f0f0f0;
        }
        .timestamp-link {
            color: #0066cc;
            text-decoration: underline;
            cursor: pointer;
        }
        </style>
    """, unsafe_allow_html=True)

    # 显示当前模式的聊天历史
    if current_messages:
        # 创建一个容器来显示聊天记录
        chat_container = st.container()
        with chat_container:
            for message in current_messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"], unsafe_allow_html=True)

    # 当点击发送按钮且有输入内容时
    if send_button and user_input and user_input.strip():
        current_input = user_input.strip()
        
        # 添加用户消息到当前模式的消息列表
        current_messages.append({"role": "user", "content": current_input})
        with st.chat_message("user"):
            st.markdown(current_input)

        # 根据用户选择决定使用哪种问答模式
        if use_video_content == "基于视频内容的智能问答":
            # 使用RAG系统搜索相关字幕
            similar_subtitles = st.session_state.rag_system.search_similar_subtitles(current_input)
            
            # 构建上下文
            context = "相关视频内容：\n"
            for sub in similar_subtitles:
                context += f"- [{sub['start_time']} --> {sub['end_time']}] {sub['text']} (相关度: {sub['similarity_score']:.2f})\n"
            
            # 获取AI回答（视频问答模式）
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                
                # 使用流式输出
                for token in st.session_state.qwen_api.chat(
                    user_input=current_input,
                    mode="video_qa",
                    context=context,
                    full_transcript=st.session_state.video_transcript
                ):
                    full_response += token
                    message_placeholder.markdown(full_response + "▌")
                
                # 显示最终回答
                message_placeholder.markdown(full_response)
                current_messages.append({"role": "assistant", "content": full_response})
        else:
            # 自由对话模式：直接使用通义千问对话
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                
                # 使用流式输出
                for token in st.session_state.qwen_api.chat(
                    user_input=current_input,
                    mode="free_chat"
                ):
                    full_response += token
                    message_placeholder.markdown(full_response + "▌")
                
                # 显示最终回答
                message_placeholder.markdown(full_response)
                current_messages.append({"role": "assistant", "content": full_response})

        # 通过更新key来清空输入框
        st.session_state.qa_input_key += 1
        st.rerun()
    
    # 笔记系统功能实现
def handle_notes():
    # 初始化笔记系统
    if 'note_system' not in st.session_state:
        st.session_state.note_system = NoteSystem()
        
    if 'note_input' not in st.session_state:
        st.session_state.note_input = ""
        
    if 'note_input_key' not in st.session_state:
        st.session_state.note_input_key = 0

    # 检查是否有视频数据和转录文本
    if not st.session_state.get("video_data") or not st.session_state.get("video_transcript"):
        st.warning("请先上传并处理视频")
        return

    if st.session_state.video_transcript:
        # 获取合并后的字幕段落用于时间点选择
        merged_transcript = merge_subtitle_segments(st.session_state.video_transcript)
        timestamps = []
        
        for paragraph in merged_transcript.split('---\n\n'):
            if not paragraph.strip():
                continue
            try:
                # 提取时间戳 [HH:MM:SS.sss --> HH:MM:SS.sss]
                time_str = paragraph[paragraph.find('[')+1:paragraph.find(']')]
                start_time = time_str.split(' --> ')[0]
                # 保存完整的时间戳和对应的时间信息
                h, m, s = map(float, start_time.split(':'))
                start_seconds = h * 3600 + m * 60 + s
                h, m, s = map(float, time_str.split(' --> ')[1].split(':'))
                end_seconds = h * 3600 + m * 60 + s
                
                timestamps.append((f"[{time_str}]", start_seconds, end_seconds))
            except Exception as e:
                continue

        # 创建两列布局用于笔记模板和时间点列表
        template_col, timestamp_col = st.columns(2)
        
        with template_col:
            # 笔记模板选择
            st.markdown("### 笔记模板")
            selected_template = st.selectbox(
                "选择笔记模板",
                options=["无模板"] + [template.value for template in NoteTemplate],
                format_func=lambda x: NOTE_TEMPLATE_NAMES.get(x, x),
                key="note_template"
            )
            
            # 当模板改变时，更新笔记输入
            if selected_template and selected_template != "无模板":
                template_type = NoteTemplate(selected_template)
                template_content = st.session_state.note_system.get_template(template_type)
                if "last_template" not in st.session_state or st.session_state.last_template != selected_template:
                    st.session_state.note_input = template_content
                    st.session_state.note_input_key += 1  # 强制更新文本区域
                    st.session_state.last_template = selected_template
            elif selected_template == "无模板":
                if "last_template" not in st.session_state or st.session_state.last_template != selected_template:
                    st.session_state.note_input = ""
                    st.session_state.note_input_key += 1
                    st.session_state.last_template = selected_template
        
        with timestamp_col:
            # 添加时间点选择器
            st.markdown("### 时间点列表")
            if timestamps:
                selected_index = st.selectbox(
                    "选择时间点",
                    options=range(len(timestamps)),
                    format_func=lambda i: timestamps[i][0],
                    key="note_timestamp"
                )
                
                if selected_index is not None:
                    selected_time = timestamps[selected_index]
                    st.session_state.current_video_time = selected_time[1]
                    st.session_state.current_video_end_time = selected_time[2]
        
        # 笔记输入区域
        st.markdown("### 添加笔记")
        note_text = st.text_area("笔记内容", 
                               value=st.session_state.note_input,
                               key=f"note_text_{st.session_state.note_input_key}",
                               height=100)
        
        # 创建两列布局用于重要性和标签
        col1, col2 = st.columns(2)
        
        with col1:
            importance = st.selectbox(
                "重要性",
                options=[imp for imp in NoteImportance],
                format_func=lambda x: f"{x.value} {x.name}",
                help="""笔记重要性等级说明：
                LOW - 普通笔记：一般性的知识点或想法
                MEDIUM - 重要笔记：需要重点关注的内容
                HIGH - 非常重要：核心知识点或关键内容
                CRITICAL - 关键笔记：必须掌握的知识点"""
            )
        
        with col2:
            tags = st.text_input("标签（用逗号分隔）", help="例如：概念,重点,待复习")
        
        # 创建两列布局
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("保存笔记"):
                current_time = st.session_state.current_video_time if hasattr(st.session_state, 'current_video_time') else 0
                current_end_time = st.session_state.current_video_end_time if hasattr(st.session_state, 'current_video_end_time') else None
                
                st.session_state.note_system.add_note(
                    text=note_text,
                    timestamp=current_time,
                    end_timestamp=current_end_time,
                    importance=importance,
                    tags=set(tags.split(",")) if tags else set(),
                    template_type=selected_template
                )
                st.success("笔记保存成功！")
                # 通过更新key来清空输入框
                st.session_state.note_input_key += 1
                st.rerun()
        
        with col2:
            if st.button("清空笔记"):
                if st.session_state.note_system.notes:  # 如果有笔记
                    if st.warning("确定要清空所有笔记吗？此操作不可恢复！", icon="⚠️"):
                        st.session_state.note_system.clear_all_notes()
                        st.success("已清空所有笔记！")
                        st.rerun()
                else:
                    st.info("当前没有保存的笔记。")
            
        # 显示笔记列表
        with st.expander("📖 查看笔记", expanded=True):
            # 筛选选项
            col1, col2, col3 = st.columns(3)
            with col1:
                filter_importance = st.selectbox(
                    "按重要性筛选",
                    options=[None] + list(NoteImportance),
                    format_func=lambda x: "全部" if x is None else f"{x.value} {x.name}"
                )
            with col3:
                all_tags = {tag for note in st.session_state.note_system.notes for tag in note.tags}
                filter_tags = st.multiselect("按标签筛选", options=list(all_tags))
                
            notes = st.session_state.note_system.get_notes(
                importance=filter_importance,
                tags=set(filter_tags) if filter_tags else None
            )
            
            if not notes:
                st.info("还没有添加任何笔记")
            else:
                for note in notes:
                    with st.container():
                        col1, col2 = st.columns([4, 1])
                        with col1:
                            st.markdown(f"**时间**: [{note.timestamp_str}] {note.importance.value}")
                            if note.tags:
                                st.markdown(f"**标签**: {', '.join(note.tags)}")
                            st.markdown(f"**内容**:\n{note.text}")
                        with col2:
                            if st.button("删除", key=f"delete_{note.id}"):
                                st.session_state.note_system.delete_note(note.id)
                                st.rerun()
                        st.markdown("---")
                
                # 添加笔记总结功能
                if st.button("生成笔记总结", key="summarize_notes"):
                    with st.spinner("正在生成笔记总结..."):
                        qwen_chat = QwenChatAPI()
                        notes_text = "\n".join([
                            f"时间 {note.timestamp_str} {note.importance.value}: {note.text}" 
                            for note in notes
                        ])
                        
                        # 创建一个markdown容器
                        st.markdown("### 📝 笔记总结")
                        summary_container = st.empty()
                        summary = ""
                        
                        # 处理流式响应
                        for chunk in qwen_chat.chat(f"请总结这些笔记内容：\n{notes_text}", stream=True):
                            if chunk:
                                summary += chunk
                                summary_container.markdown(summary)
            
                # 显示学习进度
                if st.button("查看学习进度"):
                    progress = st.session_state.note_system.get_learning_progress()
                    st.markdown("### 📊 学习进度统计")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("总笔记数", progress["total_notes"])
                    
                    with col2:
                        st.markdown("#### 重要性分布")
                        for imp, ratio in progress["importance_distribution"].items():
                            st.progress(ratio, text=f"{NoteImportance[imp].value} {ratio*100:.1f}%")
                    
                    if progress["tags_distribution"]:
                        st.markdown("#### 标签统计")
                        for tag, ratio in progress["tags_distribution"].items():
                            st.progress(ratio, text=f"{tag}: {ratio*100:.1f}%")

def handle_learning_path_tab():
    """处理学习规划标签页"""
    if not st.session_state.get('processed_video', False):
        st.info("请先上传并处理视频")
        return

    if 'knowledge_graph' not in st.session_state:
        # 获取视频标题
        video_title = None
        if 'current_video_name' in st.session_state:
            # 从文件名中提取标题（去除扩展名）
            video_title = os.path.splitext(st.session_state.current_video_name)[0]
        st.session_state.knowledge_graph = KnowledgeGraph(video_title=video_title)
    if 'path_planner' not in st.session_state:
        st.session_state.path_planner = LearningPathPlanner(st.session_state.knowledge_graph)
    if 'progress_tracker' not in st.session_state:
        st.session_state.progress_tracker = LearningProgressTracker(st.session_state.knowledge_graph)

    # 从视频中提取知识点
    if st.session_state.video_transcript and not st.session_state.knowledge_graph.nodes:
        with st.spinner("正在从视频中提取知识点..."):
            # 处理字幕内容
            transcript_text = st.session_state.video_transcript
            if isinstance(transcript_text, str):
                # 将字幕文本按段落分割
                paragraphs = [p.strip() for p in transcript_text.split('\n\n') if p.strip()]
                
                # 创建进度条
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                success_count = 0
                error_count = 0
                total_paragraphs = len(paragraphs)
                
                # 合并相邻的段落以获得更好的上下文
                merged_paragraphs = []
                current_paragraph = []
                current_start_time = None
                
                for i, paragraph in enumerate(paragraphs):
                    try:
                        status_text.text(f"正在处理段落 {i+1}/{total_paragraphs}")
                        progress_bar.progress((i + 1) / total_paragraphs)
                        
                        # 提取时间戳和内容
                        if '[' not in paragraph or ']' not in paragraph:
                            continue
                            
                        time_str = paragraph[1:paragraph.index(']')]
                        content = paragraph[paragraph.index(']')+1:].strip()
                        
                        if not content:  # 跳过空内容
                            continue
                            
                        # 解析时间戳
                        try:
                            if ' --> ' in time_str:
                                start_time = parse_timestamp(time_str.split(' --> ')[0])
                            else:
                                start_time = parse_timestamp(time_str)
                        except ValueError:
                            start_time = i * 10  # 使用索引作为近似时间戳
                        
                        if not current_paragraph:
                            current_paragraph = [content]
                            current_start_time = start_time
                        else:
                            # 如果时间间隔小于10秒，则合并段落
                            if start_time - current_start_time < 10:
                                current_paragraph.append(content)
                            else:
                                # 保存当前段落并开始新段落
                                merged_content = ' '.join(current_paragraph)
                                if len(merged_content.strip()) > 10:  # 只处理长度超过10个字符的段落
                                    merged_paragraphs.append((current_start_time, merged_content))
                                current_paragraph = [content]
                                current_start_time = start_time
                    except Exception as e:
                        error_count += 1
                        continue
                
                # 添加最后一个段落
                if current_paragraph:
                    merged_content = ' '.join(current_paragraph)
                    if len(merged_content.strip()) > 10:
                        merged_paragraphs.append((current_start_time, merged_content))
                
                # 更新进度条显示
                status_text.text("正在创建知识图谱节点...")
                progress_bar.progress(0)
                
                # 处理合并后的段落
                total_merged = len(merged_paragraphs)
                for i, (start_time, content) in enumerate(merged_paragraphs):
                    try:
                        progress_bar.progress((i + 1) / total_merged)
                        status_text.text(f"正在处理知识点 {i+1}/{total_merged}")
                        
                        node_id = f"node_{i}"
                        # 添加到知识图谱
                        node = st.session_state.knowledge_graph.add_node(
                            node_id,
                            content,
                            "current_video",  # 当前视频
                            start_time
                        )
                        if node:  # 只有当成功创建节点时才增加计数
                            success_count += 1
                        else:
                            error_count += 1
                    except Exception as e:
                        error_count += 1
                        self.logger.error(f"处理节点失败: {str(e)}")
                        continue
                
                # 显示最终统计信息
                stats = st.session_state.knowledge_graph.get_graph_statistics()
                print(f"\n知识图谱处理完成！")
                print(f"✓ 节点：{stats.get('total_nodes', 0)}")
                total_entities = sum(len(entities) for entities in stats.get('entity_types', {}).values())
                print(f"✓ 实体：{total_entities}")
                print(f"✓ 关系：{stats.get('relationship_count', 0)}")
                
                # 显示处理结果
                status_text.text(f"处理完成！成功创建 {success_count} 个知识点，失败 {error_count} 个")
            else:
                st.warning("未能识别字幕格式")
                return

    # 显示知识图谱信息
    st.subheader("📚 知识图谱概览")
    if st.session_state.knowledge_graph.nodes:
        stats = st.session_state.knowledge_graph.get_graph_statistics()
        
        # 显示基本统计信息
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("总节点数", stats.get('total_nodes', 0))
        with col2:
            concept_entities = stats.get('entity_types', {}).get('CONCEPT', [])
            st.metric("概念数量", len(concept_entities) if concept_entities else 0)
        with col3:
            tech_entities = stats.get('entity_types', {}).get('TECH', [])
            st.metric("技术术语数", len(tech_entities) if tech_entities else 0)
        
        # 显示实体类型分布
        st.subheader("实体类型分布")
        total_entities = sum(len(entities) for entities in stats.get('entity_types', {}).values())
        if total_entities > 0:
            for entity_type, entities in stats.get('entity_types', {}).items():
                count = len(entities)
                if count > 0:
                    percentage = count / total_entities
                    st.progress(percentage, text=f"{entity_type}: {count} ({percentage*100:.1f}%)")
        else:
            st.info("暂无实体数据")
        
        # 显示关系类型分布
        st.subheader("关系类型分布")
        total_relations = stats.get('relationship_count', 0)
        if total_relations > 0:
            for rel_type, count in stats.get('relationship_types', {}).items():
                if count > 0:
                    percentage = count / total_relations
                    st.progress(percentage, text=f"{rel_type}: {count} ({percentage*100:.1f}%)")
        else:
            st.info("暂无关系数据")
        
        # 用户输入
        st.subheader("🎯 学习目标设置")
        user_background = st.text_area(
            "请描述您的学习背景和已掌握的知识：",
            help="例如：我已经掌握了Python基础语法"
        )
        target_topics = st.text_area(
            "请输入您想学习的主题（每行一个）：",
            help="例如：\npandas数据处理\n数据可视化"
        )

        if st.button("生成学习路径", type="primary"):
            if user_background and target_topics:
                with st.spinner("正在生成个性化学习路径..."):
                    topics = [t.strip() for t in target_topics.split('\n') if t.strip()]
                    path = st.session_state.path_planner.generate_learning_path(
                        user_background,
                        topics
                    )
                    
                    if path:
                        st.success("学习路径生成成功！")
                        st.subheader("📍 推荐学习路径")
                        
                        # 显示学习路径
                        for i, node_id in enumerate(path, 1):
                            node = st.session_state.knowledge_graph.nodes.get(node_id)
                            if node:
                                with st.expander(f"步骤 {i}: {node.content[:50]}..."):
                                    st.write(f"**完整内容：** {node.content}")
                                    st.write(f"**视频位置：** {format_timestamp(node.timestamp)}")
                                    if node.prerequisites:
                                        st.write("**前置知识：**")
                                        for prereq_id in node.prerequisites:
                                            prereq = st.session_state.knowledge_graph.nodes.get(prereq_id)
                                            if prereq:
                                                st.write(f"- {prereq.content[:100]}...")
                    else:
                        st.warning("未能找到合适的学习路径，请尝试调整学习目标或提供更多背景信息。")
            else:
                st.warning("请填写学习背景和目标主题")
    else:
        st.warning("未能从视频中提取到知识点，请确保视频已正确处理")

# 主函数
def main():
    st.title("智能教育视频分析系统")
    
    # 初始化会话状态变量
    if 'current_video' not in st.session_state:
        st.session_state.current_video = None
    if 'video_transcript' not in st.session_state:
        st.session_state.video_transcript = None
    if 'note_system' not in st.session_state:
        st.session_state.note_system = NoteSystem()
    if 'current_video_time' not in st.session_state:
        st.session_state.current_video_time = 0
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = "视频处理"
    if 'processed_video' not in st.session_state:
        st.session_state.processed_video = False
    if 'learning_plan' not in st.session_state:
        st.session_state.learning_plan = None
    if 'chat_input' not in st.session_state:
        st.session_state.chat_input = ""
    if 'submit_chat' not in st.session_state:
        st.session_state.submit_chat = False
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = RAGSystem()
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'qwen_api' not in st.session_state:
        st.session_state.qwen_api = QwenChatAPI()

    # 创建三列布局
    col1, col2, col3 = st.columns([2, 3, 2])
    
    # 左侧列：视频上传和显示
    with col1:
        handle_video_tab()
    
    # 中间列：功能区
    with col2:
        tab1, tab2, tab3 = st.tabs(["📝 字幕", "💡 智能问答", "📚 学习规划"])
        with tab1:
            handle_subtitle_tab()
        with tab2:
            handle_qa_tab()
        with tab3:
            handle_learning_path_tab()
    
    # 右侧列：笔记系统
    with col3:
        handle_notes()

# 启动应用
if __name__ == "__main__":
    main()