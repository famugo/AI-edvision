"""
1、实现功能：
    视频处理的主要流程控制
    视频音频提取和转录
    字幕生成和格式化
    支持简繁体中文转换
    精确检测视频第一个有声音的时间点
2、主要技术：
    使用 Whisper 进行语音识别
    使用 FFmpeg 进行音视频处理
    使用 OpenCC 进行中文简繁转换
    支持 SRT 和 TXT 格式的字幕生成
    实现了精确的音频检测算法
"""

import streamlit as st
from video_tools import VideoTools
import subprocess
from opencc import OpenCC
import os
import json
from datetime import datetime

# 创建带时间戳的字幕段落
def create_paragraphs_combined(transcript, first_voice_time: float, to_simplified=True):
    try:
        if to_simplified:
            cc = OpenCC('t2s')  # 繁体转简体
        else:
            cc = OpenCC('s2t')  # 简体转繁体
    except Exception as e:
        print(f"简繁转换初始化失败: {e}")
        cc = None
    
    formatted_lines = []
    
    if not transcript or 'segments' not in transcript or not transcript['segments']:
        print("转录结果为空或格式不正确")
        return ""
    
    # 设置时间偏移
    time_offset = first_voice_time
    
    for segment in transcript['segments']:
        start_time = segment['start'] + time_offset
        end_time = segment['end'] + time_offset
        
        # 格式化时间戳
        start_formatted = f"{int(start_time // 3600):02d}:{int((start_time % 3600) // 60):02d}:{start_time % 60:06.3f}"
        end_formatted = f"{int(end_time // 3600):02d}:{int((end_time % 3600) // 60):02d}:{end_time % 60:06.3f}"
        
        # 处理文本
        text = segment['text'].strip()
        if cc:
            try:
                text = cc.convert(text)
            except Exception as e:
                print(f"转换文本失败: {e}")
        
        # 使用方括号包裹时间戳，并添加双换行符
        formatted_line = f"[{start_formatted} --> {end_formatted}] {text}\n\n"
        formatted_lines.append(formatted_line)
    
    return ''.join(formatted_lines)  

# 格式化时间戳
def format_timestamp(seconds):
    """格式化时间戳"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"

# 保存SRT格式字幕文件
def save_as_srt(segments, output_path, first_voice_time: float, to_simplified=True):
    """将转录结果保存为SRT格式"""
    try:
        if to_simplified:
            cc = OpenCC('t2s')  # 繁体转简体
        else:
            cc = OpenCC('s2t')  # 简体转繁体
    except Exception as e:
        print(f"简繁转换初始化失败: {e}")
        cc = None

    with open(output_path, 'w', encoding='utf-8') as f:
        for i, segment in enumerate(segments, 1):
            start_time = segment['start'] + first_voice_time
            end_time = segment['end'] + first_voice_time
            
            # 格式化时间为SRT格式 (HH:MM:SS,mmm)
            start_formatted = f"{int(start_time // 3600):02d}:{int((start_time % 3600) // 60):02d}:{int(start_time % 60):02d},{int((start_time % 1) * 1000):03d}"
            end_formatted = f"{int(end_time // 3600):02d}:{int((end_time % 3600) // 60):02d}:{int(end_time % 60):02d},{int((end_time % 1) * 1000):03d}"
            
            # 处理文本
            text = segment['text'].strip()
            if cc:
                try:
                    text = cc.convert(text)
                except Exception as e:
                    print(f"转换文本失败: {e}")
            
            # 写入SRT格式
            f.write(f"{i}\n")
            f.write(f"{start_formatted} --> {end_formatted}\n")
            f.write(f"{text}\n\n")

# 精确检测视频中第一个有声音的时间点
def detect_first_voice_time(video_path: str) -> float:
    try:
        # 首先提取音频
        temp_audio = f"{os.path.splitext(video_path)[0]}_temp_audio.wav"
        audio_cmd = [
            'ffmpeg', '-i', video_path,
            '-vn',  # 不处理视频
            '-acodec', 'pcm_s16le',  # 强制使用PCM 16-bit编码
            '-ar', '16000',  # 采样率16kHz
            '-ac', '1',  # 单声道
            '-y',  # 覆盖已存在的文件
            temp_audio
        ]
        
        # 使用特定的编码运行命令
        process = subprocess.Popen(
            audio_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding='utf-8',
            errors='ignore'
        )
        _, stderr = process.communicate()
        
        if not os.path.exists(temp_audio) or os.path.getsize(temp_audio) == 0:
            print(f"Failed to extract audio for silence detection: {stderr}")
            return 0
            
        # 然后检测静音
        silence_cmd = [
            'ffmpeg', '-i', temp_audio,
            '-af', 'silencedetect=n=-35dB:d=0.05',
            '-f', 'null', '-'
        ]
        
        # 同样使用特定的编码运行命令
        process = subprocess.Popen(
            silence_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding='utf-8',
            errors='ignore'
        )
        _, stderr = process.communicate()
        
        # 清理临时音频文件
        try:
            os.remove(temp_audio)
        except:
            pass
        
        # 解析输出找到第一个非静音段
        silence_end_times = []
        for line in stderr.split('\n'):
            if 'silence_end' in line:
                try:
                    time_str = line.split('silence_end: ')[1].split(' ')[0]
                    silence_end_times.append(float(time_str))
                except:
                    continue
        
        # 返回第一个非静音时间点
        first_voice_time = min(silence_end_times) if silence_end_times else 0
        return max(0, first_voice_time - 0.1)  # 稍微提前一点，确保不会丢失开头
    except Exception as e:
        print(f"Error detecting first voice time: {e}")
        return 0

# 处理视频主函数
def process_video(video_path, status_container=None, original_filename=None, output_dir='../IVS/captions', whisper_model="base", to_simplified=True, st_session_state=None):
    try:
        if original_filename is None:
            original_filename = os.path.basename(video_path)
            
        # 保存当前处理的视频文件名到session_state
        if st_session_state is not None:
            st_session_state.current_video_name = original_filename
        
        # 修正输出目录的路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.normpath(os.path.join(current_dir, output_dir))
        
        os.makedirs(output_dir, exist_ok=True)
        
        base_filename = os.path.splitext(original_filename)[0]
        srt_filepath = os.path.join(output_dir, f"{base_filename}.srt")
        txt_filepath = os.path.join(output_dir, f"{base_filename}.txt")
        json_filepath = os.path.join(output_dir, f"{base_filename}_metadata.json")
        
        # 检查视频是否包含音频流
        probe_cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_streams',
            '-i',
            video_path
        ]
        process = subprocess.Popen(
            probe_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding='utf-8',
            errors='ignore'
        )
        stdout, _ = process.communicate()
        
        has_audio = False
        try:
            video_info = json.loads(stdout)
            for stream in video_info.get('streams', []):
                if stream.get('codec_type') == 'audio':
                    has_audio = True
                    break
        except:
            pass
            
        if not has_audio:
            if status_container:
                status_container.warning("视频没有音频流，将跳过转录步骤")
            return None, None, None
        
        # 处理有音频的视频
        video_tools = VideoTools(whisper_model=whisper_model)
        
        if status_container:
            status_container.info("正在检测视频中第一个有声音的时间点...")
        first_voice_time = detect_first_voice_time(video_path)
        
        if status_container:
            status_container.info("正在转录视频...")
        
        result = video_tools.transcribe_video(video_path)
        
        if not result:
            if status_container:
                status_container.error("转录失败")
            return None, None, None
            
        # 创建带时间戳的段落
        paragraphs = create_paragraphs_combined(result, first_voice_time, to_simplified)
        
        # 保存为文本文件
        try:
            with open(txt_filepath, 'w', encoding='utf-8') as f:
                f.write(paragraphs)
        except Exception as e:
            if status_container:
                status_container.error(f"保存文本文件失败: {str(e)}")
        
        # 保存SRT格式字幕文件
        try:
            save_as_srt(result['segments'], srt_filepath, first_voice_time, to_simplified)
        except Exception as e:
            if status_container:
                status_container.error(f"保存SRT文件失败: {str(e)}")

        # 保存元数据
        try:
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'filename': original_filename,
                'duration': result.get('duration', 0),
                'language': result.get('language', 'unknown'),
                'first_voice_time': first_voice_time
            }
            with open(json_filepath, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
        except Exception as e:
            if status_container:
                status_container.error(f"保存元数据失败: {str(e)}")
        
        # 将字幕添加到RAG系统
        if st_session_state and 'rag_system' in st_session_state:
            # 使用TXT文件更新RAG系统
            st_session_state.rag_system.add_subtitles_from_txt(base_filename, txt_filepath)
        
        # 更新会话状态中的字幕内容
        import streamlit as st
        st.session_state.video_transcript = paragraphs
        
        if status_container:
            status_container.success("视频处理完成！")
            
        return txt_filepath, srt_filepath, json_filepath
        
    except Exception as e:
        if status_container:
            status_container.error(f"处理视频时出错: {str(e)}")
        import traceback
        print(f"Error details: {traceback.format_exc()}")
        return None, None, None