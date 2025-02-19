"""
1、实现功能：
    支持从多个平台（YouTube、bilibili）下载视频，支持下载进度显示
2、主要技术：
    使用 yt-dlp 库进行视频下载
    集成 Streamlit 进行进度展示
    使用正则表达式处理文件名
    实现了错误处理和重试机制
    支持多平台适配（B站和YouTube的不同配置）
"""

import os
import re
import yt_dlp
import streamlit as st
from pathlib import Path
import time

#清理文件名，移除特殊字符
def sanitize_filename(filename):
    # 移除所有不安全的字符，包括中文
    cleaned = re.sub(r'[^\w\s-]', '_', filename)
    # 将连续的下划线替换为单个下划线
    cleaned = re.sub(r'_+', '_', cleaned)
    # 移除开头和结尾的空格和下划线
    cleaned = cleaned.strip('_ ')
    # 如果文件名为空，使用默认名称
    if not cleaned:
        cleaned = 'video'
    return cleaned

# 生成安全的文件名
def get_safe_filename(original_name):
    name, ext = os.path.splitext(original_name)
    safe_name = sanitize_filename(name)
    # 限制文件名长度
    if len(safe_name) > 50:
        safe_name = safe_name[:50]
    return f"{safe_name}{ext}"

# 显示下载进度的回调函数
def progress_hook(d):
    if d['status'] == 'downloading':
        try:
            progress = float(d.get('_percent_str', '0%').replace('%', '')) / 100
            status_msg = (f"正在下载: {d.get('_percent_str', '0%')} | "
                        f"速度: {d.get('_speed_str', '0 B/s')} | "
                        f"剩余时间: {d.get('_eta_str', '未知')}")
            
            st.progress(progress)
            st.text(status_msg)
        except Exception:
            pass   

# 使用 yt-dlp 下载视频并返回可播放的文件路径(主函数调用)
def download_and_play_video(video_link, download_folder="../IVS/downloads"):
    try:
        # 确保使用正确的路径分隔符并转换为绝对路径
        download_folder = os.path.abspath(download_folder)
        os.makedirs(download_folder, exist_ok=True)
        
        # 通用User-Agent
        user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        
        # 通用的下载配置
        common_opts = {
            'merge_output_format': 'mp4',
            'outtmpl': os.path.join(download_folder, '%(title).50s.%(ext)s'),
            'progress_hooks': [lambda d: progress_hook(d)],
            'quiet': True,
            'no_warnings': True,
            'socket_timeout': 20,
            'retries': 5,
            'fragment_retries': 5,
            'retry_sleep': lambda n: 2 * (n + 1),
        }
        
        # 区分不同平台的下载配置
        if "bilibili.com" in video_link or "b23.tv" in video_link or video_link.startswith('BV'):
            # Bilibili 特定配置
            ydl_opts = {
                **common_opts,
                'format': 'bestvideo+bestaudio/best',  # B站使用最佳质量
                'http_headers': {
                    'Referer': 'https://www.bilibili.com',
                    'User-Agent': user_agent,
                    'Accept': '*/*',
                    'Accept-Language': 'zh-CN,zh;q=0.9',
                    'Accept-Encoding': 'gzip, deflate, br',
                },
                'concurrent_fragment_downloads': 8,
                'buffersize': 4096,
            }
            
            # 处理 Bilibili 短链接
            if "b23.tv" in video_link:
                try:
                    import requests
                    from urllib3.util.retry import Retry
                    from requests.adapters import HTTPAdapter
                    
                    session = requests.Session()
                    retries = Retry(total=5, backoff_factor=1)
                    session.mount('http://', HTTPAdapter(max_retries=retries))
                    session.mount('https://', HTTPAdapter(max_retries=retries))
                    
                    response = session.head(video_link, headers={'User-Agent': user_agent}, 
                                         allow_redirects=True, verify=False)
                    video_link = response.url
                except Exception as e:
                    st.warning(f"处理短链接时出错: {str(e)}")
            
            # 处理 BV 号格式
            if video_link.startswith('BV'):
                video_link = f"https://www.bilibili.com/video/{video_link}"
                
        else:
            # YouTube 等其他平台的配置
            ydl_opts = {
                **common_opts,
                'format': 'best[height<=720][ext=mp4]/best[ext=mp4]/best',  # 限制YouTube视频质量
                'http_headers': {
                    'User-Agent': user_agent,
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                },
                'concurrent_fragment_downloads': 8,
                'buffersize': 4096,
                'ratelimit': 5000000,  # 限制YouTube下载速度
            }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                # 获取视频信息
                info_dict = ydl.extract_info(video_link, download=True)
                if info_dict is None:
                    raise Exception("无法获取视频信息")
                
                # 获取视频标题
                video_title = info_dict.get('title', 'video')
                if not video_title:
                    video_title = 'video'
                
                # 生成安全的文件名
                safe_title = sanitize_filename(video_title)
                expected_file = os.path.join(download_folder, f'{safe_title}.mp4')
                
                # 检查文件是否存在
                if not os.path.exists(expected_file):
                    # 尝试从info_dict获取实际的文件路径
                    if 'requested_downloads' in info_dict:
                        for download in info_dict['requested_downloads']:
                            if 'filepath' in download and os.path.exists(download['filepath']):
                                return download['filepath'], safe_title
                    
                    # 如果还是找不到，查找最新的mp4文件
                    mp4_files = [f for f in os.listdir(download_folder) if f.endswith('.mp4')]
                    if mp4_files:
                        latest_file = max([os.path.join(download_folder, f) for f in mp4_files],
                                        key=os.path.getctime)
                        return latest_file, safe_title
                    
                    raise Exception(f"无法找到下载的视频文件")
                
                return expected_file, safe_title
                
            except Exception as e:
                raise Exception(f"下载失败: {str(e)}")
                
    except Exception as e:
        st.error(f"下载视频时出错: {str(e)}")
        return None, None
