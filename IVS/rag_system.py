"""
1、实现功能
    实现检索增强生成（RAG）系统
    支持基于字幕的相似内容检索
    高效的向量检索和相似度计算
    支持 GPU 加速
2、主要技术
    使用 FAISS 进行高效向量检索
    使用 SentenceTransformer 进行文本向量化
    实现了余弦相似度计算
    使用正则表达式处理时间戳
    支持批量处理和增量更新
"""
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
import os
import torch
import re

class RAGSystem:
    def __init__(self):
        # 初始化向量模型
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        # 存储字幕文本和时间戳
        self.subtitles: Dict = {}
        # 存储向量
        self.vectors = None
        # 初始化FAISS索引 - 使用余弦相似度
        self.index = None
        # 存储文本
        self.texts = []
        # 存储时间戳信息
        self.timestamps = []
        # 初始化 FAISS 资源
        self.res = faiss.StandardGpuResources()
        # 设置临时内存限制为 256MB
        self.res.setTempMemory(256 * 1024 * 1024)
        
    def _create_gpu_index(self, dimension):
        """创建 FAISS GPU 索引"""
        try:
            # 先清理之前的索引
            if self.index is not None:
                del self.index
                self.index = None
            
            # 创建 CPU 索引
            cpu_index = faiss.IndexFlatL2(dimension)
            
            try:
                # 尝试创建 GPU 索引
                gpu_index = faiss.index_cpu_to_gpu(self.res, 0, cpu_index)
                return gpu_index
            except RuntimeError as e:
                print(f"GPU 索引创建失败，回退到 CPU 索引: {str(e)}")
                return cpu_index
                
        except Exception as e:
            print(f"索引创建失败: {str(e)}")
            return None

    def _parse_timestamp(self, timestamp_str: str) -> Tuple[str, str]:
        """解析时间戳字符串，返回开始和结束时间"""
        match = re.match(r'\[(.*?)\s*-->\s*(.*?)\]', timestamp_str)
        if match:
            return match.group(1).strip(), match.group(2).strip()
        return None, None
        
    def add_subtitles_from_txt(self, video_name: str, txt_file_path: str):
        """从TXT文件添加字幕到RAG系统"""
        try:
            # 读取字幕文件
            with open(txt_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 存储字幕内容
            self.subtitles[video_name] = content
            
            # 将内容分割成段落
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
            
            # 提取文本内容和时间戳
            texts = []
            timestamps = []
            for p in paragraphs:
                try:
                    timestamp_end = p.find(']')
                    if timestamp_end == -1:
                        continue
                    
                    timestamp_str = p[:timestamp_end+1]
                    text = p[timestamp_end+1:].strip()
                    
                    start_time, end_time = self._parse_timestamp(timestamp_str)
                    if start_time and end_time:
                        texts.append(text)
                        timestamps.append({
                            "start_time": start_time,
                            "end_time": end_time
                        })
                except:
                    continue
            
            # 更新文本和时间戳列表
            start_idx = len(self.texts)
            self.texts.extend(texts)
            self.timestamps.extend(timestamps)
            
            # 计算文本向量
            vectors = self.model.encode(texts, convert_to_tensor=True)
            vectors = vectors.cpu().numpy()
            
            # 更新或初始化向量存储
            if self.vectors is None:
                self.vectors = vectors
            else:
                self.vectors = np.vstack([self.vectors, vectors])
            
            # 创建或更新索引
            if self.index is None:
                self.index = self._create_gpu_index(vectors.shape[1])
            
            if self.index is not None:
                self.index.add(vectors)
                print(f"成功添加 {len(texts)} 个段落到索引")
            else:
                print("索引创建失败，无法添加向量")
                
        except Exception as e:
            print(f"添加字幕失败: {str(e)}")
            import traceback
            print(f"错误详情: {traceback.format_exc()}")
    
    def search_similar_subtitles(self, query: str, top_k: int = 3) -> List[Dict]:
        """搜索与问题最相关的字幕"""
        if not self.subtitles or self.index is None:
            return []
            
        # 将查询转换为向量并归一化
        with torch.device('cuda' if torch.cuda.is_available() else 'cpu'):
            query_vector = self.model.encode([query], convert_to_tensor=True)
            query_vector = torch.nn.functional.normalize(query_vector, p=2, dim=1)  # L2归一化
            query_vector = query_vector.cpu().numpy()
        
        # 搜索最相似的字幕
        similarities, indices = self.index.search(query_vector.astype(np.float32), top_k)
        
        # 获取相关字幕和分数
        similar_subtitles = []
        for i, idx in enumerate(indices[0]):
            if 0 <= idx < len(self.texts):
                subtitle = {
                    "text": self.texts[idx],
                    "similarity_score": float((similarities[0][i] + 1) / 2),
                    "start_time": self.timestamps[idx]["start_time"],
                    "end_time": self.timestamps[idx]["end_time"]
                }
                similar_subtitles.append(subtitle)
        
        # 按相似度分数排序
        similar_subtitles.sort(key=lambda x: x['similarity_score'], reverse=True)
        return similar_subtitles
    
    def load_subtitles(self, video_id: str) -> bool:
        """加载指定视频的TXT字幕"""
        txt_file = os.path.join("captions", f"{video_id}.txt")
        if not os.path.exists(txt_file):
            return False
            
        return self.add_subtitles_from_txt(video_id, txt_file)
        
    def _ensure_captions_dir(self):
        """确保字幕目录存在"""
        if not os.path.exists("captions"):
            os.makedirs("captions")
