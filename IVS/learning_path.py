"""
知识图谱和学习路径规划模块
1、实现功能：
    构建知识图谱
    生成个性化学习路径
    追踪学习进度
2、主要技术：
    使用知识图谱进行学习路径规划
    支持学习进度追踪"""

from typing import List, Dict, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import json
import os
from datetime import datetime
from py2neo import Graph, Node, Relationship
import spacy
from collections import defaultdict
import re
import sys
import logging
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class KnowledgeNode:
    """知识点节点"""
    id: str
    content: str
    source_video: str
    timestamp: float
    entity_type: str = None
    difficulty: float = 0.5
    vector: Optional[np.ndarray] = None
    prerequisites: List[str] = field(default_factory=list)
    related_concepts: List[str] = field(default_factory=list)
    properties: Dict[str, str] = field(default_factory=dict)
    relationships: Dict[str, List[str]] = field(default_factory=dict)

class KnowledgeGraph:
    """知识图谱"""
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="cjx20040328", video_title=None):
        # 首先设置日志记录器
        self.setup_logger(video_title)
        
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.nodes = {}  # 存储所有节点
        self.graph = nx.DiGraph()  # 有向图
        self.node_embeddings = np.array([])
        self.node_ids = []
        self.index = None
        self.total_nodes = 0
        self.total_entities = 0
        self.total_relations = 0
        self.expected_total = 110  # 预期总节点数
        
        # 初始化NLP模型
        try:
            self.nlp = spacy.load("zh_core_web_sm")
            self.logger.info("成功加载中文NLP模型")
        except OSError:
            self.logger.info("尝试下载中文NLP模型...")
            try:
                import subprocess
                subprocess.run([sys.executable, "-m", "spacy", "download", "zh_core_web_sm"], 
                             check=True)
                self.nlp = spacy.load("zh_core_web_sm")
                self.logger.info("成功下载并加载中文NLP模型")
            except Exception as e:
                self.logger.error(f"无法加载中文NLP模型: {str(e)}")
                self.logger.info("使用基础分词方式")
                self.nlp = None
        
        # 连接Neo4j数据库
        self.neo4j_graph = None
        self.max_retries = 3
        self.retry_delay = 2  # 重试延迟（秒）
        self.connection_failed = False  # 添加标志来追踪连接状态
        
        for attempt in range(self.max_retries):
            try:
                # 使用py2neo连接Neo4j
                self.neo4j_graph = Graph(uri, auth=(user, password))
                self.logger.info("Neo4j数据库连接成功！")
                
                # 创建唯一性约束
                try:
                    # 为content属性创建唯一性约束
                    self.neo4j_graph.run("CREATE CONSTRAINT unique_content IF NOT EXISTS FOR (n:Knowledge) REQUIRE n.content IS UNIQUE")
                    self.logger.debug("创建Neo4j唯一性约束成功")
                    break  # 如果成功连接并创建约束，跳出重试循环
                except Exception as e:
                    self.logger.error(f"创建Neo4j约束失败: {str(e)}")
            except Exception as e:
                self.logger.error(f"Neo4j连接尝试 {attempt + 1}/{self.max_retries} 失败: {str(e)}")
                if attempt == self.max_retries - 1:  # 如果是最后一次尝试
                    self.connection_failed = True  # 标记连接失败
                    self.logger.error("Neo4j连接失败，已达到最大重试次数")
                else:
                    import time
                    time.sleep(self.retry_delay)  # 等待一段时间后重试
                    
    def setup_logger(self, video_title=None):
        """设置日志记录器"""
        # 创建logs目录(如果不存在)
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        # 生成日志文件名
        if video_title:
            # 使用视频标题作为日志文件名
            log_file = os.path.join(log_dir, f"{video_title}.log")
        else:
            # 如果没有视频标题，使用默认的时间戳命名
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(log_dir, f"knowledge_graph_{current_time}.log")
        
        # 配置日志记录器
        self.logger = logging.getLogger("KnowledgeGraph")
        self.logger.setLevel(logging.DEBUG)
        
        # 文件处理器 - 记录所有日志
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # 控制台处理器 - 显示重要信息
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 设置日志格式
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_formatter = logging.Formatter('%(message)s')  # 简化控制台输出格式
        
        file_handler.setFormatter(file_formatter)
        console_handler.setFormatter(console_formatter)
        
        # 清除之前的处理器
        self.logger.handlers.clear()
        
        # 添加处理器
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # 初始化计数器
        self.progress_shown = False
        
        # 显示初始化信息
        print(f"\n知识图谱处理中...")
        print(f"详细日志已保存至：{log_file}")
        
    def extract_entities(self, text: str) -> List[Tuple[str, str, int, int]]:
        """使用NER提取实体"""
        entities = []
        self.logger.debug(f"\n正在从文本中提取实体，文本长度：{len(text)}")
        
        if self.nlp:
            # 使用spaCy的NER
            doc = self.nlp(text)
            for ent in doc.ents:
                entities.append((ent.text, ent.label_, ent.start_char, ent.end_char))
                self.logger.debug(f"发现实体：{ent.text} (类型: {ent.label_})")
        
        # 使用规则识别技术术语和概念
        tech_terms = self._identify_tech_terms(text)
        concepts = self._identify_concepts(text)
        
        # 添加自定义规则来识别更多实体
        custom_entities = self._extract_custom_entities(text)
        
        entities.extend(tech_terms)
        entities.extend(concepts)
        entities.extend(custom_entities)
        
        # 去重，保留第一次出现的实体
        unique_entities = []
        seen = set()
        for entity in entities:
            if entity[0] not in seen:
                seen.add(entity[0])
                unique_entities.append(entity)
        
        self.logger.debug(f"总共识别到 {len(unique_entities)} 个唯一实体")
        return unique_entities

    def extract_relationships(self, text: str, entities: List[Tuple[str, str, int, int]]) -> List[Tuple[str, str, str]]:
        """从文本中提取实体之间的关系"""
        relationships = []
        try:
            if not self.nlp:
                self.logger.warning("NLP模型未加载")
                return relationships
                
            doc = self.nlp(text)
            self.logger.debug(f"\n分析文本的依存关系：{text}")
            
            # 基于依存句法分析提取关系
            for token in doc:
                if token.dep_ in ["nsubj", "dobj", "pobj", "attr"]:
                    head = token.head.text
                    child = token.text
                    relationship = token.dep_
                    self.logger.debug(f"发现关系：{head} -{relationship}-> {child}")
                    relationships.append((head, relationship, child))
                    
            # 基于实体共现提取关系
            entity_texts = [e[0] for e in entities]
            for i, e1 in enumerate(entity_texts):
                for e2 in entity_texts[i+1:]:
                    if e1 != e2:
                        relationships.append((e1, "相关", e2))
                        self.logger.debug(f"发现共现关系：{e1} -相关-> {e2}")
                        
        except Exception as e:
            self.logger.error(f"提取关系时出错：{str(e)}")
            
        return relationships

    def _extract_custom_entities(self, text: str) -> List[Tuple[str, str, int, int]]:
        """使用自定义规则提取实体"""
        custom_entities = []
        
        # 时间相关
        time_pattern = r'\b\d{1,2}[:：]\d{1,2}\b|\b\d{4}年\d{1,2}月\d{1,2}日\b'
        time_matches = re.finditer(time_pattern, text)
        for match in time_matches:
            custom_entities.append((match.group(), 'TIME', match.start(), match.end()))
            
        # 数值相关
        number_pattern = r'\b\d+(?:\.\d+)?%?\b'
        number_matches = re.finditer(number_pattern, text)
        for match in number_matches:
            custom_entities.append((match.group(), 'NUMBER', match.start(), match.end()))
            
        # 引用表达式
        quote_pattern = r'"[^"]+"|「[^」]+」|『[^』]+』'
        quote_matches = re.finditer(quote_pattern, text)
        for match in quote_matches:
            custom_entities.append((match.group(), 'QUOTE', match.start(), match.end()))
        
        return custom_entities

    def _identify_tech_terms(self, text: str) -> List[Tuple[str, str, int, int]]:
        """识别技术术语"""
        tech_terms = []
        
        # 技术术语的特征模式
        patterns = [
            r'\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*\b',  # 驼峰命名
            r'\b[A-Z]+(?:\s*[A-Z]+)*\b',  # 全大写缩写
            r'\b\w+(?:-\w+)+\b',  # 带连字符的术语
            r'\b\w+(?:\.js|\.py|\.java|\.cpp)\b',  # 文件扩展名
            r'\b(?:https?|ftp)://\S+\b',  # URL
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                term = match.group()
                tech_terms.append((term, 'TECH', match.start(), match.end()))
                self.logger.debug(f"识别到技术术语: {term}")
        
        return tech_terms

    def _identify_concepts(self, text: str) -> List[Tuple[str, str, int, int]]:
        """识别概念"""
        concepts = []
        
        # 概念的特征词
        concept_indicators = [
            r'概念', r'理论', r'方法', r'原理',
            r'技术', r'系统', r'框架', r'模型',
            r'算法', r'结构', r'模式', r'架构'
        ]
        
        # 构建匹配模式
        pattern = f"({'|'.join(concept_indicators)})"
        matches = re.finditer(pattern, text)
        
        for match in matches:
            # 获取概念指示词前后的文本
            start = max(0, match.start() - 20)
            end = min(len(text), match.end() + 20)
            context = text[start:end]
            
            # 尝试提取完整概念
            concept_match = re.search(r'[\u4e00-\u9fa5a-zA-Z]+' + match.group() + r'[\u4e00-\u9fa5a-zA-Z]+', context)
            if concept_match:
                concept = concept_match.group()
                concepts.append((concept, 'CONCEPT', match.start(), match.end()))
                self.logger.debug(f"识别到概念: {concept}")
        
        return concepts

    def add_node(self, node_id: str, content: str, source_video: str, timestamp: float) -> Optional[KnowledgeNode]:
        """添加知识点节点到图谱中"""
        if self.connection_failed:
            self.logger.error("由于之前的连接失败，跳过节点添加")
            return
            
        try:
            # 检查参数有效性
            if not content or not content.strip():
                self.logger.error("节点内容为空")
                return None
                
            # 详细日志记录到文件
            self.logger.debug(f"处理节点: {content[:50]}...")
            
            # 提取实体和关系
            entities = self.extract_entities(content)
            relationships = self.extract_relationships(content, entities)
            
            # 创建节点向量表示
            try:
                vector = self.model.encode(content)
            except Exception as e:
                self.logger.error(f"创建节点向量失败: {str(e)}")
                vector = None
            
            # 创建知识点节点
            node = KnowledgeNode(
                id=node_id,
                content=content,
                source_video=source_video,
                timestamp=timestamp,
                vector=vector
            )
            
            # 添加实体信息
            entity_count = 0
            for entity, entity_type, _, _ in entities:
                if entity_type not in node.relationships:
                    node.relationships[entity_type] = []
                node.relationships[entity_type].append(entity)
                entity_count += 1
            
            self.total_entities += entity_count
            
            # 添加关系信息
            relation_count = 0
            for head, rel_type, tail in relationships:
                if rel_type not in node.relationships:
                    node.relationships[rel_type] = []
                node.relationships[rel_type].append(f"{head}->{tail}")
                relation_count += 1
            
            self.total_relations += relation_count
            
            # 存储到Neo4j
            max_retries = 3
            retry_delay = 1.5
            
            for attempt in range(max_retries):
                try:
                    # 创建Neo4j节点
                    neo4j_node = Node(
                        "Knowledge",
                        id=node_id,
                        content=content,
                        source_video=source_video,
                        timestamp=timestamp
                    )
                    
                    # 添加实体和关系属性
                    for entity_type, entities in node.relationships.items():
                        neo4j_node[entity_type] = json.dumps(entities, ensure_ascii=False)
                    
                    # 使用MERGE而不是CREATE来避免重复
                    self.neo4j_graph.merge(neo4j_node, "Knowledge", "content")
                    
                    # 创建关系
                    for head, rel_type, tail in relationships:
                        # 为头实体创建节点
                        head_node = Node("Entity", name=head)
                        self.neo4j_graph.merge(head_node, "Entity", "name")
                        
                        # 为尾实体创建节点
                        tail_node = Node("Entity", name=tail)
                        self.neo4j_graph.merge(tail_node, "Entity", "name")
                        
                        # 创建关系
                        rel = Relationship(head_node, rel_type, tail_node)
                        self.neo4j_graph.merge(rel)
                    
                    break
                    
                except Exception as e:
                    self.logger.error(f"Neo4j操作失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
                    if attempt < max_retries - 1:
                        import time
                        time.sleep(retry_delay * (attempt + 1))
                    else:
                        self.logger.error("❌ 达到最大重试次数，节点创建失败")
                        return None
            
            # 更新内存中的图谱
            self.nodes[node_id] = node
            self.graph.add_node(node_id)
            
            # 更新向量索引
            if vector is not None:
                if self.index is None:
                    # 初始化FAISS索引
                    self.index = faiss.IndexFlatL2(vector.shape[0])
                self.node_embeddings = np.vstack([self.node_embeddings, vector]) if self.node_embeddings.size else vector.reshape(1, -1)
                self.node_ids.append(node_id)
                self.index.add(vector.reshape(1, -1))
            
            self.total_nodes += 1
            # 显示进度信息
            if self.total_nodes % 5 == 0 or self.total_nodes == 1 or self.total_nodes == self.expected_total:
                self.logger.info(f"已处理 {self.total_nodes}/{self.expected_total} 个知识点 | 实体: {self.total_entities} | 关系: {self.total_relations}")
            
            return node
            
        except Exception as e:
            self.logger.error(f"❌ 节点创建失败: {str(e)}")
            return None

    def get_graph_statistics(self) -> Dict[str, any]:
        """获取知识图谱统计信息"""
        try:
            # 初始化统计信息
            stats = {
                'total_nodes': self.total_nodes,
                'entity_types': {},
                'relationship_types': defaultdict(int),
                'relationship_count': self.total_relations
            }
            
            # 统计实体类型
            entity_collections = defaultdict(list)
            for node in self.nodes.values():
                if hasattr(node, 'relationships'):
                    for entity_type, entities in node.relationships.items():
                        if isinstance(entities, list):
                            entity_collections[entity_type].extend(entities)
            
            # 去重并存储实体列表
            for entity_type, entities in entity_collections.items():
                stats['entity_types'][entity_type] = list(set(entities))
            
            # 获取Neo4j统计信息
            if self.neo4j_graph is not None:
                try:
                    neo4j_stats = self.neo4j_graph.run("""
                        MATCH (n)
                        OPTIONAL MATCH (n)-[r]->()
                        RETURN count(DISTINCT n) as nodes, count(DISTINCT r) as relationships
                    """).data()[0]
                    
                    stats['neo4j_nodes'] = neo4j_stats['nodes']
                    stats['neo4j_relationships'] = neo4j_stats['relationships']
                except Exception as e:
                    self.logger.error(f"获取Neo4j统计信息失败: {str(e)}")
                    stats['neo4j_nodes'] = 0
                    stats['neo4j_relationships'] = 0
            
            return stats
            
        except Exception as e:
            self.logger.error(f"获取图谱统计信息失败: {str(e)}")
            # 返回一个基本的统计信息
            return {
                'total_nodes': self.total_nodes,
                'entity_types': {},
                'relationship_types': defaultdict(int),
                'relationship_count': self.total_relations,
                'neo4j_nodes': 0,
                'neo4j_relationships': 0
            }

    def analyze_content_relationships(self):
        """分析节点之间的内容关系"""
        try:
            # 如果没有节点，直接返回
            if not self.nodes:
                self.logger.warning("没有节点可供分析")
                return
            
            self.logger.info("开始分析节点间的内容关系...")
            
            # 获取所有节点的内容向量
            node_ids = list(self.nodes.keys())
            content_vectors = []
            
            for node_id in node_ids:
                node = self.nodes[node_id]
                if node.vector is not None:
                    content_vectors.append(node.vector)
                else:
                    try:
                        # 如果节点没有向量，重新计算
                        node.vector = self.model.encode(node.content)
                        content_vectors.append(node.vector)
                    except Exception as e:
                        self.logger.error(f"计算节点 {node_id} 的向量失败: {str(e)}")
                        continue
            
            # 如果没有有效的向量，直接返回
            if not content_vectors:
                self.logger.warning("没有有效的内容向量可供分析")
                return
            
            # 计算节点间的相似度
            content_vectors = np.array(content_vectors)
            similarities = cosine_similarity(content_vectors)
            
            # 为每个节点找出最相关的其他节点
            threshold = 0.6  # 相似度阈值
            for i, node_id in enumerate(node_ids):
                related_indices = np.where(similarities[i] > threshold)[0]
                related_nodes = []
                
                for idx in related_indices:
                    if idx != i:  # 排除节点自身
                        similarity = similarities[i][idx]
                        related_node_id = node_ids[idx]
                        related_nodes.append((related_node_id, similarity))
                
                # 按相似度排序，取前5个最相关的节点
                related_nodes.sort(key=lambda x: x[1], reverse=True)
                top_related = related_nodes[:5]
                
                # 在Neo4j中创建关系
                if self.neo4j_graph is not None:
                    try:
                        for related_id, similarity in top_related:
                            # 创建RELATED_TO关系
                            query = """
                            MATCH (n1:Knowledge {id: $node1_id})
                            MATCH (n2:Knowledge {id: $node2_id})
                            MERGE (n1)-[r:RELATED_TO]->(n2)
                            SET r.similarity = $similarity
                            """
                            self.neo4j_graph.run(query, node1_id=node_id, 
                                               node2_id=related_id,
                                               similarity=float(similarity))
                    except Exception as e:
                        self.logger.error(f"创建节点关系失败: {str(e)}")
                
                # 更新节点的相关节点信息
                self.nodes[node_id].related_concepts = [rel_id for rel_id, _ in top_related]
            
            self.logger.info(f"完成内容关系分析，共处理 {len(node_ids)} 个节点")
            
        except Exception as e:
            self.logger.error(f"内容关系分析失败: {str(e)}")

    def __del__(self):
        """析构函数，显示最终统计信息"""
        if hasattr(self, 'total_nodes'):
            print(f"\n\n知识图谱处理完成！")
            print(f"✓ 节点：{self.total_nodes}")
            print(f"✓ 实体：{self.total_entities}")
            print(f"✓ 关系：{self.total_relations}")

class LearningPathPlanner:
    """学习路径规划器"""
    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.knowledge_graph = knowledge_graph
        self.model = knowledge_graph.model

    def evaluate_user_level(self, user_background: str) -> Dict[str, float]:
        """评估用户对各个知识点的理解程度"""
        knowledge_levels = {}
        
        # 将用户背景转换为向量
        user_vector = self.model.encode(user_background).reshape(-1)  # 确保是1维数组
        
        # 计算用户背景与每个知识点的相似度
        for node_id, node in self.knowledge_graph.nodes.items():
            if node.vector is None:
                node.vector = self.model.encode(node.content)
            node_vector = node.vector.reshape(-1)  # 确保是1维数组
            
            # 计算余弦相似度
            similarity = float(np.dot(user_vector, node_vector)) / (
                float(np.linalg.norm(user_vector)) * float(np.linalg.norm(node_vector))
            )
            
            # 将相似度转换为理解程度（0-1之间）
            knowledge_levels[node_id] = max(0.0, min(1.0, similarity))
        
        return knowledge_levels

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """计算两段文本的相似度"""
        vec1 = self.model.encode(text1).reshape(-1)  # 确保是1维数组
        vec2 = self.model.encode(text2).reshape(-1)  # 确保是1维数组
        return float(np.dot(vec1, vec2)) / (
            float(np.linalg.norm(vec1)) * float(np.linalg.norm(vec2))
        )

    def generate_learning_path(self, user_background: str, target_topics: List[str]) -> List[str]:
        """生成个性化学习路径"""
        if not self.knowledge_graph.nodes:
            return []

        # 首先分析知识点之间的关系
        self.knowledge_graph.analyze_content_relationships()

        # 评估用户知识水平
        knowledge_levels = self.evaluate_user_level(user_background)
        
        # 找到与目标主题相关的知识点
        target_nodes = []
        for topic in target_topics:
            topic_vector = self.model.encode(topic).reshape(-1)  # 确保是1维数组
            for node_id, node in self.knowledge_graph.nodes.items():
                if node.vector is None:
                    node.vector = self.model.encode(node.content)
                node_vector = node.vector.reshape(-1)  # 确保是1维数组
                
                # 计算相似度
                similarity = float(np.dot(topic_vector, node_vector)) / (
                    float(np.linalg.norm(topic_vector)) * float(np.linalg.norm(node_vector))
                )
                
                if similarity > 0.5:  # 相似度阈值
                    # 考虑用户已有知识水平
                    if knowledge_levels.get(node_id, 0) < 0.7:  # 如果用户对该知识点的掌握程度不高
                        target_nodes.append((node_id, similarity))
        
        # 按相似度排序
        target_nodes.sort(key=lambda x: x[1], reverse=True)
        
        # 生成学习路径
        path = []
        visited = set()
        
        for node_id, _ in target_nodes:
            if node_id not in visited:
                # 添加前置知识点
                current_node = self.knowledge_graph.nodes[node_id]
                if current_node.prerequisites:
                    # 按时间戳排序前置知识
                    prereqs = sorted(
                        [(prereq_id, self.knowledge_graph.nodes[prereq_id].timestamp)
                         for prereq_id in current_node.prerequisites
                         if prereq_id in self.knowledge_graph.nodes],
                        key=lambda x: x[1]
                    )
                    for prereq_id, _ in prereqs:
                        if prereq_id not in visited and knowledge_levels.get(prereq_id, 0) < 0.7:
                            path.append(prereq_id)
                            visited.add(prereq_id)
                
                # 添加当前知识点
                path.append(node_id)
                visited.add(node_id)
        
        return path

class LearningProgressTracker:
    """学习进度追踪器"""
    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.knowledge_graph = knowledge_graph
        self.progress = {}
    
    def update_progress(self, node_id: str, score: float):
        """更新学习进度"""
        self.progress[node_id] = score
    
    def get_progress(self) -> Dict[str, float]:
        """获取学习进度"""
        return self.progress
    
    def get_next_topic(self) -> Optional[str]:
        """获取下一个建议学习的主题"""
        # 找到进度最低的知识点
        min_score = float('inf')
        next_topic = None
        
        for node_id in self.knowledge_graph.nodes:
            score = self.progress.get(node_id, 0.0)
            if score < min_score:
                min_score = score
                next_topic = node_id
        
        return next_topic

class LearningPathSystem:
    # 初始化学习路径系统
    def __init__(self):
        self.planning_keywords = [
            "规划", "学习路径", "如何学", "怎么学", "学习方法",
            "学习计划", "路线图", "教学计划", "课程规划"
        ]
        
    # 判断是否为学习规划请求
    def is_planning_request(self, text: str) -> bool:
        return any(keyword in text for keyword in self.planning_keywords)
    
    # 生成学习规划
    def generate_plan(self, request: str) -> str:
        # 生成三个不同的方案
        plans = self._generate_multiple_plans(request)
        
        # 优化并整合方案
        final_plan = self._optimize_plans(plans, request)
        
        return final_plan
    
    # 生成多个学习方案
    def _generate_multiple_plans(self, request: str) -> List[str]:
        # 这里应实现实际的LLM调用逻辑
        # 示例返回三个方案
        return [
            self._generate_single_plan(request, "快速入门方案"),
            self._generate_single_plan(request, "深入学习方案"),
            self._generate_single_plan(request, "实践导向方案")
        ]
    
    # 生成单个学习方案
    def _generate_single_plan(self, request: str, plan_type: str) -> str:
        # 实际实现需要调用LLM
        return f"""
{plan_type}：

1. 学习目标
   - 目标1
   - 目标2

2. 学习步骤
   - 步骤1
   - 步骤2

3. 时间安排
   - 阶段1：xx时间
   - 阶段2：xx时间

4. 学习资源
   - 资源1
   - 资源2
"""
    
    # 优化并整合多个方案
    def _optimize_plans(self, plans: List[str], original_request: str) -> str:
        # 这里应该再次调用LLM来优化方案
        # 示例实现
        return f"""# 优化后的学习规划方案

## 总体目标
根据您的学习需求"{original_request}"，为您制定如下最优学习方案：

## 分阶段学习计划

### 第一阶段：基础入门
1. 学习内容...
2. 预期目标...
3. 建议时间...

### 第二阶段：进阶提升
1. 学习内容...
2. 预期目标...
3. 建议时间...

### 第三阶段：实战运用
1. 项目实践...
2. 技能提升...
3. 建议时间...

## 学习建议
1. 建议1...
2. 建议2...

## 推荐资源
1. 资源1...
2. 资源2...
"""