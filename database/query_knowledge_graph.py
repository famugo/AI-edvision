"""
知识图谱查询工具
用于查询 Neo4j 数据库中存储的知识图谱内容
"""

from py2neo import Graph
from typing import List, Dict, Optional
import pandas as pd


class KnowledgeGraphQuery:
    def __init__(self):
        # 连接 Neo4j 数据库
        self.graph = Graph("neo4j://localhost:7687", auth=("neo4j", "cjx20040328"))

    def get_all_nodes(self) -> pd.DataFrame:
        """获取所有知识点节点"""
        query = """
        MATCH (n)
        RETURN n
        """
        result = self.graph.run(query)
        nodes = []
        for record in result:
            node = record['n']
            nodes.append({
                'id': node.get('id', ''),
                'content': node.get('content', ''),
                'source_video': node.get('source_video', ''),
                'entity_type': node.get('entity_type', ''),
                'difficulty': node.get('difficulty', 0.5)
            })
        return pd.DataFrame(nodes)

    def get_node_by_id(self, node_id: str) -> Dict:
        """根据ID获取特定知识点节点"""
        query = """
        MATCH (n {id: $node_id})
        RETURN n
        """
        result = self.graph.run(query, node_id=node_id).data()
        if result:
            node = result[0]['n']
            return dict(node)
        return {}

    def get_relationships(self) -> pd.DataFrame:
        """获取所有知识点之间的关系"""
        query = """
        MATCH (n1)-[r]->(n2)
        RETURN n1.id as source, n2.id as target, type(r) as relationship_type
        """
        result = self.graph.run(query)
        relationships = []
        for record in result:
            relationships.append({
                'source': record['source'],
                'target': record['target'],
                'relationship_type': record['relationship_type']
            })
        return pd.DataFrame(relationships)

    def get_prerequisites(self, node_id: str) -> List[Dict]:
        """获取指定知识点的前置知识"""
        query = """
        MATCH (n {id: $node_id})<-[r:PREREQUISITE]-(prereq)
        RETURN prereq
        """
        result = self.graph.run(query, node_id=node_id)
        prerequisites = []
        for record in result:
            prerequisites.append(dict(record['prereq']))
        return prerequisites

    def get_related_concepts(self, node_id: str) -> List[Dict]:
        """获取指定知识点的相关概念"""
        query = """
        MATCH (n {id: $node_id})-[r:RELATED_TO]-(related)
        RETURN related
        """
        result = self.graph.run(query, node_id=node_id)
        related = []
        for record in result:
            related.append(dict(record['related']))
        return related

    def search_nodes_by_content(self, keyword: str) -> pd.DataFrame:
        """根据内容关键词搜索知识点"""
        query = """
        MATCH (n)
        WHERE n.content CONTAINS $keyword
        RETURN n
        """
        result = self.graph.run(query, keyword=keyword)
        nodes = []
        for record in result:
            node = record['n']
            nodes.append({
                'id': node.get('id', ''),
                'content': node.get('content', ''),
                'source_video': node.get('source_video', ''),
                'entity_type': node.get('entity_type', ''),
                'difficulty': node.get('difficulty', 0.5)
            })
        return pd.DataFrame(nodes)


def main():
    # 使用示例
    kg_query = KnowledgeGraphQuery()
    
    print("1. 获取所有知识点：")
    all_nodes = kg_query.get_all_nodes()
    print(all_nodes)
    
    print("\n2. 获取所有关系：")
    relationships = kg_query.get_relationships()
    print(relationships)
    
    print("\n3. 搜索包含特定关键词的知识点：")
    keyword = "故宫"  # 示例关键词
    search_results = kg_query.search_nodes_by_content(keyword)
    print(search_results)


if __name__ == "__main__":
    main()
