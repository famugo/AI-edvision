"""
1、实现功能
    实现了一个现代化的视频学习笔记系统
    支持笔记的增删改查
    包含笔记重要性、心情标记
    支持笔记模板和标签管理
    实现了笔记之间的关联关系
2、主要技术
    使用枚举类（Enum）定义笔记属性
    采用 emoji 进行心情和重要性标记
    实现了完整的 CRUD 操作
    使用 JSON 序列化支持数据持久化
    采用类型注解确保代码类型安全
    实现了笔记模板系统
    支持多维度的笔记过滤和检索"""

from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union, Set
import json
import os
import emoji
from enum import Enum
import time

class NoteImportance(Enum):
    LOW = "💡"      # 普通笔记
    MEDIUM = "⭐"   # 重要笔记
    HIGH = "🌟"     # 非常重要
    CRITICAL = "🔥"  # 关键笔记

class NoteTemplate(Enum):
    CONCEPT = "概念模板"
    QUESTION = "问题模板"
    SUMMARY = "总结模板"
    REVIEW = "复习模板"

class Note:
    def __init__(self, text: str, timestamp: float, importance: NoteImportance = NoteImportance.LOW,
                 tags: Set[str] = None, template_type: Optional[NoteTemplate] = None,
                 note_id: Optional[float] = None, timestamp_str: Optional[str] = None,
                 end_timestamp: Optional[float] = None):
        self.text = text
        self.timestamp = timestamp
        self.end_timestamp = end_timestamp
        self.importance = importance
        self.tags = tags if tags is not None else set()
        self.template_type = template_type
        self.id = note_id if note_id is not None else time.time()
        self.timestamp_str = timestamp_str if timestamp_str else self._format_timestamp(timestamp)

    def _format_timestamp(self, timestamp: float) -> str:
        """将时间戳格式化为字符串"""
        # 计算开始时间
        total_seconds = int(timestamp)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        milliseconds = int((timestamp - total_seconds) * 1000)
        start_time = f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
        
        # 使用传入的结束时间
        end_time = ""
        if self.end_timestamp is not None:
            end_seconds = int(self.end_timestamp)
            end_hours = end_seconds // 3600
            end_minutes = (end_seconds % 3600) // 60
            end_secs = end_seconds % 60
            end_time = f"{end_hours:02d}:{end_minutes:02d}:{end_secs:02d}.{milliseconds:03d}"
        else:
            # 如果没有结束时间，使用开始时间
            end_time = start_time
        
        return f"[{start_time} --> {end_time}]"

    def to_dict(self) -> Dict:
        """将笔记转换为字典格式"""
        return {
            'text': self.text,
            'timestamp': self.timestamp,
            'end_timestamp': self.end_timestamp,
            'importance': self.importance.value,
            'tags': list(self.tags),
            'template_type': self.template_type.value if self.template_type else None,
            'id': self.id,
            'timestamp_str': self.timestamp_str
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Note':
        """从字典创建笔记"""
        return cls(
            text=data['text'],
            timestamp=data['timestamp'],
            end_timestamp=data.get('end_timestamp'),
            importance=NoteImportance(data['importance']),
            tags=set(data['tags']),
            template_type=NoteTemplate(data['template_type']) if data.get('template_type') else None,
            note_id=data['id'],
            timestamp_str=data.get('timestamp_str')
        )

class NoteSystem:
    def __init__(self):
        self.notes: List[Note] = []
        self.templates: Dict[NoteTemplate, str] = {
            NoteTemplate.CONCEPT: "概念名称：\n定义：\n关键点：\n例子：\n",
            NoteTemplate.QUESTION: "问题：\n思考：\n答案：\n相关概念：\n",
            NoteTemplate.SUMMARY: "主要内容：\n重点总结：\n疑问：\n后续学习：\n",
            NoteTemplate.REVIEW: "知识点回顾：\n掌握程度：\n需要加强：\n"
        }
        
    def add_note(
        self,
        text: str,
        timestamp: float,
        end_timestamp: Optional[float] = None,
        importance: NoteImportance = NoteImportance.LOW,
        tags: Set[str] = None,
        template_type: Optional[NoteTemplate] = None
    ) -> bool:
        """添加新笔记"""
        if not text.strip():
            return False
            
        note = Note(
            text=text,
            timestamp=timestamp,
            end_timestamp=end_timestamp,
            importance=importance,
            tags=tags,
            template_type=template_type
        )
        self.notes.append(note)
        self.notes.sort(key=lambda x: x.timestamp)
        return True

    def get_notes(self, 
                 importance: Optional[NoteImportance] = None,
                 tags: Optional[Set[str]] = None,
                 template_type: Optional[NoteTemplate] = None) -> List[Note]:
        """获取笔记，支持多种过滤条件"""
        filtered_notes = self.notes
        
        if importance:
            filtered_notes = [n for n in filtered_notes if n.importance == importance]
        if tags:
            filtered_notes = [n for n in filtered_notes if tags & n.tags]
        if template_type:
            filtered_notes = [n for n in filtered_notes if n.template_type == template_type]
            
        return filtered_notes

    def get_note_by_id(self, note_id: int) -> Optional[Note]:
        """通过ID获取笔记"""
        for note in self.notes:
            if note.id == note_id:
                return note
        return None

    def edit_note(
        self,
        note_id: int,
        new_text: Optional[str] = None,
        new_timestamp: Optional[float] = None,
        new_end_timestamp: Optional[float] = None,
        new_importance: Optional[NoteImportance] = None,
        new_tags: Optional[Set[str]] = None
    ) -> bool:
        """编辑笔记"""
        note = self.get_note_by_id(note_id)
        if not note:
            return False
            
        if new_text is not None:
            note.text = new_text.strip()
        if new_timestamp is not None:
            note.timestamp = new_timestamp
            note.timestamp_str = note._format_timestamp(new_timestamp)
        if new_end_timestamp is not None:
            note.end_timestamp = new_end_timestamp
            note.timestamp_str = note._format_timestamp(note.timestamp)
        if new_importance is not None:
            note.importance = new_importance
        if new_tags is not None:
            note.tags = new_tags
            
        return True

    def delete_note(self, note_id: int) -> bool:
        """删除笔记"""
        for i, note in enumerate(self.notes):
            if note.id == note_id:
                self.notes.pop(i)
                # 删除其他笔记中对该笔记的引用
                for other_note in self.notes:
                    if note_id in other_note.related_notes:
                        other_note.related_notes.remove(note_id)
                return True
        return False

    def add_related_notes(self, note_id: int, related_note_ids: List[int]) -> bool:
        """添加相关笔记链接"""
        note = self.get_note_by_id(note_id)
        if not note:
            return False
            
        for related_id in related_note_ids:
            if related_id not in note.related_notes and self.get_note_by_id(related_id):
                note.related_notes.append(related_id)
        return True

    def get_template(self, template_type: NoteTemplate) -> str:
        """获取笔记模板"""
        return self.templates.get(template_type, "")

    def mark_note_reviewed(self, note_id: int) -> bool:
        """标记笔记已复习"""
        note = self.get_note_by_id(note_id)
        if not note:
            return False
            
        note.last_reviewed = datetime.now()
        return True

    def get_notes_for_review(self, days_threshold: int = 7) -> List[Note]:
        """获取需要复习的笔记"""
        review_threshold = datetime.now() - timedelta(days=days_threshold)
        return [
            note for note in self.notes
            if note.last_reviewed is None or note.last_reviewed < review_threshold
        ]

    def get_learning_progress(self) -> Dict:
        """获取学习进度统计"""
        if not self.notes:
            return {
                'total_notes': 0,
                'importance_distribution': {},
                'tags_distribution': {}
            }
        
        total_notes = len(self.notes)
        importance_dist = {}
        tags_dist = {}
        
        for note in self.notes:
            # 统计重要性分布
            imp = note.importance.name
            importance_dist[imp] = importance_dist.get(imp, 0) + 1
            
            # 统计标签分布
            if note.tags:
                for tag in note.tags:
                    tags_dist[tag] = tags_dist.get(tag, 0) + 1
        
        return {
            'total_notes': total_notes,
            'importance_distribution': importance_dist,
            'tags_distribution': tags_dist
        }
    
    def clear_all_notes(self):
        """清空所有笔记"""
        self.notes = []

    def save_notes(self, filepath: str) -> bool:
        """保存笔记到文件"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                notes_data = [note.to_dict() for note in self.notes]
                json.dump(notes_data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"Error saving notes: {e}")
            return False

    def load_notes(self, filepath: str) -> bool:
        """从文件加载笔记"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                notes_data = json.load(f)
                self.notes = [Note.from_dict(data) for data in notes_data]
                self.notes.sort(key=lambda x: x.timestamp)
            return True
        except FileNotFoundError:
            self.notes = []
            return False
        except Exception as e:
            print(f"Error loading notes: {e}")
            return False

    def export_notes(self, filepath: str, format_type: str = 'txt') -> bool:
        """导出笔记"""
        try:
            if format_type == 'txt':
                with open(filepath, 'w', encoding='utf-8') as f:
                    for note in self.notes:
                        f.write(f"时间: [{note.timestamp_str}]\n")
                        f.write(f"重要性: {note.importance.value}\n")
                        if note.tags:
                            f.write(f"标签: {', '.join(note.tags)}\n")
                        if note.template_type:
                            f.write(f"模板类型: {note.template_type.value}\n")
                        f.write(f"内容:\n{note.text}\n")
                        if note.related_notes:
                            related_texts = []
                            for rel_id in note.related_notes:
                                rel_note = self.get_note_by_id(rel_id)
                                if rel_note:
                                    related_texts.append(f"[{rel_note.timestamp_str}] {rel_note.text[:30]}...")
                            f.write(f"相关笔记:\n" + "\n".join(related_texts) + "\n")
                        f.write("\n" + "="*50 + "\n\n")
            elif format_type == 'json':
                return self.save_notes(filepath)
            return True
        except Exception as e:
            print(f"Error exporting notes: {e}")
            return False