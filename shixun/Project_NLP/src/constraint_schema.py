"""
SCoRE2026 统一约束Schema定义

所有领域的约束都遵循此Schema，确保求解器间的互操作性。
融合域求解时，先求解领域A，结果注入领域B。
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Union
from enum import Enum


class Domain(str, Enum):
    TIME = "time"
    SPACE = "space"
    SOCIAL = "social"
    NATURE = "nature"


class QuestionType(str, Enum):
    FILL_BLANK = "fill_blank"       # ____填空
    SELECT_CORRECT = "select_correct"   # 选正确
    SELECT_INCORRECT = "select_incorrect"  # 选不正确


# ============================================================
# 时间域约束
# ============================================================

@dataclass
class AbsoluteTimeConstraint:
    """绝对时间约束: 事件发生在某个确定的时间点"""
    event: str
    time_point: str  # e.g., "星期三", "1994"


@dataclass
class RelativeTimeConstraint:
    """相对时间约束: 事件A相对于事件B的时间偏移"""
    event_a: str
    event_b: str
    relation: str       # "before" | "after"
    offset: int         # 偏移量
    unit: str = "day"   # 单位: day, year, week


@dataclass
class TimeConstraints:
    """时间域约束集合"""
    absolute: List[AbsoluteTimeConstraint] = field(default_factory=list)
    relative: List[RelativeTimeConstraint] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    is_weekly_cycle: bool = False  # 是否每周循环


# ============================================================
# 空间域约束
# ============================================================

class SpatialStructure(str, Enum):
    GRID_3X2 = "grid_3x2"         # 3层×2列花架/货架
    CIRCULAR = "circular"          # 圆形排列
    LINEAR = "linear"              # 线性排列


@dataclass
class PositionConstraint:
    """位置约束: 某实体在确定位置"""
    entity: str
    row: Optional[int] = None
    col: Optional[str] = None  # "东"/"西" 或 "左"/"右"
    position: Optional[str] = None  # 圆中位置描述


@dataclass
class SpatialRelationConstraint:
    """空间关系约束: 实体间的方位关系"""
    entity_a: str
    entity_b: str
    relation: str  # "above", "below", "adjacent_left", "adjacent_right",
                   # "same_row", "same_col", "diagonal",
                   # "left_of", "right_of", "clockwise", "counterclockwise"
    gap: int = 0  # 间隔层数/位置数


@dataclass
class SpaceConstraints:
    """空间域约束集合"""
    structure: SpatialStructure = SpatialStructure.GRID_3X2
    positions: List[PositionConstraint] = field(default_factory=list)
    relations: List[SpatialRelationConstraint] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    rows: int = 3
    cols: int = 2
    col_labels: List[str] = field(default_factory=lambda: ["东", "西"])


# ============================================================
# 社会域约束
# ============================================================

@dataclass
class KinshipConstraint:
    """亲属/社会关系约束"""
    person_a: str
    person_b: str
    relation: str  # "父亲", "母亲", "姐姐", "弟弟", "老公", "老婆",
                   # "儿子", "女儿", "岳母", "岳父", "儿媳", "女婿",
                   # "姐夫", "小姨", "奶奶", "爷爷", "外公", "外婆",
                   # "teacher", "student", "leader", "buddy",
                   # "ex-girlfriend", "ex-wife", "ex-husband",
                   # "classmate", "neighbor", "bestie", "boss"
    # 从关系可以推断的性别和代际信息
    inferred_gender_a: Optional[str] = None  # "male" | "female"
    inferred_gender_b: Optional[str] = None
    generation_diff: Optional[int] = None  # A比B高几代 (正=长辈, 负=晚辈)


@dataclass
class SocialConstraints:
    """社会域约束集合"""
    relations: List[KinshipConstraint] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    # 推理出的信息
    genders: Dict[str, str] = field(default_factory=dict)  # name -> male/female
    generations: Dict[str, int] = field(default_factory=dict)  # name -> generation level


# ============================================================
# 自然域约束
# ============================================================

@dataclass
class PropertyConstraint:
    """属性约束: 某位置的实体具有某属性"""
    position: str       # e.g., "1号田", "photo No.1"
    property_name: str
    property_value: str
    entity: Optional[str] = None  # 可选：已知该位置的实体名


@dataclass
class CategoryConstraint:
    """类别约束: 某位置的实体属于某类别"""
    position: str
    category: str  # e.g., "蔬菜", "fish", "工具"


@dataclass
class EntityProperty:
    """实体属性定义"""
    entity: str
    properties: Dict[str, Any] = field(default_factory=dict)
    # e.g., {"类别": "蔬菜", "花色": "黄色", "可食用部分": "种子"}


@dataclass
class NatureConstraints:
    """自然域约束集合"""
    entities: List[str] = field(default_factory=list)
    positions: List[str] = field(default_factory=list)
    property_constraints: List[PropertyConstraint] = field(default_factory=list)
    category_constraints: List[CategoryConstraint] = field(default_factory=list)
    entity_properties: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # 已知的实体属性表


# ============================================================
# 统一约束容器
# ============================================================

@dataclass
class StructuredConstraints:
    """统一约束容器，用于所有领域"""
    domain: str  # e.g., "time", "space", "social", "nature", "time+social"
    language: str = "cn"
    time: Optional[TimeConstraints] = None
    space: Optional[SpaceConstraints] = None
    social: Optional[SocialConstraints] = None
    nature: Optional[NatureConstraints] = None
    raw_text: str = ""
