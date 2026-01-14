from dataclasses import dataclass
from typing import Dict
import numpy as np


@dataclass
class EmbeddedChunk:
    vector: np.ndarray # embedding 向量（维度由模型决定）
    metadata: Dict # TextChunk.metadata