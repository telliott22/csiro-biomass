"""Model architectures for pasture biomass prediction."""

from .baseline import BaselineModel
from .teacher import TeacherModel
from .student import StudentModel
from .auxiliary import AuxiliaryModel
from .losses import DistillationLoss, AuxiliaryLoss, CompetitionLoss

__all__ = [
    'BaselineModel',
    'TeacherModel',
    'StudentModel',
    'AuxiliaryModel',
    'DistillationLoss',
    'AuxiliaryLoss',
    'CompetitionLoss'
]
