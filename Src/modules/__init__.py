"""
CogniDir Modules

This package contains the core modules for the CogniDir framework:
- InitializedProportion: Initial sampling proportion module
- FakeNewsDetector: Main fake news detection model
- InfoDirichletResampling: Adaptive resampling mechanism
"""

from .initialized_proportion import InitializedProportion
from .fake_news_detector import FakeNewsDetector
from .infodirichlet_resampling import InfoDirichletResampling
from .dataset import SampledRationaleDataset

__all__ = [
    'InitializedProportion',
    'FakeNewsDetector',
    'InfoDirichletResampling',
    'SampledRationaleDataset'
]
