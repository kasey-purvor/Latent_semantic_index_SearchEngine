"""
Indexing package for the Latent Semantic Search Engine.
Provides different indexing implementations and utilities.
"""

from .base import BaseIndexer
from .utils import (
    load_papers,
    extract_fields,
    combine_fields,
    create_output_dirs
)

# Re-export for backward compatibility
__all__ = [
    'BaseIndexer',
    'load_papers',
    'extract_fields',
    'combine_fields',
    'create_output_dirs'
] 