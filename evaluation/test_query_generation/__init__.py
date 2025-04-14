# __init__.py
"""
Test query generation package for the latent semantic index search engine.
"""

from .test_queries import (
    test_queries,
    get_all_queries,
    get_queries_by_category,
    get_queries_by_topic_id,
    get_metadata
)

__all__ = [
    'test_queries',
    'get_all_queries',
    'get_queries_by_category',
    'get_queries_by_topic_id',
    'get_metadata'
]