"""
Tests for utility functions.
"""
import os
import pytest
from src.utils.config import load_config


def test_load_config():
    """Test that config loading works with environment variables."""
    # Set test environment variables
    os.environ["TEST_VAR"] = "test_value"
    
    # Test with default value when env var exists
    assert load_config("TEST_VAR", "default") == "test_value"
    
    # Test with default value when env var doesn't exist
    assert load_config("NON_EXISTENT_VAR", "default") == "default"
    
    # Clean up
    del os.environ["TEST_VAR"]


def test_load_config_with_type_conversion():
    """Test that config loading works with type conversion."""
    # Set test environment variables
    os.environ["TEST_INT"] = "42"
    os.environ["TEST_FLOAT"] = "3.14"
    os.environ["TEST_BOOL"] = "True"
    
    # Test integer conversion
    assert load_config("TEST_INT", 0, int) == 42
    
    # Test float conversion
    assert load_config("TEST_FLOAT", 0.0, float) == 3.14
    
    # Test boolean conversion
    assert load_config("TEST_BOOL", False, lambda x: x.lower() == "true") is True
    
    # Clean up
    del os.environ["TEST_INT"]
    del os.environ["TEST_FLOAT"]
    del os.environ["TEST_BOOL"] 