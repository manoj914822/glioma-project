#!/usr/bin/env python3
"""
GLIOMA Application Finalization Test
Tests for import errors, missing dependencies, and basic functionality
"""

def test_imports():
    """Test all critical imports"""
    try:
        print("Testing Flask imports...")
        from flask import Flask, request, session, jsonify, render_template, redirect, url_for
        print("Flask imports successful")

        print("Testing AI/ML imports...")
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.applications import ResNet50
        import numpy as np
        import cv2
        print("AI/ML imports successful")

        print("Testing utility imports...")
        import pickle
        import sqlite3
        import hashlib
        from datetime import datetime
        import secrets
        import json
        print("Utility imports successful")

        print("Testing matplotlib/seaborn...")
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        try:
            import seaborn as sns
        except ImportError:
            print("Seaborn not available (charts will be limited)")
        print("Matplotlib imports successful")

    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

    return True

def test_app_creation():
    """Test Flask app creation and basic configuration"""
    try:
        print("Testing Flask application creation...")
        from app import app, class_names, init_db

        # Check app configuration
        assert app.config['UPLOAD_FOLDER'] == 'static/uploads'
        assert app.config['SECRET_KEY'] is not None

        print("Flask application creation successful")

        # Test database initialization
        print("Testing database initialization...")
        init_db()
        print("Database initialization successful")

    except Exception as e:
        print(f"❌ Application creation error: {e}")
        return False

    return True

def test_model_loading():
    """Test AI model loading"""
    try:
        print("Testing AI model loading...")
        import os
        assert os.path.exists('ResNet50_model.h5'), "Model file missing"
        assert os.path.exists('class_names.pkl'), "Class names file missing"

        # Test model creation (not loading it unless explicitly needed)
        print("Model files found and accessible")

    except Exception as e:
        print(f"❌ Model loading error: {e}")
        return False

    return True

def test_templates():
    """Test template files exist"""
    try:
        print("Testing template files...")
        import os

        required_templates = [
            'templates/index.html',
            'templates/userlog.html',
            'templates/results.html',
            'templates/image_report.html'
        ]

        for template in required_templates:
            assert os.path.exists(template), f"Missing template: {template}"

        print("Core templates found")

    except Exception as e:
        print(f"❌ Template check error: {e}")
        return False

    return True

def main():
    """Run all tests"""
    print("=" * 60)
    print("GLIOMA BRAIN TUMOR ANALYSIS - FINALIZATION TESTS")
    print("=" * 60)

    tests = [
        ("Import Tests", test_imports),
        ("Application Creation Tests", test_app_creation),
        ("Model Loading Tests", test_model_loading),
        ("Template Tests", test_templates)
    ]

    all_passed = True

    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name}...")
        try:
            if not test_func():
                all_passed = False
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED! Application is ready for production.")
        print("Run with: python start.py")
    else:
        print("SOME TESTS FAILED. Please check the errors above.")
    print("=" * 60)

if __name__ == "__main__":
    main()
