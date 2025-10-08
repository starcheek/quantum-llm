#!/usr/bin/env python3
"""
Setup script for Fintel package
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""

# Read requirements
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    if os.path.exists(requirements_path):
        with open(requirements_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return []

setup(
    name="quantum-llm",
    version="0.1.0",
    author="Quantum Research Team",
    author_email="quantum@example.com",
    description="Quantum LLM - Tools for quantum research data collection and analysis",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/quantum-research/quantum-llm",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
    install_requires=read_requirements(),
    entry_points={
        "console_scripts": [
            "quantum-llm=quantum_llm.cli:app",
        ],
    },
    keywords="quantum, llm, research, arxiv, knowledge-graph, neo4j, cli",
    project_urls={
        "Bug Reports": "https://github.com/quantum-research/quantum-llm/issues",
        "Source": "https://github.com/quantum-research/quantum-llm",
        "Documentation": "https://github.com/quantum-research/quantum-llm#readme",
    },
) 