#!/usr/bin/env python3
"""
Project Context Generator for MLOps ETF Forecasting

This script creates a comprehensive markdown file containing all project information
that can be fed to AI models for context and understanding.
"""

import os
import glob
from pathlib import Path
from datetime import datetime

def read_file_content(file_path, max_lines=None, max_size=None):
    """Read file content with optional line and size limits"""
    try:
        # Check file size first
        file_size = os.path.getsize(file_path)
        if max_size and file_size > max_size:
            return f"File too large ({file_size} bytes) - content not included"
        
        # Try to read as text
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            if max_lines:
                lines = content.split('\n')
                content = '\n'.join(lines[:max_lines])
                if len(lines) > max_lines:
                    content += f"\n\n... (truncated at {max_lines} lines)"
            return content
    except UnicodeDecodeError:
        return f"Binary file ({file_size} bytes) - content not included"
    except Exception as e:
        return f"Error reading file: {e}"

def get_file_stats(file_path):
    """Get file statistics"""
    try:
        stat = os.stat(file_path)
        return {
            'size': stat.st_size,
            'modified': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
        }
    except:
        return {'size': 'Unknown', 'modified': 'Unknown'}

def create_project_context():
    """Create comprehensive project context markdown file"""
    
    project_root = Path(__file__).parent
    output_file = project_root / "PROJECT_CONTEXT.md"
    
    # Define what to exclude
    EXCLUDE_DIRS = {
        '.venv', '__pycache__', '.git', 'mlruns', 
        'node_modules', '.pytest_cache', '.mypy_cache',
        'build', 'dist', '*.egg-info'
    }
    
    EXCLUDE_FILES = {
        'PROJECT_CONTEXT.md', '.DS_Store', 'Thumbs.db',
        '*.pyc', '*.pyo', '*.log', '*.tmp', '*.cache'
    }
    
    EXCLUDE_EXTENSIONS = {
        '.pyc', '.pyo', '.log', '.tmp', '.cache', '.bak',
        '.swp', '.swo', '.DS_Store', '.gitignore'
    }
    
    MAX_FILE_SIZE = 1024 * 1024  # 1MB limit for file content
    
    def should_exclude_dir(dirname):
        """Check if directory should be excluded"""
        return any(exclude in dirname for exclude in EXCLUDE_DIRS)
    
    def should_exclude_file(filename):
        """Check if file should be excluded"""
        # Check exact filename matches
        if filename in EXCLUDE_FILES:
            return True
        
        # Check file extensions
        file_ext = Path(filename).suffix.lower()
        if file_ext in EXCLUDE_EXTENSIONS:
            return True
        
        # Check pattern matches
        for pattern in EXCLUDE_FILES:
            if '*' in pattern and filename.endswith(pattern.replace('*', '')):
                return True
        
        return False
    
    # Start building the context
    context = []
    context.append("# MLOps ETF Forecasting - Complete Project Context\n")
    context.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    context.append("This file contains the complete context of the MLOps ETF Forecasting project for AI consumption.\n")
    
    # Project Overview
    context.append("## 📋 Project Overview\n")
    context.append("This is a machine learning project focused on ETF price forecasting, specifically predicting whether the SPY ETF (S&P 500 ETF) will go up or down on the next trading day.\n")
    
    # Project Structure
    context.append("## 🏗️ Project Structure\n")
    context.append("```\n")
    for root, dirs, files in os.walk(project_root):
        # Filter out excluded directories
        dirs[:] = [d for d in dirs if not should_exclude_dir(d)]
        
        level = root.replace(str(project_root), '').count(os.sep)
        indent = ' ' * 2 * level
        context.append(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        
        # Filter and add files
        for file in files:
            if not should_exclude_file(file):
                context.append(f"{subindent}{file}")
    context.append("```\n")
    
    # README
    readme_path = project_root / "README.md"
    if readme_path.exists():
        context.append("## 📖 README\n")
        context.append("```markdown\n")
        context.append(read_file_content(readme_path))
        context.append("\n```\n")
    
    # Requirements
    requirements_path = project_root / "requirements.txt"
    if requirements_path.exists():
        context.append("## 📦 Dependencies\n")
        context.append("```\n")
        context.append(read_file_content(requirements_path))
        context.append("\n```\n")
    
    # Data Information
    context.append("## 📊 Data Information\n")
    context.append("### Raw Data Files\n")
    raw_data_dir = project_root / "data" / "raw"
    if raw_data_dir.exists():
        for csv_file in raw_data_dir.glob("*.csv"):
            stats = get_file_stats(csv_file)
            context.append(f"- **{csv_file.name}**: {stats['size']} bytes, modified {stats['modified']}\n")
    
    # Processed Data
    processed_data_dir = project_root / "data" / "processed"
    if processed_data_dir.exists():
        context.append("\n### Processed Data Files\n")
        for data_file in processed_data_dir.glob("*"):
            stats = get_file_stats(data_file)
            context.append(f"- **{data_file.name}**: {stats['size']} bytes, modified {stats['modified']}\n")
    
    # Notebooks
    context.append("\n## 📓 Jupyter Notebooks\n")
    notebooks_dir = project_root / "notebooks"
    if notebooks_dir.exists():
        for notebook in notebooks_dir.glob("*.ipynb"):
            context.append(f"\n### {notebook.name}\n")
            context.append("```python\n")
            content = read_file_content(notebook, max_lines=200, max_size=MAX_FILE_SIZE)
            context.append(content)
            context.append("\n```\n")
    
    # Docker and Infrastructure
    dockerfile_path = project_root / "Dockerfile"
    if dockerfile_path.exists():
        context.append("## 🐳 Docker Configuration\n")
        context.append("```dockerfile\n")
        context.append(read_file_content(dockerfile_path))
        context.append("\n```\n")
    
    # License
    license_path = project_root / "LICENSE"
    if license_path.exists():
        context.append("## 📄 License\n")
        context.append("```\n")
        context.append(read_file_content(license_path))
        context.append("\n```\n")
    
    # DVC Configuration
    dvc_files = list(project_root.glob("*.dvc")) + list((project_root / "data").glob("*.dvc"))
    if dvc_files:
        context.append("## 🔄 Data Version Control (DVC)\n")
        for dvc_file in dvc_files:
            context.append(f"\n### {dvc_file.name}\n")
            context.append("```yaml\n")
            context.append(read_file_content(dvc_file))
            context.append("\n```\n")
    
    # Project Summary
    context.append("\n## 🎯 Project Summary\n")
    context.append("""
This MLOps ETF Forecasting project demonstrates:

1. **Data Pipeline**: Automated data acquisition from Yahoo Finance (yfinance)
2. **Feature Engineering**: Technical indicators, lagged returns, volatility measures
3. **Machine Learning**: Binary classification for ETF price direction prediction
4. **MLOps Practices**: Data versioning (DVC), experiment tracking (MLflow), containerization
5. **Data Quality**: 3,288 samples with 33 features (improved from 588 samples by switching VIX→UVXY)

**Key Technologies**:
- Python 3.12, pandas, numpy, scikit-learn
- XGBoost, Optuna (hyperparameter tuning)
- MLflow (experiment tracking)
- DVC (data version control)
- Jupyter notebooks for development

**Target**: Predict SPY ETF price direction (up/down) for next trading day
**Features**: Technical indicators, multi-asset returns, volatility measures
**Dataset**: 3,288 samples spanning 2012-2025 with 33 engineered features
""")
    
    # Write the context file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(''.join(context))
    
    print(f"✅ Project context generated: {output_file}")
    print(f"📊 File size: {os.path.getsize(output_file)} bytes")
    
    return output_file

if __name__ == "__main__":
    create_project_context()
