"""
Simple UI validators for QModelTrainer plugin

These are lightweight pre-checks before sending to server.
The server performs complete validation.
"""

import os
import json
from pathlib import Path
from typing import Tuple, Optional, Dict


def validate_dataset_path_exists(dataset_path: str) -> Tuple[bool, Optional[str]]:
    """Quick check that dataset path exists."""
    if not dataset_path:
        return False, "Dataset path is empty"
    
    if not os.path.exists(dataset_path):
        return False, f"Dataset path does not exist: {dataset_path}"
    
    if not os.path.isdir(dataset_path):
        return False, f"Dataset path is not a directory: {dataset_path}"
    
    return True, None


def validate_metadata_exists(dataset_path: str) -> Tuple[bool, Optional[str], Optional[Dict]]:
    """Quick check that metadata file exists and is readable."""
    metadata_path = Path(dataset_path) / 'qannotate_metadata.json'
    
    if not metadata_path.exists():
        return False, "QAnnotate metadata file not found (qannotate_metadata.json)", None
    
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        return True, None, metadata
    except Exception as e:
        return False, f"Error reading metadata: {str(e)}", None


def get_basic_dataset_info(metadata: Dict) -> Dict:
    """Extract basic info for UI display (no validation)."""
    class_catalog = metadata.get('class_catalog', {})
    classes = class_catalog.get('classes', [])
    
    class_names = ['Background']
    class_names.extend([c['label'] for c in classes])
    
    export_info = metadata.get('export_info', {})
    
    return {
        'num_classes': len(classes) + 1,
        'class_names': class_names,
        'export_format': export_info.get('export_format'),
        'image_format': export_info.get('image_format', 'tif'),
        'mask_format': export_info.get('mask_format', 'tif'),
        'num_images': metadata.get('statistics', {}).get('num_images', 0),
        'metadata': metadata
    }


def format_dataset_info(dataset_info: Dict) -> str:
    """Format dataset info for UI display."""
    lines = [
        f"Format: {dataset_info['export_format']}",
        f"Images: {dataset_info['num_images']}",
        f"Classes: {dataset_info['num_classes']} (including background)",
        f"Class names: {', '.join(dataset_info['class_names'][:5])}{'...' if len(dataset_info['class_names']) > 5 else ''}"
    ]
    return '\n'.join(lines)