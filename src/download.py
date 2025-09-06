"""
Download script for Roboflow dataset
"""

from roboflow import Roboflow
import sys
from pathlib import Path


def download_dataset(api_key: str = "jEfYRaQcHGaKHlxNvu1d",
                     workspace: str = "ai-sjglq",
                     project_name: str = "basketball-bs0zc-lj816",
                     version: int = 1,
                     format: str = "coco") -> str:
    """
    Download dataset from Roboflow
    
    Args:
        api_key: Roboflow API key
        workspace: Workspace slug from the dataset URL
        project_name: Project slug from the dataset URL
        version: Dataset version to download
        format: Format to download (coco, yolov5, etc.)
        
    Returns:
        Path to downloaded dataset
    """
    # Initialize Roboflow
    rf = Roboflow(api_key=api_key)
    
    # Get project
    project = rf.workspace(workspace).project(project_name)
    
    # Download dataset
    dataset = project.version(version).download(format)
    
    # Print location for parsing by other scripts
    print(f"Saved to: {dataset.location}")
    
    return dataset.location


def main():
    """Main entry point for command line usage"""
    # You can customize these parameters or read from command line args
    dataset_path = download_dataset(
        api_key="jEfYRaQcHGaKHlxNvu1d",
        workspace="ai-sjglq", 
        project_name="basketball-bs0zc-lj816",
        version=1,
        format="coco"
    )
    
    return dataset_path


if __name__ == "__main__":
    main()
