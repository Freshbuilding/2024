"""
This module provides setup configuration for a Python package,
including dynamically loading package requirements from a file.
"""

from typing import List
from setuptools import find_packages, setup

HYPEN_E_DOT = '-e .'


def get_requirements(file_path: str) -> List[str]:
    '''
    This function returns the list of requirements from a requirements file.
    '''
    requirements = []
    with open(file_path, "r", encoding="utf-8") as file_obj:
        requirements = [req.strip() for req in file_obj.readlines() if req.strip() != HYPEN_E_DOT]

    return requirements


requirements_file_path = 'requirements.txt'


setup(
    name='2024',
    version='0.0.1',
    author='Vincent',
    packages=find_packages(),
    install_requires=get_requirements(requirements_file_path)
)
