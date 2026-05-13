from setuptools import setup, find_packages
from typing import List

def get_requirements(file_path: str) -> List[str]:
    requirement_lst: List[str] = []
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                requirement=line.strip()
                #ignore empty lines and -e
                if requirement and requirement != '-e .':
                    requirement_lst.append(requirement)
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
    return requirement_lst

setup(
    name='healthcare-outcome-mlops',
    version='0.1.0',
    author='Carlos Roa',
    author_email='roapalaciocarlos@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
)