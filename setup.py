# Complex_regional_systems/setup.py
from setuptools import setup, find_packages

setup(
    name="village_simulation",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.1.0",
        "gym>=0.17.0",
        "pyyaml>=5.3.0"
    ]
)