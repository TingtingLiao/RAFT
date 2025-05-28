from setuptools import setup, find_packages

setup(
    name='RAFT',
    version='0.0.1',
    description='RAFT', 
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'tqdm',  
    ],
)