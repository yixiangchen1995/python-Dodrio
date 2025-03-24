'''
FilePath: /python-Dodrio/setup.py
Descripttion: 
Author: Yixiang Chen
version: 
Date: 2025-03-24 15:57:41
LastEditors: Yixiang Chen
LastEditTime: 2025-03-24 19:53:05
'''

from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'Data Package for TTS '
LONG_DESCRIPTION = 'Data Package for TTS with a slightly longer description'

# 配置
setup(
        name="dodrio", 
        version=VERSION,
        author="Yixiang Chen",
        author_email="<yixiangchen1995@gmail.com>",
        license='MIT',
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        url="https://github.com/yixiangchen1995/python-Dodrio",
        packages=find_packages(),
        install_requires=[
            'numpy',
            'pandas',
            'pyarrow',
            'scipy',
            'tqdm',
            'librosa'
        ], # add any additional packages that 
        
        keywords=['python', 'first package'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Developers",
            'Programming Language :: Python',
            'Programming Language :: Python :: 2',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.4',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            'Programming Language :: Python :: 3.10',
            'Programming Language :: Python :: 3.11',
            'Programming Language :: Python :: 3.12',
            'Programming Language :: Python :: 3.13',
        ]
)