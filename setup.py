
from setuptools import setup, find_packages

VERSION = '0.1.1' 
DESCRIPTION = 'Data Package for TTS '
#LONG_DESCRIPTION = 'Data Package for TTS with a slightly longer description'
LONG_DESCRIPTION = open("README.md", encoding="utf-8").read()

# 配置
setup(
        name="dodrio", 
        version=VERSION,
        author="Yixiang Chen",
        author_email="<yixiangchen1995@gmail.com>",
        license='MIT',
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        long_description_content_type="text/markdown",
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