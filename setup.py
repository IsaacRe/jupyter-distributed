from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="jupyter-distributed",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A Jupyter extension for distributed parallel execution of cells",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/jupyter-distributed",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Framework :: Jupyter",
    ],
    python_requires=">=3.7",
    install_requires=[
        "ipython>=7.0.0",
        "jupyter>=1.0.0",
        "multiprocess>=0.70.0",
        "dill>=0.3.0",
    ],
    entry_points={
        "console_scripts": [
            "jupyter-distributed=jupyter_distributed.cli:main",
        ],
    },
)
