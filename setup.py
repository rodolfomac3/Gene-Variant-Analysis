from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="genomic-variant-analysis",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Advanced Genomic Variant Analysis Pipeline with MLOps",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rodolfomac3/Gene-Variant-Analysis",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "train-variant-model=scripts.train:main",
            "predict-variants=scripts.predict:main",
        ],
    },
)