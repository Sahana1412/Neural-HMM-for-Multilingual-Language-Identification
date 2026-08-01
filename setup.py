"""Setup script for neural-hmm-lang-id package."""

from setuptools import setup, find_packages

setup(
    name="neural-hmm-lang-id",
    version="1.0.0",
    description="Neural-HMM for Multilingual Language Identification: Comparative Study",
    author="Research Team",
    author_email="research@example.com",
    url="https://github.com/yourusername/neural-hmm-lang-id",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.11",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "torch>=2.0.0",
        "pyyaml>=6.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tqdm>=4.66.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "ruff>=0.0.292",
            "mypy>=1.5.0",
            "isort>=5.12.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "notebook>=7.0.0",
            "ipywidgets>=8.1.0",
            "umap-learn>=0.5.0",
        ],
        "tracking": [
            "tensorboard>=2.14.0",
            "wandb>=0.15.0",
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
