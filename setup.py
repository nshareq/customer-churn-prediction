from setuptools import setup, find_packages

setup(
    name="customer-churn-prediction",
    version="0.1.0",
    description="A machine learning pipeline for customer churn prediction.",
    long_description=open("README.md").read(),  
    long_description_content_type="text/markdown",
    author="Naim Shareq",
    author_email="naimshareq@gmail.com",
    url="https://github.com/nshareq/customer-churn-prediction",
    packages=find_packages(exclude=("tests", "docs")),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=0.24.0",
        "mlflow>=2.0.0",
        "pydantic>=1.10.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "pre-commit>=3.0.0",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "customer-churn=customer_churn.cli:main",
        ]
    },
    include_package_data=True,
    zip_safe=False,
)
