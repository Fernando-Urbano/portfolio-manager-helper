from setuptools import setup, find_packages

setup(
    name="portfolio_management_helper",
    version="0.1.0",
    description="Portfolio Management Helper",
    author="Fernando Rocha Urbano",
    author_email="fernando.rocha.urbano@gmail.com",
    packages=find_packages(),
    install_requires=[
        "pandas==2.2.3",
        "numpy==2.1.1",
        "matplotlib==3.9.2",
        "scipy==1.14.1",
        "yfinance==0.2.43",
        "datetime==3.0.3",
        "statsmodels==0.14.3",
        "scikit-learn==1.4.2",
        "seaborn==0.13.2",
        "openpyxl==3.1.3",
        "ipywidgets==8.1.5",
        "arch==7.1.0",
        "setuptools==57.5.0"
    ],
    python_requires='>=3.8',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)