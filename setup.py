from setuptools import setup, find_packages

setup(
    name="auto_pwlf",
    version="0.1.0",
    author="Nedeesha Weerasuriya",
    author_email="nedeeshawork@gmail.com",
    description="Automated piecewise linear fitting",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/auto_pwlf",
    packages=find_packages(),
    install_requires=[
        "numpy" >= 2.0,
        "scipy" >= 1.0,
        "matplotlib" >= 3.0,
        "pwlf" >= 2.0,
        "pandas" >= 2.0,
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
    ],
    extras_require={"docs": ["sphinx", "sphinx_rtd_theme",]},
    python_requires=">=3.7",
)
