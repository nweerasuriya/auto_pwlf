from setuptools import setup, find_packages

setup(
    name="auto_pwlf",
    version="0.1.0",
    author="Nedeesha Weerasuriya",
    author_email="nedeeshawork@gmail.com",
    description="Automated piecewise linear fitting",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/nweerasuriya/auto_pwlf",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=2.0.1",
        "scipy>=1.14",
        "pwlf>=2.2.1",
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
    extras_require={
        "docs": ["sphinx>=3.0", "sphinx_rtd_theme"],
    },
    python_requires=">=3.7",
)