from setuptools import setup, find_packages

setup(
    name="autopwlf",
    version="0.7.1",
    author="Nedeesha Weerasuriya",
    author_email="nedeeshawork@gmail.com",
    description="Automated piecewise linear fitting",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/nweerasuriya/auto_pwlf",
    packages=["autopwlf"],
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.11.4",
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
    project_urls={"GitHub": "https://github.com/nweerasuriya/auto_pwlf"},
    license="MIT",
)