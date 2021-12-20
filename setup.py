import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="GSKpy",
    version="1.3.2",
    author="Hazem Nomer",
    author_email="h.nomer@nu.edu.eg",
    description="Gaining-sharing knowledge alorithm GSK for continuous optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ha2emnomer/GSKPy",
    project_urls={
        "Bug Tracker": "https://github.com/ha2emnomer/GSKPy/issues",
    },

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",

)
