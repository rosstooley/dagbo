import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dagbo",
    version="0.0.1",
    author="Ross Tooley",
    author_email="rjt80@cantab.ac.uk",
    description="Bayesian optimisation with semi-parametric DAG models.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rosstooley/dagbo",
    project_urls={
        "Bug Tracker": "https://github.com/rosstooley/dagbo/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.7",
)
