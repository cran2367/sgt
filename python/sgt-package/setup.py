import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sgt",
    version="2.0.3",
    author="Chitta Ranjan",
    author_email="cran2367@gmail.com",
    description="Sequence Graph Transform (SGT) is a sequence embedding function. SGT extracts the short- and long-term sequence features and embeds them in a finite-dimensional feature space. The long and short term patterns embedded in SGT can be tuned without any increase in the computation.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cran2367/sgt",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)