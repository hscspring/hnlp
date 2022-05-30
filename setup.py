import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="hnlp",
    version="0.0.1",
    author="Yam",
    author_email="haoshaochun@gmail.com",
    description="Humanly Deeplearning NLP.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hscspring/hnlp",
    packages=setuptools.find_packages(
        exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    install_requires=[
        'pnlp',
    ],
    package_data={
        'hnlp': ["task/*.json", "config/vocab/bert/*.txt", "config/model/*.json", "config/vocab/bert/*.json"]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
