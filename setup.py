import sys , os
import setuptools # USe to build pckg

with open("README.md","r") as fh:
    long_description = fh.read()

requirements = []
with open('requirements.txt', 'r') as fh:
    for line in fh:
        requirements.append(line.strip())
        
setuptools.setup(
    name = "twitter_nlp_toolkit", # name shown on pypi and used with import
    version ="0.1.6",
    author = "Eric Schibli , Mohamad (Moe) Antar",
    author_email = "moe.antar14@gmail.com",
    description = "Tools for collecting , processing and analyzing twitter data",
    long_description = long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/eschibli/twitter-toolbox", # url of github repo
    keywords = "twitter sentiment analysis",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Operating System :: OS Independent" 
    ],
    install_requires = requirements
    python_requires='>=3.6',
    include_package_data=True


)
