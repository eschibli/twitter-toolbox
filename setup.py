import sys , os
import setuptools # USe to build pckg

with open("README.md","r") as fh:
    long_description = fh.read()

setuptools.setup(
    name = "twitter_nlp_toolkit", # name shown on pypi and used with import
    version ="0.0.6",
    author = "Dr. Eric Schibli , Mohamad (Moe) Antar",
    author_email = "moe.antar14@gmail.com",
    description = "Tools for collecting , processing and analyzing twitter data",
    long_description = long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/eschibli/twitter-toolbox", # url of github repo
    keywords = "package numbers calculations",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Operating System :: OS Independent" 
    ],
    install_requires=[
          'numpy',
          'pandas',
          'scikit-learn',
          
      ],
    python_requires='>=3.6',
    include_package_data=True


)
