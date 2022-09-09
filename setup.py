#!/usr/bin/env python3

from pathlib import Path
from typing import List

from setuptools import setup, find_packages


def parse_requirements_txt(filename: str) -> List[str]:
    """Return requirements from requirements file."""
    with open(str(Path(__file__).with_name(filename))) as f:
        return f.readlines()

long_description = "Multimodal array allows handling multimodal numpy array and dataframes"

setup(name='multimodal_array',
      version='0.0.2-dev',
      description='A multimodal array interface',
      author='otavio.napoli',
      author_email='onapoli@lmcad.ic.unicamp.br',
      license="MIT",
      long_description=long_description,
      long_description_content_type="text/markdown",
      url="",
      install_requires=parse_requirements_txt("requirements.txt"),
      classifiers=[
          "Intended Audience :: Developers",
          "Intended Audience :: Science/Research",
          "Programming Language :: Python :: 3",
          "Programming Language :: Python :: 3.7",
          "Programming Language :: Python :: 3.8",
          "Programming Language :: Python :: 3.9",
          "License :: OSI Approved :: MIT",
          "Operating System :: OS Independent",
          ],
      platforms='any',
      packages=find_packages(),
      python_requires=">=3.7",
)
