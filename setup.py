import os

from setuptools import find_packages, setup

install_requires = [
]

extra_requires = [
    "black==21.4b2",
    "flake8==3.9.2",
    "isort==5.8.0",
    "pre-commit==2.15.0",
]


package_name = "clasymm"

setup(
    name=package_name,
    version="0.4",
    author="SIA",
    install_requires=install_requires,
    extra_requires=extra_requires,
    packages=[os.path.join(package_name, i) for i in find_packages(where=package_name)],
)

