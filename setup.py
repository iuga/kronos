from setuptools import setup, find_packages

setup(
    name="kronos",
    version="0.1",
    description="Deep Learning Helpers and Utilities",
    author="Iuga77",
    packages=find_packages(exclude=["LICENSE", ".github", ".gitignore"]),
    install_requires=[
        'Pillow==3.2.0',
        'numpy==1.11.1',
    ]
)
