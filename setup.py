from setuptools import setup, find_packages

setup(
    name="tea_multiplex",
    version="0.1.0",
    description="Multi-emotional multiplex extension of TEA Networks using Plutchik emotions",
    author="Jacopo Schenetti",
    author_email="jschenetti@gmail.com",
    url="https://github.com/Jacoposchenetti/TEA-Multiplex-Network",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "scipy",
        "networkx",
        "nrclex",
        "spacy>=3.0",
        "pandas",
    ],
    extras_require={
        "tea": ["teanets @ git+https://github.com/MassimoStel/TEA_Networks.git"],
    },
    license="BSD-3-Clause",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
)
