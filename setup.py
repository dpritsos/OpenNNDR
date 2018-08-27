import setuptools
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

setuptools.setup(
    name="OpenNNDR",
    version="0.0.1",
    author="Dimitrios Pritsos",
    author_email="dpritsos@extremepro.gr",
    description="",
    long_description="",
    long_description_content_type="",
    url="",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 2",
        "License :: OSI Approved :: GPL License",
        "Operating System :: OS Independent",
    ],
)

ext_modules = [
    Extension(
        "OpenNNDR/dsmeasures/dsmeasures",
        ["OpenNNDR/dsmeasures/dsmeasures.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    )
]

setup(
    setup_requires=[
        'setuptools>=18.0',
        'cython>=0.19.1',
    ],
    name='cy',
    ext_modules=cythonize(ext_modules),
)
