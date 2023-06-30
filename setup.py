from setuptools import setup, find_packages
from distutils.util import convert_path

main_ns = {}
ver_path = convert_path('quick_gmx_edr_check/version.py')
with open(ver_path) as ver_file:
    exec(ver_file.read(), main_ns)
    
setup(
    name='quick_gmx_edr_check',
    version=main_ns['__version__'],
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'fuzzywuzzy',
    ],
)
