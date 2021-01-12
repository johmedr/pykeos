from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='pykeos',
    version='0.0.2dev5',
    author='Johan Medrano',
    python_requires='>=3.4',
    author_email='',
    description='',
    long_description=long_description,
    platforms='Linux',
    packages=find_packages(),
    #package_dir={
    #    '': 'pykeos'
    #},
    include_package_data=True,
    install_requires=(
        'nolds',
        "plotly",
        'plotly_express'
    ),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Natural Language :: English',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)
