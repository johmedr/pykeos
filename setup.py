from setuptools import find_packages, setup

setup(
    name='pykeos',
    version='0.0.1',
    author='Johan Medrano',
    python_requires='>=3.4',
    author_email='',
    description='',
    platforms='Linux',
    packages=find_packages(where='./pykeos'),
    package_dir={
        '': 'pykeos'
    },
    include_package_data=True,
    install_requires=(
        'nolds',
        "plotly",
        'plotly_express',
        "pyunicorn",
        "plotly-orca"
    ),
    classifiers=[
        'Development Status :: 1 - Alpha',
        'Natural Language :: English',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)
