from setuptools import setup, find_packages
from sitelle import version

packages = find_packages(where=".")
setup(
    name="sitelle",
    packages=packages,
    url='https://gitlab.ethz.ch/blaunet/sitelle',
    author='Barthelemy Launet',
    author_email='barthelemy.launet@obspm.fr',
    maintainer='Barthelemy Launet',
    maintainer_email='barthelemy.launet@obspm.fr',
    version=version.__version__,
    description='Helper functions to work with M31 observations on SITELLE',
    include_package_data=False,
    platforms='any',
)
