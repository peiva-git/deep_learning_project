from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

with open('requirements.txt') as f:
    required_packages = f.read()

setup(
    name='dlproject',
    version='0.1dev',
    description='Exam project for the Deep Learning course @ UniTS',
    long_description=readme,
    author='Federico Calandra, Ivan Pelizon',
    license=license,
    packages=find_packages(exclude=('tests', 'docs', 'output', 'assets')),
    python_requires='>=3.8,<=3.11',
    install_requires=required_packages
)
