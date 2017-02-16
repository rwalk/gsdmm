from setuptools import setup

VERSION=0.1
INSTALL_REQUIRES = [
    'numpy'
]

setup(
    name='gsdmm',
    packages=['gsdmm'],
    version=0.1,
    url='https://www.github.com/rwalk/gsdmm',
    author='Ryan Walker',
    author_email='ryan@ryanwalker.us',
    description='GSDMM: Short text clustering ',
    license='MIT',
    install_requires=INSTALL_REQUIRES
)
