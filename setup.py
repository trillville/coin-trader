#!/usr/bin/env python

from setuptools import setup, find_packages

install_requires = [
    'gdax==1.0.6',
    'tensorforce==0.3.5.1',
    'pandas==0.22.0'
]

setup(
    name='trader',
    version='0.0.1',
    author='Tillman Elser, Stedman Hood',
    author_email='tillman.elser@gmail.com, stedmanhood@gmail.com',
    license='MIT',
    url='https://github.com/trillville/coin-trader',
    packages=find_packages(),
    install_requires=install_requires,
    description='GDAX trading bot'
)
