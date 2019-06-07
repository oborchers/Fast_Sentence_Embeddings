#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Run with:

sudo python ./setup.py install
'''

import os
import platform
import sys
import warnings
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext


class custom_build_ext(build_ext):
    '''Allow C extension building to fail.
    '''
    warning_message = '''
********************************************************************
WARNING: %s could not be compiled. %s

Here are some hints for popular operating systems:

If you are seeing this message on Linux you probably need to
install GCC and/or the Python development package for your
version of Python.

Debian and Ubuntu users should issue the following command:

    $ sudo apt-get install build-essential python-dev

RedHat, CentOS, and Fedora users should issue the following command:

    $ sudo yum install gcc python-devel

If you are seeing this message on OSX please read the documentation
here:

http://api.mongodb.org/python/current/installation.html#osx
********************************************************************
'''

    def run(self):
        try:
            build_ext.run(self)
        except Exception:
            e = sys.exc_info()[1]
            sys.stdout.write('%s\n' % str(e))
            warnings.warn(
                self.warning_message +
                'Extension modules' +
                'There was an issue with your platform configuration - see above.')

    def build_extension(self, ext):
        name = ext.name
        try:
            build_ext.build_extension(self, ext)
        except Exception:
            e = sys.exc_info()[1]
            sys.stdout.write('%s\n' % str(e))
            warnings.warn(
                self.warning_message +
                'The %s extension module' % (name,) +
                'The output above this warning shows how the compilation failed.')

    def finalize_options(self):
        build_ext.finalize_options(self)
        if isinstance(__builtins__, dict):
            __builtins__['__NUMPY_SETUP__'] = False
        else:
            __builtins__.__NUMPY_SETUP__ = False

        import numpy
        self.include_dirs.append(numpy.get_include())

mod_dir = os.path.join(os.path.dirname(__file__), 'fse', 'models')
dev_dir = os.path.join(os.path.dirname(__file__), 'fse', 'exp')
fse_dir = os.path.join(os.path.dirname(__file__), 'fse')

cmdclass = {'build_ext': custom_build_ext}

setup(
    name='fse',
    version='0.0.1',
    description='Fast Sentence Embeddings for Gensim',

    author=u'Oliver Borchers',
    author_email='borchers@bwl.uni-mannheim.de',

    url="https://github.com/oborchers/Fast_Sentence_Embeddings",

    ext_modules=[
        Extension('fse.models.sentence2vec_inner',
                sources=['./fse/models/sentence2vec_inner.pyx'],
                include_dirs=[mod_dir]),
        Extension('fse.exp.sif_variants_cy',
                sources=['./fse/exp/sif_variants_cy.pyx'],
                include_dirs=[dev_dir]),
        ],
        
    cmdclass=cmdclass,
    packages=find_packages(),

    zip_safe=False,

    install_requires=[
        'numpy >= 1.11.3',
        'scipy >= 0.18.1',
        'six >= 1.5.0',
        'smart_open >= 1.5.0',
        'scikit-learn >= 0.19.1',
        'gensim >= 3.4.0',
        'wordfreq >= 2.2.1',
    ],

    include_package_data=True,
)
