# This file is part of Camiba.
#
# Camiba is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Camiba is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Camiba. If not, see <http://www.gnu.org/licenses/>.
# This file is part of Camiba.
#
# Camiba is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Camiba is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Camiba. If not, see <http://www.gnu.org/licenses/>.
from setuptools import setup, find_packages, Extension
from sphinx.setup_command import BuildDoc
from distutils import sysconfig
from Cython.Build import cythonize
import numpy

sysconfig.get_config_vars()['CFLAGS'] = ''
sysconfig.get_config_vars()['OPT'] = ''
sysconfig.get_config_vars()['PY_CFLAGS'] = ''
sysconfig.get_config_vars()['PY_CORE_CFLAGS'] = ''
sysconfig.get_config_vars()['CC'] = 'gcc'
sysconfig.get_config_vars()['CXX'] = 'gcc'
sysconfig.get_config_vars()['BASECFLAGS'] = ''
sysconfig.get_config_vars()['CCSHARED'] = '-fPIC -Ofast'
sysconfig.get_config_vars()['LDSHARED'] = 'gcc -shared'
sysconfig.get_config_vars()['CPP'] = ''
sysconfig.get_config_vars()['CPPFLAGS'] = ''
sysconfig.get_config_vars()['BLDSHARED'] = ''
sysconfig.get_config_vars()['CONFIGURE_LDFLAGS'] = ''
sysconfig.get_config_vars()['LDFLAGS'] = ''
sysconfig.get_config_vars()['PY_LDFLAGS'] = ''

lstIncludes = [numpy.get_include()]

ext_modules = [
    Extension(
        "*",
        ["camiba/linalg/*.pyx"],
        include_dirs=lstIncludes,
        extra_compile_args=['-march=native', '-march=native']
    ),
    Extension(
        "*",
        ["camiba/algs/*.pyx"],
        include_dirs=lstIncludes,
        extra_compile_args=['-march=native', '-march=native']
    )
]

setup(
    name='camiba',
    version='0.1.2',
    release='0.1.2',
    author='Sebastian Semper',
    author_email='sebastian.semper@tu-ilmenau.de',
    license='LGPL3',
    description='Compressed Sensing Algorithms for Python',
    url='https://github.com/SebastianSemper/camiba',
    packages=[
        'camiba',
        'camiba.algs',
        'camiba.cs',
        'camiba.linalg'
    ],
    cmdclass={'build_doc': BuildDoc},
    command_options={
        'build_doc': {
        }},
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License (GPL)',
        'Programming Language :: Python :: 3',
    ],
    project_urls={
        'Documentation':
            'https://sebastiansemper.github.io/camiba/',
        'Funding':
            'https://www.tu-ilmenau.de/it-ems/',
        'Say Thanks!':
            'https://www.tu-ilmenau.de/it-ems/',
        'Source':
            'https://github.com/sebastiansemper/camiba',
        'Tracker':
            'https://github.com/sebastiansemper/camiba/issues',
    },
    install_requires=['numpy', 'sphinx', 'scipy', 'numpydoc'],
    python_requires='>=3',
    ext_modules=cythonize(ext_modules)
)
