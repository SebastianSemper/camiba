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
from setuptools import setup, find_packages
from sphinx.setup_command import BuildDoc

setup(
    name='camiba',
    version='0.1',
    release='0.1.0',
    author='Sebastian Semper',
    author_email='sebastian.semper@tu-ilmenau.de',
    license='LGPL3',
    install_requires=['sphinx', 'numpydoc'],
    description='Compressed Sensing Algorithms for Python',
    url='https://github.com/SebastianSemper/camiba',
    packages=['camiba'],
    cmdclass={'build_doc': BuildDoc},
    # these are optional and override conf.py settings
    command_options={
        'build_doc': {
        }}
)
