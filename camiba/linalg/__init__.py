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
"""
Camiba – Linear Algebra
=======================

This module provides some routines, which are mostly related
to concepts from linear algebra, since they produce or manipulate
matrices.

Basic
-----
.. automodule:: camiba.linalg.basic
    :members:

Packings
--------
.. automodule:: camiba.linalg.pack
    :members:

Vandermonde Matrices
--------------------
.. automodule:: camiba.linalg.vand
    :members:

Multilevel Matrices
-------------------
.. automodule:: camiba.linalg.multilevel
    :members:
"""

from .basic import *
from .pack import *
from .vand import *
