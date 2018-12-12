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
# -*- coding: utf-8 -*-
"""
Camiba – Algorithms
===================

This submodule aims at providing a rich ensemble of algorithms, which are
implementend with easy to read code and not many optimizations for rapid
prototyping and adaption for own more specifically tailored applications.

ADMM
----
.. automodule:: camiba.algs.admm
    :members:

ESPRIT
-----
.. automodule:: camiba.algs.esprit
    :members:

IHT
---
.. automodule:: camiba.algs.iht
    :members:

IRLS
----
.. automodule:: camiba.algs.irls
    :members:

ISTA
----
.. automodule:: camiba.algs.ista
    :members:

FISTA
-----
.. automodule:: camiba.algs.fista
    :members:

LASSO
-----
.. automodule:: camiba.algs.lasso
    :members:

OMP
---
.. automodule:: camiba.algs.omp
    :members:
"""

from .admm import *
from .esprit import *
from .iht import *
from .irls import *
from .ista import *
from .fista import *
from .lasso import *
from .omp import *
