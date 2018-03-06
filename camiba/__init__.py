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
r"""Camiba – Algorithms Coming to Life

We aim at providing a collection of non tested but heavily used algorithms,
which revolve around compressed sensing and sparse recovery in the widest
sense.

Camiba provides methods for:
 * sparse recovery
 * sensing matrix design
 * performance metric estimation
 * an abstract wrapper to describe CS scenarios
 * methods for sparsity order estimation

Contents
--------
 * :py:mod:`camiba.data`

Submodules
----------

Here we list the packages submodules for easy referencing and access.

 * :py:mod:`camiba.algs`
 * :py:mod:`camiba.cs`
 * :py:mod:`camiba.linalg`
"""

from .algs.admm import *
from .algs.iht import *
from .algs.irls import *
from .algs.ista import *
from .algs.omp import *

from .cs import *
from .data import *
