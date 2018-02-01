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
