"""
This type stub file was generated by pyright.
"""

"""
Crystal structure data.

Adds *crystal_structure* to the periodic table.  Each crystal structure
is a dictionary which contains the key 'symmetry'.  Depending on the
value of crystal_structure['symmetry'], one or more parameters
'a', 'c/a', 'b/a', 'd', and 'alpha' may be present according to
the following table:

.. table:: Crystal lattice parameters

    ============ ===========
    Symmetry     Parameters
    ============ ===========
    atom
    diatom       d
    BCC          a
    fcc          a
    hcp          c/a, a
    Tetragonal   c/a, a
    Cubic        a
    Diamond      a
    Orthorhombic c/a, a, b/a
    Rhombohedral a, alpha
    SC           a
    Monoclinic
    ============ ===========

Example:

.. doctest::

    >>> import periodictable as elements
    >>> print(elements.C.crystal_structure['symmetry'])
    Diamond
    >>> print(elements.C.crystal_structure['a'])
    3.57

This data is from Ashcroft and Mermin.
"""
from __future__ import annotations

crystal_structures = ...

def init(table, reload=...):  # -> None:
    """
    Add crystal_structure field to the element properties.
    """
    ...