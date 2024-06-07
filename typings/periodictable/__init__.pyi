"""
This type stub file was generated by pyright.
"""

from __future__ import annotations

from . import core

"""
Extensible periodic table of elements

The periodictable package contains mass for the isotopes and density for the
elements. It calculates xray and neutron scattering information for
isotopes and elements. Composite values can be calculated from
chemical formula and density.

The table is extensible. See the user manual for details.

----

Disclaimer:

This data has been compiled from a variety of sources for the user's
convenience and does not represent a critical evaluation by the authors.
While we have made efforts to verify that the values we use match
published values, the values themselves are based on measurements
whose conditions may differ from those of your experiment.

----

"""
__docformat__ = ...
__all__ = [
    "elements",
    "neutron_sld",
    "xray_sld",
    "formula",
    "mix_by_weight",
    "mix_by_volume",
]
__version__ = ...
elements: core.PeriodicTable = ...

def data_files():  # -> list[Unknown]:
    """
    Return the data files associated with all periodic table attributes.

    The format is a list of (directory, [files...]) pairs which can be
    used directly in setup(..., data_files=...) for setup.py.

    """

__all__ += core.define_elements(elements, globals())

def formula(*args, **kw):
    """
    Chemical formula representation.

    Example initializers:

       string:
          m = formula( "CaCO3+6H2O" )
       sequence of fragments:
          m = formula( [(1, Ca), (2, C), (3, O), (6, [(2, H), (1, O)]] )
       molecular math:
          m = formula( "CaCO3" ) + 6*formula( "H2O" )
       another formula (makes a copy):
          m = formula( formula("CaCO3+6H2O") )
       an atom:
          m = formula( Ca )
       nothing:
          m = formula()

    Additional information can be provided:

       density (|g/cm^3|)   material density
       natural_density (|g/cm^3|) material density with natural abundance
       name (string) common name for the molecule
       table (PeriodicTable) periodic table with customized data

    Operations:
       m.atoms returns a dictionary of isotope: count for the
          entire molecule

    Formula strings consist of counts and atoms such as "CaCO3+6H2O".
    Groups can be separated by '+' or space, so "CaCO3 6H2O" works as well.
    Groups and be defined using parentheses, such as "CaCO3(H2O)6".
    Parentheses can nest: "(CaCO3(H2O)6)1"
    Isotopes are represented by index, e.g., "CaCO[18]3+6H2O".
    Counts can be integer or decimal, e.g. "CaCO3+(3HO0.5)2".

    For full details see help(periodictable.formulas.formula_grammar)

    The chemical formula is designed for simple calculations such
    as molar mass, not for representing bonds or atom positions.
    However, we preserve the structure of the formula so that it can
    be used as a basis for a rich text representation such as
    matplotlib TeX markup.
    """

def mix_by_weight(*args, **kw):  # -> Formula:
    """
    Generate a mixture which apportions each formula by weight.

    :Parameters:

        *formula1* : Formula OR string
            Material

        *quantity1* : float
            Relative quantity of that material

        *formula2* : Formula OR string
            Material

        *quantity2* : float
            Relative quantity of that material

        ...

        *density* : float
            Density of the mixture, if known

        *natural_density* : float
            Density of the mixture with natural abundances, if known.

        *name* : string
            Name of the mixture

    :Returns:

        *formula* : Formula

    If density is not given, then it will be computed from the density
    of the components, assuming equal volume.
    """

def mix_by_volume(*args, **kw):  # -> Formula:
    """
    Generate a mixture which apportions each formula by volume.

    :Parameters:

        *formula1* : Formula OR string
            Material

        *quantity1* : float
            Relative quantity of that material

        *formula2* : Formula OR string
            Material

        *quantity2* : float
            Relative quantity of that material

        ...

        *density* : float
            Density of the mixture, if known

        *natural_density* : float
            Density of the mixture with natural abundances, if known.

        *name* : string
            Name of the mixture

    :Returns:

        *formula* : Formula

    Densities are required for each of the components.  If the density of
    the result is not given, it will be computed from the components
    assuming the components take up no more nor less space because they
    are in the mixture.
    """

def neutron_sld(
    *args, **kw
):  # -> tuple[Literal[0], Literal[0], Literal[0]] | tuple[Unknown, Unknown, Unknown] | None:
    """
    Compute neutron scattering length densities for molecules.

    Returns scattering length density (real, imaginary and incoherent).

    See :class:`periodictable.nsf.neutron_sld` for details.
    """

def neutron_scattering(
    *args, **kw
):  # -> tuple[None, None, None] | tuple[tuple[Literal[0], Literal[0], Literal[0]], tuple[Literal[0], Literal[0], Literal[0]], float] | tuple[tuple[Unknown, Unknown, Unknown], tuple[Unknown, Unknown, Unknown], Unknown]:
    """
    Compute neutron scattering cross sections for molecules.

    Returns scattering length density (real, imaginary and incoherent),
    cross sections (coherent, absorption, incoherent) and penetration
    depth.

    See :func:`periodictable.nsf.neutron_scattering` for details.
    """

def xray_sld(
    *args, **kw
):  # -> tuple[Literal[0], Literal[0]] | tuple[Unknown, Unknown]:
    """
    Compute neutron scattering length densities for molecules.

    Either supply the wavelength (A) or the energy (keV) of the X-rays.

    Returns scattering length density (real, imaginary).

    See :class:`periodictable.xsf.Xray` for details.
    """
