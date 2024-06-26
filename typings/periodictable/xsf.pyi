"""
This type stub file was generated by pyright.
"""

from __future__ import annotations

from .util import require_keywords

r"""
This module has one class and nine fuctions.

   :class:`Xray`
       X-ray scattering properties for the elements.

The following attributes are added to each element:

   :func:`Xray.sftable`
       Three column table of energy vs. scattering factors f1, f2.

   :func:`Xray.scattering_factors`
       Returns f1, f2, the X-ray scattering factors for the given wavelengths
       interpolated from sftable.

   :func:`Xray.f0`
       Returns f0 for the given vector Q, with Q[i] in $[0, 24\pi]$ |1/Ang|.

   :func:`Xray.sld`
       Returns scattering length density (*real*, *imaginary*) for the
       given wavelengths or energies.

The following functions are available for X-ray scatting information processing:

     :func:`xray_wavelength`
         Finds X-ray wavelength in angstroms given energy in keV.

     :func:`xray_energy`
         Finds X-ray energy in keV given wavelength in angstroms.

     :func:`init`
         Initializes a periodic table with the Lawrence Berkeley Laboratory
         Center for X-Ray Optics xray scattering factors.

     :func:`init_spectral_lines`
         Sets the K_alpha and K_beta1 wavelengths for select elements.

     :func:`sld_table`
         Prints the xray SLD table for the given wavelength.

     :func:`xray_sld`
         Computes xray scattering length densities for molecules.

     :func:`index_of_refraction`
         Express xray scattering length density as an index of refraction

     :func:`mirror_reflectivity`
         X-ray reflectivity from a mirror made of a single compound.

     :func:`xray_sld_from_atoms`
         The underlying scattering length density calculator. This works with
         a dictionary of atoms and quantities directly.

     :func:`emission_table`
         Prints a table of emission lines.

K_alpha, K_beta1 (|Ang|):
    X-ray emission lines for elements beyond neon, with
    $K_\alpha = (2 K_{\alpha 1} + K_{\alpha 2})/3$.

X-ray scattering factors:
    Low-Energy X-ray Interaction Coefficients: Photoabsorption, scattering
    and reflection for E in 30 to 30,000 eV, and Z in 1 to 92.

.. Note::

    For :ref:`custom tables <custom-table>`, use :func:`init` and
    :func:`init_spectral_lines` to set the data.


Emission line tables
====================

Data for the $K_\alpha$ and $K_\beta$ lines comes from
[#Deslattes2003], with the full tables available at
`<http://www.nist.gov/pml/data/xraytrans/index.cfm>`_.
Experimental Values are used, truncated to 4 digits
of precision to correspond to the values for the subset
of elements previously defined in the periodictable package.

X-ray f1 and f2 tables
======================
The data for the tables is stored in the ``periodictable/xsf``.
directory.  The following information is from ``periodictable/xsf/read.me``,
with minor formatting changes:

  These ``[*.nff]`` files were used to generate the tables published in
  reference [#Henke1993]_. The files contain three columns of data:

    Energy(eV), *f_1*, *f_2*,

  where *f_1* and *f_2* are the atomic (forward) scattering factors.
  There are 500+ points on a uniform logarithmic mesh with points
  added 0.1 eV above and below "sharp" absorption edges. The
  tabulated values of *f_1* contain a relativistic, energy
  independent, correction given by:

  .. math::

    Z^* = Z - (Z/82.5)^{2.37}

  .. Note::
    Below 29 eV *f_1* is set equal to -9999.

  The atomic photoabsorption cross section, $\mu_a$, may be readily
  obtained from the values of $f_2$ using the relation:

  .. math::

    \mu_a = 2 r_e \lambda f_2

  where $r_e$ is the classical electron radius, and $\lambda$ is
  the wavelength. The index of refraction for a material with *N* atoms per
  unit volume is calculated by:

  .. math::

    n = 1 - N r_e \lambda^2 (f_1 + i f_2)/(2 \pi).

  These (semi-empirical) atomic scattering factors are based upon
  photoabsorption measurements of elements in their elemental state.
  The basic assumption is that condensed matter may be modeled as a
  collection of non-interacting atoms.  This assumption is in general
  a good one for energies sufficiently far from absorption thresholds.
  In the threshold regions, the specific chemical state is important
  and direct experimental measurements must be made.

  These tables are based on a compilation of the available experimental
  measurements and theoretical calculations.  For many elements there is
  little or no published data and in such cases it was necessary to
  rely on theoretical calculations and interpolations across Z.
  In order to improve the accuracy in the future considerably more
  experimental measurements are needed.

  Note that the following elements have been updated since the
  publication of Ref. [#Henke1993]_ in July 1993. Check
  `<http://henke.lbl.gov/optical_constants/update.html>`_ for more
  recent updates.

  .. table::

           ========  ==========   =================
           Element   Updated      Energy Range (eV)
           ========  ==========   =================
           Mg        Jan 2011     10-1300
           Zr        Apr 2010     20-1000
           La        Jun 2007     14-440
           Gd        Jun 2007     12-450
           Sc        Apr 2006     50-1300
           Ti        Aug 2004     20-150
           Ru        Aug 2004     40-1300
           W         Aug 2004     35-250
           Mo        Aug 2004     25-60
           Be        Aug 2004     40-250
           Mo        Nov 1997     10-930
           Fe        Oct 1995     600-800
           Si        Jun 1995     30-500
           Au        Jul 1994     2000-6500
           Mg,Al,Si  Jan 1994     30-200
           Li        Nov 1994     2000-30000
           ========  ==========   =================


  Data available at:
    #. http://henke.lbl.gov/optical_constants/asf.html

.. [#Henke1993] B. L. Henke, E. M. Gullikson, and J. C. Davis.  "X-ray interactions:
       photoabsorption, scattering, transmission, and reflection at E=50-30000 eV,
       Z=1-92", Atomic Data and Nuclear Data Tables 54 no.2, 181-342 (July 1993).

.. [#Deslattes2003] R. D. Deslattes, E. G. Kessler, Jr., P. Indelicato, L. de Billy,
       E. Lindroth, and J. Anton.  Rev. Mod. Phys. 75, 35-99 (2003).

"""
__all__ = [
    "Xray",
    "init",
    "init_spectral_lines",
    "xray_energy",
    "xray_wavelength",
    "xray_sld",
    "xray_sld_from_atoms",
    "emission_table",
    "sld_table",
    "plot_xsf",
    "index_of_refraction",
    "mirror_reflectivity",
]

def xray_wavelength(energy):  # -> NDArray[floating[Any]]:
    r"""
    Convert X-ray energy to wavelength.

    :Parameters:
        *energy* : float or vector | keV

    :Returns:
        *wavelength* : float | |Ang|

    Energy can be converted to wavelength using

    .. math::

        \lambda = h c / E

    where:

        $h$ = planck's constant in eV\ |cdot|\ s

        $c$ = speed of light in m/s
    """

def xray_energy(wavelength):  # -> NDArray[floating[Any]]:
    r"""
    Convert X-ray wavelength to energy.

    :Parameters:
        *wavelength* : float or vector | |Ang|

    :Returns:
        *energy* : float or vector | keV

    Wavelength can be converted to energy using

    .. math::

        E = h c / \lambda

    where:

        $h$ = planck's constant in eV\ |cdot|\ s

        $c$ = speed of light in m/s
    """

class Xray:
    """
    X-ray scattering properties for the elements. Refer help(periodictable.xsf)
    from command prompt for details.
    """

    _nff_path = ...
    sftable_units = ...
    scattering_factors_units = ...
    sld_units = ...
    _table = ...
    def __init__(self, element) -> None: ...

    sftable = ...
    @require_keywords
    def scattering_factors(
        self, energy=..., wavelength=...
    ):  # -> tuple[None, None] | tuple[Unknown, Unknown]:
        """
        X-ray scattering factors f', f''.

        :Parameters:
            *energy* : float or vector | keV
                X-ray energy.

        :Returns:
            *scattering_factors* : (float, float)
                Values outside the range return NaN.

        Values are found from linear interpolation within the Henke Xray
        scattering factors database at the Lawrence Berkeley Laboratory
        Center for X-ray Optics.
        """
    def f0(self, Q):
        r"""
        Isotropic X-ray scattering factors *f0* for the input Q.

        :Parameters:
            *Q* : float or vector in $[0, 24\pi]$ | |1/Ang|
                X-ray scattering properties for the elements.

        :Returns:
            *f0* : float
                Values outside the valid range return NaN.


        .. Note::

            *f0* is often given as a function of $\sin(\theta)/\lambda$
            whereas we are using  $Q = 4 \pi \sin(\theta)/\lambda$, or
            in terms of energy $Q = 4 \pi \sin(\theta) E/(h c)$.

        Reference:
             D. Wassmaier, A. Kerfel, Acta Crystallogr. A51 (1995) 416.
             http://dx.doi.org/10.1107/S0108767394013292
        """
    @require_keywords
    def sld(
        self, wavelength=..., energy=...
    ):  # -> tuple[None, None] | tuple[Unknown, Unknown]:
        r"""
        X ray scattering length density.

        :Parameters:
            *wavelength* : float or vector | |Ang|
                Wavelength of the X-ray.

            *energy* : float or vector | keV
                Energy of the X-ray (if *wavelength* not specified).

            .. note:
                Only one of *wavelength* and *energy* is needed.

        :Returns:
            *sld* : (float, float) | |1/Ang^2|
                (*real*, *imaginary*) X-ray scattering length density.

        :Raises:
            *TypeError* : neither *wavelength* nor *energy* was specified.

        The scattering length density is $r_e N (f_1 + i f_2)$.
        where $r_e$ is the electron radius and $N$ is the
        number density.  The number density is $N = \rho_m/m N_A$,
        with mass density $\rho_m$ molar mass $m$ and
        Avogadro's number $N_A$.

        The constants are available directly:

            $r_e$ = periodictable.xsf.electron_radius

            $N_A$ = periodictable.constants.avogadro_number

        Data comes from the Henke Xray scattering factors database at the
        Lawrence Berkeley Laboratory Center for X-ray Optics.
        """

@require_keywords
def xray_sld(
    compound, density=..., natural_density=..., wavelength=..., energy=...
):  # -> tuple[Literal[0], Literal[0]] | tuple[Unknown, Unknown]:
    """
    Compute xray scattering length densities for molecules.

    :Parameters:
        *compound* : Formula initializer
            Chemical formula.
        *density* : float | |g/cm^3|
            Mass density of the compound, or None for default.
        *natural_density* : float | |g/cm^3|
            Mass density of the compound at naturally occurring isotope abundance.
        *wavelength* : float or vector | |Ang|
            Wavelength of the X-ray.
        *energy* : float or vector | keV
            Energy of the X-ray, if *wavelength* is not specified.

    :Returns:
        *sld* : (float, float) | |1e-6/Ang^2|
            (*real*, *imaginary*) scattering length density.

    :Raises:
        *AssertionError* :  *density* or *wavelength*/*energy* is missing.
    """

@require_keywords
def index_of_refraction(
    compound, density=..., natural_density=..., energy=..., wavelength=...
):  # -> NDArray[complexfloating[Any, Any]]:
    """
    Calculates the index of refraction for a given compound

    :Parameters:
        *compound* : Formula initializer
            Chemical formula.
        *density* : float | |g/cm^3|
            Mass density of the compound, or None for default.
        *natural_density* : float | |g/cm^3|
            Mass density of the compound at naturally occurring isotope abundance.
        *wavelength* : float or vector | |Ang|
            Wavelength of the X-ray.
        *energy* : float or vector | keV
            Energy of the X-ray, if *wavelength* is not specified.

    :Returns:
        *n* : float or vector | unitless
            index of refraction of the material at the given energy

    :Notes:

    Formula taken from http://xdb.lbl.gov (section 1.7) and checked
    against http://henke.lbl.gov/optical_constants/getdb2.html
    """

@require_keywords
def mirror_reflectivity(
    compound,
    density=...,
    natural_density=...,
    energy=...,
    wavelength=...,
    angle=...,
    roughness=...,
):  # -> NDArray[floating[Any]]:
    """
    Calculates reflectivity of a thick mirror as function of energy and angle

    :Parameters:
        *compound* : Formula initializer
            Chemical formula.
        *density* : float | |g/cm^3|
            Mass density of the compound, or None for default.
        *natural_density* : float | |g/cm^3|
            Mass density of the compound at naturally occurring isotope abundance.
        *roughness* : float | |Ang|
            High-spatial-frequency surface roughness.
        *wavelength* : float or vector | |Ang|
            Wavelength of the X-ray.
        *energy* : float or vector | keV
            Energy of the X-ray, if *wavelength* is not specified.
        *angle* : vector | |deg|
            Incident beam angles.

    :Returns:
        *reflectivity* : matrix
            matrix of reflectivity as function of (angle, energy)

    :Notes:

    Formula taken from http://xdb.lbl.gov (section 4.2) and checked
    against http://henke.lbl.gov/optical_constants/mirror2.html
    """

def xray_sld_from_atoms(
    *args, **kw
):  # -> tuple[Literal[0], Literal[0]] | tuple[Unknown, Unknown]:
    """
    .. deprecated:: 0.91

        :func:`xray_sld` now accepts a dictionary of *{atom: count}* directly.
    """

spectral_lines_data = ...

def init_spectral_lines(table):  # -> None:
    """
    Sets the K_alpha and K_beta1 wavelengths for select elements
    """

def init(table, reload=...): ...
def plot_xsf(el):  # -> None:
    """
    Plots the xray scattering factors for the given element.

    :Parameters:
        *el* : Element

    :Returns: None
    """

def sld_table(wavelength=..., table=...):  # -> None:
    """
    Prints the xray SLD table for the given wavelength.

    :Parameters:
        *wavelength* = Cu K-alpha : float | |Ang|
            X-ray wavelength.
        *table* : PeriodicTable
            The default periodictable unless a specific table has been requested.

    :Returns: None

    Example

        >>> sld_table()  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        X-ray scattering length density for 1.5418 Ang
         El    rho   irho
          H   1.19   0.00
         He   1.03   0.00
         Li   3.92   0.00
         Be  13.93   0.01
          B  18.40   0.01
          C  18.71   0.03
          N   6.88   0.02
          O   9.74   0.04
          F  12.16   0.07
         Ne  10.26   0.09
         Na   7.98   0.09
         Mg  14.78   0.22
          ...
    """

def emission_table(table=...):  # -> None:
    """
    Prints a table of emission lines.

    :Parameters:
        *table* : PeriodicTable
            The default periodictable unless a specific table has been requested.

    :Returns: None

    Example

        >>> emission_table()  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
         El  Kalpha  Kbeta1
         Ne 14.6102 14.4522
         Na 11.9103 11.5752
         Mg  9.8902  9.5211
         Al  8.3402  7.9601
         Si  7.1263  6.7531
         ...
    """
