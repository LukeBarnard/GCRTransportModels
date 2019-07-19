import numpy as np
from scipy import sparse
import scipy.constants as constants
import astropy.units as u


class GetProton:

    def __init__(self, kinetic_energy=1000 * u.MeV):
        """
        Initialise the proton.
        kinetic_energy should be a scalar or array of kinetic energies, using MeV units.
        # Beta and Rigidity values have been validated against onine calculators and tables
        http://www.geomagsphere.org/index.php/rigidity-energy-conversion
        https://ccmc.gsfc.nasa.gov/pub/modelweb/cosmic_rays/cutoff_rigidity_sw/rpt-6.doc
        """
        self.c = constants.c * (u.m / u.s)
        self.M = constants.proton_mass * u.kg
        self.q = constants.e * u.coulomb
        self.Em = (self.M * self.c ** 2).to('MeV')  # rest mass in MeV
        self.T = kinetic_energy  # Kinetic energy input - can be an array of kinetic energies
        self.Et = self.T + self.Em  # Total energy
        self.momentum = np.sqrt((self.Et ** 2) - (self.Em ** 2)) / self.c
        self.beta = np.sqrt(1 - (self.Em / self.Et) ** 2)
        self.rigidity = (np.sqrt((self.T ** 2) + (2 * self.T * self.Em)) / self.q).to('GV')  # eqn 1 from geomagsphere
        self.alpha = (self.T + 2 * self.Em) / (self.T + self.Em)
        self.dalpha_dT = -self.Em / ((self.T + self.Em) ** 2)
        return


class ModelDomain:

    def __init__(self):
        """
        Define the spatial and energy coordinates of the model.
        """

        # Define number of spatial and energy grid points
        self.nx = 500
        self.nt = 1000

        # Spatial domain, in km. Use GU71 limits of 1 sol Rad and 10 AU.
        # Use astropy units to convert between solar rad and AU.
        x_min = (1.0 * u.solRad).to('km').value
        x_max = (100.0 * u.AU).to('km').value

        # Setup spatial grid, including boundary points.
        self.xb, self.dx = np.linspace(x_min, x_max, self.nx, retstep=True)
        self.xb = self.xb * u.km
        self.dx = self.dx * u.km
        # Get also only the interior points for solving later
        self.x = self.xb[0:-1]

        # Energy domain in MeV, using GU71 substitution (eq.3.1 inverted.)
        # Go from 1MeV - 1e4MeV using variable dT step given by the GU71 substitution.
        proton = GetProton()  # Need proton rest mass for working out energy grid.
        t_min = 1e0 * u.MeV
        t_max = 1e6 * u.MeV
        s_min = np.arctan(np.sqrt(2 * t_min / proton.Em))
        s_max = np.arctan(np.sqrt(2 * t_max / proton.Em))
        self.s, self.ds = np.linspace(s_max.value, s_min.value, self.nt, retstep=True)
        self.T = (proton.Em / 2.0) * np.tan(self.s) ** 2
        self.dT = np.diff(self.T)

        # Store a proton instance inside the model, to compute the rigidities, betas, etc.
        self.proton = GetProton(kinetic_energy=self.T)

        # Setup array for the full solution, with proper unit.
        self.differential_intensity_unit = 1.0 / (u.m ** 2 * u.s * u.steradian * u.MeV)
        self.U = np.zeros((self.T.size, self.xb.size)) * self.differential_intensity_unit

        # Also compute the solar wind solution and the chi parameter (which needs the solar wind solution)
        self.v, self.dv_dr = solar_wind_profile(self.x)
        self.chi = compute_solar_wind_chi(self.x, self.v, self.dv_dr)
        return


class GCR1D_diffusion:

    def __init__(self):
        self.domain = ModelDomain()
        self.differential_intensity_unit = self.domain.differential_intensity_unit
        self.diffusion_type = "simple"

    def compute_a_coefficient(self, proton, dT):
        a = self.chi * proton.alpha * proton.T
        a = (a / dT)
        return a

    def compute_b_coefficient(self, proton):
        kappa, dkappa_dr = self.diffusion_coefficient(proton)
        b = -kappa / (2.0 * self.dx ** 2)
        return b

    def compute_c_coefficient(self, proton):
        kappa, dkappa_dr = self.diffusion_coefficient(proton)
        c = self.V - (2.0 * kappa / self.x) - dkappa_dr
        c = c / (4.0 * self.dx)
        return c

    def compute_d_coefficient(self, proton):
        d = self.chi - (self.chi / 3.0) * (proton.alpha + proton.T * proton.dalpha_dT)
        return d

    def compute_matrices(self, proton, dT, bv1, bv2):
        a = self.compute_a_coefficient(proton, dT)
        b = self.compute_b_coefficient(proton)
        c = self.compute_c_coefficient(proton)
        d = self.compute_d_coefficient(proton)

        # Matrix A
        main_diag = a + 2.0 * b
        # Update inner Neumann boundary for dUdx=0
        main_diag[0] = a[0] + b[0] + c[0]

        off_diag_up = -b - c
        off_diag_dn = -b + c
        diagonals = [main_diag, off_diag_dn, off_diag_up]
        # This command destroys the unit. manhandle it.
        F = sparse.diags(diagonals, [0, -1, 1], shape=(self.x.size, self.x.size)).toarray()
        F = F * main_diag.unit

        # Matrix B
        main_diag = a - 2.0 * b + d
        # Update inner Neumann boundary for dUdx=0
        main_diag[0] = a[0] - b[0] - c[0] + d[0]

        off_diag_up = b + c
        off_diag_dn = b - c
        diagonals = [main_diag, off_diag_dn, off_diag_up]
        # This command destroys the unit, manhandle it.
        G = sparse.diags(diagonals, [0, -1, 1], shape=(self.x.size, self.x.size)).toarray()
        G = G * main_diag.unit

        # Compute boundary condition vectors for outer Dirichlet boundary.
        BC1 = np.zeros(self.x.shape) * self.differential_intensity_unit * G.unit
        BC1[-1] = (b[-1] + c[-1]) * bv1

        BC2 = np.zeros(self.x.shape) * self.differential_intensity_unit * G.unit
        BC2[-1] = (b[-1] + c[-1]) * bv2
        return F, G, BC1, BC2

    def solve(self, boundary_type="gaussian", diffusion_type="simple"):
        # Update diffusion type, so diffusion_coefficient knows which type to call.
        self.diffusion_type = diffusion_type

        # Make sure the solution array is clear
        self.U = np.zeros(self.U.shape) * self.differential_intensity_unit

        # Add in the boundary profile
        self.U[:, -1] = self.boundary_profile(boundary_type=boundary_type)

        # Add in initial condition at high energy/low modulation.
        self.U[0, 1:-1] += self.U[0, -1]

        for j, dT in enumerate(self.dT):
            jn = j + 1
            # Get a proton at this energy for computing the diffusion coefficient and enrgy terms.
            proton = GetProton(kinetic_energy=self.T[j])
            # Get the Dirichlet boundary conditions
            d1 = self.U[j, -1]
            d2 = self.U[jn, -1]

            # Compute diffusion matrices
            F, G, BC1, BC2 = self.compute_matrices(proton, dT, d1, d2)
            # Solve for this energy
            # Numpy matmul and linalg seem to fuckup the units. Manhandle them.
            g = np.matmul(G.value, np.array(self.U[j, 1:-1])) + BC1.value + BC2.value
            self.U[jn, 1:-1] = np.linalg.solve(F.value, g) * self.U.unit

        return


def solar_wind_profile(x, vo=400):
    """
    Compute model solar wind as per GU71, with asymptotic wind speed of Vo=400km/s, and decay halflife of
    15 solar radii
    """
    vo = vo * u.km / u.s
    lambda_v = 15.0 * u.solRad
    ro = 1.0 * u.solRad
    arg = (ro.to('km') - x) / lambda_v.to('km')
    v = vo * (1.0 - np.exp(arg))
    dv_dr = (vo - v) / lambda_v.to('km')
    return v, dv_dr


def compute_solar_wind_chi(x, v, dv_dr):
    """
    Return the chi term from the transport equation, which depends only on the solar wind flow.
    """
    chi = (2.0 * v / x) + dv_dr
    return chi


def boundary_profile(model, boundary_type):
    """
    Function to compute the Dirichlet boundary condtion for the outer spatial boundary. Input argument boundary_type
    should string describing type of boundary.
    :param model: An instance of the ModelDomain class, which defines the energy coordinates.
    :param boundary_type: Valid arguments are "constant": A constant boundary with Uhp = 1.0 for all energies.
                                          "gaussian": A gaussian boundary centered at 280MeV, as used in Fig 1 of GU71.
                                          "webber": The local interstellar GCR spectrum of Webber and Lockwood 2001.
                                          "usoskin": The local interstellar GCR spectrum of Usoskin et al 2011.
    :return: profile: The requested boundary profile in standard differential intensity unit.
    """
    if boundary_type == "constant":
        profile = 1.0 * np.ones(model.T.shape) * model.differential_intensity_unit

    elif boundary_type == "gaussian":
        arg = -280.0 * ((model.T / (280.0 * u.MeV)) - 1.0) ** 2
        profile = 1.0 * np.exp(arg) * model.differential_intensity_unit

    elif boundary_type == "webber":
        lis = webber_lis(model)
        profile = lis.to(model.differential_intensity_unit)

    elif boundary_type == "usoskin":
        lis = usoskin_lis(model)
        profile = lis.to(model.differential_intensity_unit)
    else:
        print "Error, invalid boundary profile: {}".format(boundary_type['form'])

    return profile


def webber_lis(model):
    """
    Function to compute the Webber and Lockwood 2001 LIS, J = AT^i / (1 + BT^j + CT^k), with T in GeV and
    LIS in 1/(m^2 s sr MeV)
    :param model: An instance of the ModelDomain class, which defines the energy coordinates.
    :return: lis: The local interstellar spectrum at energies defined in model.
    """
    t = model.T.to('GeV').value
    a = 21.1
    i = -2.8
    b = 5.585
    j = -1.22
    c = 1.18
    k = -2.54
    lis = (a * t ** i) / (1.0 + (b * t ** j) + (c * t ** k))
    unit = 1 / (u.m ** 2 * u.s * u.steradian * u.MeV)
    lis = lis * unit
    return lis.to(model.differential_intensity_unit)


def usoskin_lis(model):
    """
    Function to compute the Usoskin et al 2011 LIS. Equation 2 of DOI:10.1029/2010JA016105
    Article states that this is the Burger 2000 LIS, but Burger 2000 provides a different functional form.
    Note, this form only works for protons.
    :param model: An instance of the ModelDomain class, which defines the energy coordinates.
    :return: lis: The local interstellar spectrum at energies defined in model.
    """
    t = model.T.to('GeV').value
    tr = model.proton.Em.to('GeV').value
    a = 1.9e4
    i = -2.78
    b = 0.4866
    j = -2.51
    p = np.sqrt(t * (t + 2 * tr))
    lis = (a * p ** i) / (1.0 + b * p ** j)
    unit = 1 / (u.m ** 2 * u.s * u.steradian * u.GeV)
    lis = lis * unit
    return lis.to(model.differential_intensity_unit)


def diffusion_simple(model, ao=4.38e22):
    """
    Return a simple constant diffusion coefficient, doesn't depend on space or energy
    :param model: An instance of ModelDomain, which defines the spatial and energy coordinates.
    :param ao:
    :return:
    """
    a = ao * u.cm ** 2 / u.s
    kappa = a.to(u.km ** 2 / u.s) * np.ones(model.x.shape)
    dkappa_dr = 0.0 * np.ones(model.x.shape) * (u.km / u.s)
    return kappa, dkappa_dr


def diffusion_cm04(model, proton, ao=4.38e22):
    """
    Compute diffusion coefficeint as per CM04.
    In this instance kappa is independent of r, so dkappa_dr = 0.
    :param model: An instance of ModelDomain, which defines the spatial and energy coordinates.
    :param proton: An instance of GetProton, for the single energy value of interest.
    :param ao: Diffusion scaling coefficient (float)
    :return:
    """
    a = ao * u.cm ** 2 / (u.s * u.GV)
    kappa = (a * proton.beta * proton.rigidity).to(u.km ** 2 / u.s) * np.ones(model.x.shape)
    dkappa_dr = 0.0 * np.ones(model.x.shape) * (u.km / u.s)
    return kappa, dkappa_dr


def diffusion_gu71(model, proton, ao=1.9e21):
    """
    Compute diffusion coefficeint as per GU71.
    :param model: An instance of ModelDomain, which defines the spatial and energy coordinates.
    :param proton: An instance of GetProton, for the single energy value of interest.
    :param ao: Diffusion scaling coefficient (float)
    :return:
    """
    a = ao * u.cm ** 2 / (u.s * u.GV * u.AU)
    kappa = (a * model.x.to('AU') * proton.beta * proton.rigidity).to(u.km ** 2 / u.s)
    dkappa_dr = (a * proton.beta * proton.rigidity).to(u.km / u.s) * np.ones(model.x.shape)
    return kappa, dkappa_dr


def diffusion_coefficient(gcr, proton):
    """
    Return the diffusion coefficient and it's spatial gradient corresponding to the diffusion type specified in
    self.
    :param gcr: An instance of GCR1D_diffusion
    :param proton: An instance of GetProton for the single energy value of interest
    :return:
    """
    if gcr.diffusion_type == "simple":
        kappa, dkappa_dr = diffusion_simple()
    elif gcr.diffusion_type == "cm04":
        kappa, dkappa_dr = diffusion_cm04(proton)
    elif gcr.diffusion_type == "gu71":
        kappa, dkappa_dr = diffusion_gu71(proton)

    # Units should be OK, but force a conversion to be sure.
    kappa_unit = u.km ** 2 / u.s
    dkappa_dr_unit = u.km / u.s
    return kappa.to(kappa_unit), dkappa_dr.to(dkappa_dr_unit)
