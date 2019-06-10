import numpy as np
from scipy import sparse
import scipy.constants as constants
import astropy.units as u


class GetProton:
    
    def __init__(self, kinetic_energy=1000*u.MeV):
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
        self.Em = (self.M * self.c**2).to('MeV')  # rest mass in MeV
        self.T = kinetic_energy  # Kinetic energy input - can be an array of kinetic energies
        self.Et = self.T + self.Em  # Total energy
        self.momentum = np.sqrt((self.Et**2)-(self.Em**2)) / self.c
        self.beta = np.sqrt(1 - (self.Em/self.Et)**2)
        self.rigidity = (np.sqrt((self.T**2) + (2 * self.T * self.Em)) / self.q).to('GV')  # eqn 1 from geomagsphere
        self.alpha = (self.T + 2*self.Em) / (self.T + self.Em)
        self.dalpha_dT = -self.Em / ((self.T + self.Em)**2)
        return


class GCR1DCrankNicolson:

    def __init__(self, spatial_res='medium', energy_res='medium'):
        # Create unit for differential intensity flux to be used throughout model.
        self.differential_intensity_unit = 1/(u.m**2 * u.s * u.steradian * u.MeV)
        self.setup_model_domain(spatial_res, energy_res)

    def setup_model_domain(self, spatial_res, energy_res):
        """
        Define the spatial and energy coordinates of the model.
        """

        if spatial_res == "low":
            Nx = 200
        elif spatial_res == "medium":
            Nx = 500
        elif spatial_res == "high":
            Nx = 1000

        if energy_res == "low":
            Ns = 200
        elif energy_res == "medium":
            Ns = 500
        elif energy_res == "high":
            Ns = 1000

        # Spatial domain, in km. Use GU71 limits of 1 sol Rad and 10 AU.
        # Use astropy units to convert between solar rad and AU.
        xmin = (1.0 * u.solRad).to('km').value
        xmax = (100.0 * u.AU).to('km').value 

        # Setup spatial grid, including boundary points.
        self.xb, self.dx = np.linspace(xmin, xmax, Nx, retstep=True)
        self.xb = self.xb * u.km
        self.dx = self.dx * u.km
        # Get also only the interior points for solving later
        self.x = self.xb[1:-1]
        
        # Energy domain in MeV, using GU71 substitution (eq.3.1 inverted.)
        # Go from 1MeV - 1e4MeV using variable dT step given by the GU71 substitution.
        proton = GetProton() # Need proton rest mass for working out energy grid.
        Tmin = 1e0 * u.MeV
        Tmax = 1e5 * u.MeV
        smin = np.arctan(np.sqrt(2*Tmin/proton.Em))
        smax = np.arctan(np.sqrt(2*Tmax/proton.Em))
        self.s, self.ds = np.linspace(smax.value, smin.value, Ns, retstep=True)
        self.T = (proton.Em / 2.0) * np.tan(self.s)**2 
        self.dT = np.diff(self.T)

        # Store a proton instance inside the model, to compute the rigidities, betas, etc.
        self.proton = GetProton(kinetic_energy=self.T)
        
        # Setup array for the full solution. 
        self.U = np.zeros((self.T.size, self.xb.size))*self.differential_intensity_unit
        
        # Also compute the solar wind solution and the chi parameter (which needs the solar wind solution)
        self.solar_wind_profile()
        self.compute_chi()
        return

    def boundary_profile(self, boundary_type):
        """
        Function to compute the Dirichlet boundary condtion for the outer spatial boundary. Input argument boundary_type
        should string describing type of boundary. Valid terms are constant, gaussian, WEBBER, and USOSKIN. 
        constant = a constant boundary with Uhp = 1.0 for all energies.
        gaussian = a guassian boundary centered at 280MeV, as used in Fig 1 of GU71.
        webber = The local interstellar GCR spectrum of Webber and Lockwood 2001.
        usoskin = The local interstellar GCR spectrum of Usoskin et al 2011.
        """
        if boundary_type == "constant":
            profile = 1.0 * np.ones(self.T.shape) * self.differential_intensity_unit
        
        elif boundary_type == "gaussian":
            arg = -280.0 * ((self.T / (280.0 * u.MeV)) - 1.0)**2
            profile = 1.0 * np.exp(arg) * self.differential_intensity_unit

        elif boundary_type == "webber":
            LIS = self.webber_lis()
            profile = LIS.to(self.differential_intensity_unit)
        
        elif boundary_type =="usoskin":
            LIS = self.usoskin_lis()
            profile = LIS.to(self.differential_intensity_unit)
        else:
            print "Error, invalid boundary profile: {}".format(boundary_type['form'])
            
        return profile
    
    def webber_lis(self):
        """
        Function to compute the Webber and Lockwood 2001 LIS
        J = AT^i / (1 + BT^j + CT^k), with T in GeV and 
        LIS in 1/(m^2 s sr MeV)
        """
        T = self.T.to('GeV').value
        A = 21.1
        i = -2.8
        B = 5.585
        j = -1.22
        C = 1.18
        k = -2.54
        LIS = A * T**i / (1.0 + B * T**j + C * T**k)
        unit = 1 / (u.m**2 * u.s * u.steradian * u.MeV)
        LIS = LIS*unit
        return LIS.to(self.differential_intensity_unit)

    def usoskin_lis(self):
        """
        Function to compute the Usoskin et al 2011 LIS. Equation 2 of DOI:10.1029/2010JA016105
        Article states that this is the Burger 2000 LIS, but Burger 2000 provides a different functional form.
        Note, this form only works for protons. 
        T = Kinetic energy (GeV)
        Tr = Rest mass energy (GeV)
        LIS in 1/(m^2 s sr GeV), but convert to 1/(cm^2 s sr MeV), as used throughout here
        """
        T = self.T.to('GeV').value
        Tr = self.proton.Em.to('GeV').value
        A = 1.9e4
        i = -2.78
        B = 0.4866
        j = -2.51
        P = np.sqrt(T*(T + 2*Tr))
        LIS = A * P**i / (1.0 + B * P**j)
        unit = 1 / (u.m**2 * u.s * u.steradian * u.GeV)
        LIS = LIS*unit
        return LIS.to(self.differential_intensity_unit)
        
    def diffusion_simple(self):
        """
        Return a simple constant diffusion coefficient, doesn't depend on space or energy
        """
        A = 4.38e22 * u.cm**2 / u.s
        kappa = A.to(u.km**2 / u.s) * np.ones(self.x.shape)
        dkappa_dr = 0.0 * np.ones(self.x.shape) * (u.km / u.s)
        return kappa, dkappa_dr
    
    def diffusion_cm04(self, proton):
        """
        Compute diffusion coefficeint as per CM04.
        In this instance kappa is independent of r, so dkappa_dr = 0.
        """
        A = 4.38 * u.cm**2 / (u.s * u.GV)
        kappa = (A * proton.beta * proton.rigidity).to(u.km**2 / u.s) * np.ones(self.x.shape)
        dkappa_dr = 0.0 * np.ones(self.x.shape) * (u.km / u.s)
        return kappa, dkappa_dr
    
    def diffusion_gu71(self, proton):
        """
        Compute diffusion coefficeint as per GU71.
        """
        A = 1.9e21 * u.cm**2 / (u.s * u.GV * u.AU)
        kappa = (A * self.x.to('AU') * proton.beta * proton.rigidity).to(u.km**2 / u.s)
        dkappa_dr = (A * proton.beta * proton.rigidity).to(u.km / u.s) * np.ones(self.x.shape)
        return kappa, dkappa_dr
    
    def diffusion_coefficient(self, proton):
        """
        Return the diffusion coefficient and it's spatial gradient corresponding to the diffusion type specified in
        self.
        """
        if self.diffusion_type == "simple":
            kappa, dkappa_dr = self.diffusion_simple()
        elif self.diffusion_type == "cm04":
            kappa, dkappa_dr = self.diffusion_CM04(proton)
        elif self.diffusion_type == "gu71":
            kappa, dkappa_dr = self.diffusion_GU71(proton)
        return kappa, dkappa_dr
    
    def solar_wind_profile(self):
        """
        Compute model solar wind as per GU71, with asymptotic wind speed of Vo=400km/s, and decay halflife of
        15 solar radii
        """
        Vo = 400.0 * u.km / u.s
        lambda_v = 15.0 * u.solRad
        ro = 1.0 * u.solRad
        arg = (ro.to('km') - self.x) / lambda_v.to('km')
        self.V = Vo * (1.0 - np.exp(arg))
        self.dV_dr = (Vo - self.V) / lambda_v.to('km')
        return
    
    def compute_chi(self):
        """
        Return the chi term from the transport equation, which depends only on the solar wind flow.
        """
        self.chi = (2.0 * self.V / self.x) + self.dV_dr
        return
    
    def compute_a_coefficient(self, proton, dT):
        a = self.chi * proton.alpha * proton.T
        a = (a / dT)
        return a
    
    def compute_b_coefficient(self, proton):
        kappa, dkappa_dr = self.diffusion_coefficient(proton)
        b = -kappa / (2.0 * self.dx**2)
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
        main_diag = a + 2.0*b
        # Update inner Neumann boundary for dUdx=0
        main_diag[0] = a[0] + b[0] + c[0]
        
        off_diag_up = -b - c
        off_diag_dn = -b + c
        diagonals = [main_diag, off_diag_dn, off_diag_up]
        # This command destroys the unit. manhandle it.
        F = sparse.diags(diagonals, [0, -1, 1], shape=(self.x.size, self.x.size)).toarray()
        F = F * main_diag.unit
        
        # Matrix B
        main_diag = a - 2.0*b + d
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
