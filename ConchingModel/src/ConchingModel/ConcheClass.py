"""
ConcheClass provides the ConcheClass objects, which are used for simulating
the conching process and calculating dimensionless numbers associated with
the experiment/process.
Usage:
conche = ConcheClass.ConcheClass(meta_data: MetaData.MetaData)
conche = meta_data.spawn_conche()
sim_result = conche.run_sim(t, c0, mtc, K)
"""
import ConchingModel.Data.MetaData as MetaData
import ConchingModel.Data.SimData as SimData
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

plt.style.use("seaborn-whitegrid")


class ConcheClass:
    """This class contains all necessary methods and data to carry out the
    simulation of the conche mass transport system."""

    COLORS = ((1, 0.753, 0), (0.68, 0.33, 0.09), (0, 0, 0), (0.5, 0, 0.5))

    def __init__(self, meta_data):
        """Initialize the class with the characterising length of the conche
        geometry

        Parameters:
            meta_data dict -> Information necessary for setting up conche.
              Required fields are "substance" and "mass"
        Returns:
            ConcheClass: initialized conche object

        """

        if not ConcheClass.check_required_fields(meta_data):
            raise ValueError
        self.meta_data = meta_data

    @staticmethod
    def check_required_fields(meta_data) -> bool:
        "Check if the field in meta_data contain required keys and vals"
        if not isinstance(meta_data, MetaData.aMetaData):
            raise TypeError
        required_keys = ("substance", "mass")
        check = [hasattr(meta_data, key) for key in required_keys]
        return all(check)

    def construct_dNdt(self, t, c, mtc, K):
        "Construct the system of equations governing mass transport in the conche"
        if len(c) > 1:
            if isinstance(c, np.ndarray):
                c = c.flatten().tolist()
            delta_c = -(np.multiply(c[1:], K) - c[0])

            dc_disp = delta_c * mtc[:-1]

            dc_cb = -self.dc_disperse_to_dc_continuous(dc_disp) - (c[0] * mtc[-1])
            dcdt = [dc_cb, *dc_disp]
        else:
            dcdt = -(mtc * c)
        return dcdt

    def dc_disperse_to_dc_continuous(self, dc_disp):
        """
        Change in concentration of disperse phases is calculated as total amount
        and converted to change in c_cb.
        """
        return np.sum(
            ((dc_disp * self.meta_data.phase_mass[1:]) / self.meta_data.phase_mass[0])
        )

    def run_sim(self, t, c0, mtc, K) -> np.ndarray:
        """
        Run the conching simulation by supplying starting conditions, a timespan,
        partition_coefficients and mass transfer coefficients.

        Args:
            t ([float]): Times at which the system is sampled
            c0 ([float]): Starting concentration
            mtc ([float]): mass transfer coefficients to run the system with
            K ([float]): Partition coefficients between the phases

        Returns:
            np.ndarray: Concentration at sampling times
        """
        args = (mtc, K)
        if len(t) == 1:
            t_span = (0.0, t[0])
        else:
            t_span = (min(t), max(t))
        atol = 1e-12
        if isinstance(c0, (tuple, list)) or c0.dtype != "complex128":
            method = "LSODA"
        else:
            method = "RK45"
        sim_result = solve_ivp(
            self.construct_dNdt,
            t_span,
            c0,
            method=method,
            dense_output=True,
            t_eval=t,
            vectorized=True,
            atol=atol,
            args=args,
        )
        assert sim_result.status == 0
        return sim_result.y


class ConcheClassCombinedParticles(ConcheClass):
    """
    This Subclass is used to treat an experiment setup with non-complete
    observability. Particle phases are treated as a single phase, conjoining
    their concentrations. This is due to experiments being unable to separate
    the phases, leading to non complete observability. The underlying system
    is still the same, just the observable states are reduced.

    Args:
        ConcheClass (MetaData.aMetaData): meta_data for the conche
    """

    def run_sim(self, t, c0, mtc, K) -> np.ndarray:
        total_observable_state = super().run_sim(t, c0, mtc, K)
        result = self.mass_to_concentration(
            self.add_masses(self.concentration_to_mass(total_observable_state))
        )
        return result

    def concentration_to_mass(self, concentration):
        "Turn concentration of phases into absolute substance amount"
        result = np.multiply(concentration.T, self.meta_data.phase_mass)
        return result.T

    def add_masses(self, masses):
        """Combine masses for particle phases and return the reduced
        mass array in order to correctly calculate a combined particle phase"""
        particle_masses = np.sum(masses[1:, :], axis=0)
        result = np.vstack((masses[0, :], particle_masses))
        return result

    def mass_to_concentration(self, masses):
        "Turn substance masses associated with phases to concentration of phase"
        combined_phase_mass = [
            self.meta_data.phase_mass[0],
            sum(self.meta_data.phase_mass[1:]),
        ]
        assert len(combined_phase_mass) == masses.shape[0]
        result = np.divide(masses.T, combined_phase_mass)
        return result.T


if __name__ == "__main__":
    pass
