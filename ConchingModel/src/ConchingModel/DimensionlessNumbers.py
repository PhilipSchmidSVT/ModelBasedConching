" Module for the calculation of dimensionless number and derived mass transport coefficients. "
from ConchingModel import DataClass


class DimNumberCalculator:
    """
    This class handles the calculation of dimensionless characteristic numbers
    from the meta_data supplied by the experiment/process.
    """

    def __init__(self, meta_data: DataClass.MetaData):
        self.meta_data = meta_data

    @property
    def rpm(self) -> float:
        "Agitator frequency/rpm"
        return self.meta_data.rpm

    @property
    def power(self) -> float:
        "Agitator power uptake"
        return self.meta_data.power

    @property
    def viscosity(self) -> float:
        "Suspension viscosity"
        return self.meta_data.viscosity

    @property
    def density(self) -> float:
        "Suspension density"
        return self.meta_data.density

    @property
    def characteristic_length(self) -> float:
        "Characteristic length of system"
        return self.meta_data.characteristic_length

    @characteristic_length.setter
    def characteristic_length(self, val: float):
        "Set characteristic map of system"
        self.meta_data.characteristic_length = val

    @property
    def reynolds_number(self) -> float:
        "Reynolds Number"
        re = (self.characteristic_length**2 * self.frequency) / self.viscosity
        return re

    @property
    def newton_number(self) -> float:
        "Newton Number"
        ne = self.power / (
            self.density * self.frequency**3 * self.characteristic_length**5
        )
        return ne

    @property
    def schmidt_number(self) -> float:
        "Schmidt Number"
        sc = self.viscosity / self.diffusion_coeff
        return sc

    @property
    def peclet_number(self) -> float:
        "Peclet Number"
        pe = self.reynolds_number * self.schmidt_number
        return pe

    @property
    def diffusion_coeff(self) -> float:
        "Diffusion coefficient"
        result = self.meta_data.diffusion_coefficient
        return result

    def calc_sherwood_number(self, mtc, surface_area):
        """Calculate the Sherwood number, which represents the characteristic
        number linking the mechanical and the mass transport characteristic
        numbers as it is a function of the Newton number and the Péclet number

        Parameters:
        mtc (float): mass transport coefficient of air to cocoa
                                      butter boundary
        surface_area (float): surface area of air to cocoa butter boundary
        diffusion_coeff (float): diffusion coefficient of aroma compound in
                                 cocoa butter

        Returns:
        sh (float): Sherwood number

        """
        sh = (mtc * surface_area) / (self.diffusion_coeff * self.characteristic_length)
        return sh
