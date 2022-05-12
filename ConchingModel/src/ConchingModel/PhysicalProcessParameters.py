" Wrapper for MetaData concerning physical process parameters. "


class PhysicalProcessParameters:
    "Handle physical process parameters."

    def __init__(self, meta_data):
        self.meta_data = meta_data

    @property
    def temp(self) -> float:
        "Temperature in degree Celsius during conching."
        if not isinstance(self.meta_data._meta_info["temp"], (int, float)):
            raise TypeError
        return self.meta_data._meta_info["temp"]

    @property
    def rpm(self):
        "Revolutions per minute of conche shaft."
        return self.meta_data._meta_info["rpm"]

    @property
    def power(self):
        return


class SuspensionProcessParameters:
    "Handle suspension process parameters."

    def __init__(self, meta_data):
        self.meta_data = meta_data

    @property
    def viscosity(self) -> float:
        "Return the viscosity of the suspension during the experiment."
        return self.meta_data._meta_info["viscosity"]

    @property
    def density(self) -> float:
        density = 920.8 - (0.7 * self.meta_data.PPP.temp)
        return density

    @property
    def area(self) -> dict:
        "Calculate the geometrical dimensions of the phases"
        grammage = (0.374, 0.236)  # m^2/g
        grammage_cp, grammage_sp = grammage
        area = dict()
        area["cb"] = self.meta_data.phase_mass["cb"] * 1e-3 / self.density
        area["cp"] = self.meta_data.phase_mass["cp"] * grammage_cp
        area["sp"] = self.meta_data.phase_mass["sp"] * grammage_sp
        area["pp"] = area["cp"] + area["sp"]
        return area
