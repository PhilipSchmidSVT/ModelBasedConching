import tkinter as tk
from tkinter import ttk

import pandas as pd
from ConchingModel import AnalysisLib, DataClass, DataLoaders


def label_entry_combo(parent, label_txt, grid, entrytype=tk.DoubleVar):
    label = ttk.Label(parent, text=label_txt)
    label.grid(row=grid[0], column=grid[1])
    id = entrytype()
    entry = ttk.Entry(parent, textvariable=id)
    entry.grid(row=grid[0], column=grid[1] + 1)
    return label, id


class MainFrame(ttk.Labelframe):
    "Pre-scripted Frame for Mainframe"

    def __init__(self, master=None, text=None):
        super().__init__(master, text=text)
        self.grid()
        self.create_subframes()
        self.create_widgets()
        self.exp = None

    def create_subframes(self):
        self.sampling_times = SamplingTime(self, text="Sampling Times")
        self.sampling_times.grid(row=0, column=0)

        self.recipe = Recipe(self, text="Meta Info")
        self.recipe.grid(row=0, column=1)

        self.measurement = Measurement(
            self.sampling_times.n_st.get(), master=self, text="Measurements"
        )
        self.measurement.grid(row=1, column=0, columnspan=2)

    def create_widgets(self):
        refresh_button = ttk.Button(
            self.sampling_times, text="Clear", command=self.clear
        )
        refresh_button.grid(row=1, column=0)

        restart_button = ttk.Button(
            self.sampling_times, text="Refresh", command=self.refresh
        )
        restart_button.grid(row=1, column=1)

        load_json_button = ttk.Button(
            self.recipe, text="Load JSON", command=self.load_json
        ).grid(row=10, column=0)

        experiment_button = ttk.Button(
            self, text="Create experiment", command=self.create_experiment
        )
        experiment_button.grid(row=2, column=0)

    def get_nonzero_phases(self):
        phase_keys = ["cb", "cp", "sp"]
        phase_per = [self.recipe.entries[phase_key].get() for phase_key in phase_keys]
        is_phase = [(phase is not None and phase > 0) for phase in phase_per]
        return is_phase

    def clear(self):
        if len(self.measurement.children) > 0:
            self.measurement.subframe.destroy()
            self.grid()
        else:
            pass

    def refresh(self):
        if len(self.measurement.children) == 0:
            is_phase = self.get_nonzero_phases()
            self.measurement.draw_frame(self.sampling_times.n_st.get(), is_phase)
            self.measurement.grid()
            self.grid()
        else:
            pass

    def create_experiment(self):
        self.exp = DataClass.Experiment(
            DataClass.MetaData(self.create_meta_info_for_experiment()),
            partition_coeffs=self.recipe.get_K(),
        )
        self.data = self.measurement.entries_to_df(self.measurement.read_entries())
        ExperimentWindow(
            master=self,
            exp=self.exp,
            data=self.measurement.entries_to_df(self.measurement.read_entries()),
        )

    def create_meta_info_for_experiment(self):
        meta_info = {
            "mass": self.recipe.get_mass(),
            "sampling_times": self.measurement.get_sampling_times(),
            "timeseries": self.measurement.get_timeseries(),
            "fractions": self.recipe.get_fractions(),
            "temp": self.recipe.get_temp(),
            "substance": self.recipe.get_substance(),
        }
        return meta_info

    def load_json(self):
        filepath = self.recipe.json_path.get()
        data = DataLoaders.FromJson(filepath).data
        n_st = len(data.sampling_times)
        self.sampling_times.n_st.set(str(n_st))
        self.set_recipe_fields(data)
        self.clear()
        self.refresh()
        self.set_measurement_fields(data)

    def set_recipe_fields(self, data) -> None:
        for key in dir(data):
            if key in ("temp", "mass", "substance"):
                self.recipe.entries[key].set(getattr(data, key))
            elif key == "fractions":
                for subkey, subvalue in getattr(data, key).items():
                    self.recipe.entries[subkey].set(subvalue)

    def set_measurement_fields(self, data: dict) -> None:
        for i, i_t in enumerate(self.measurement.t):
            i_t.set(data.sampling_times[i])

        for i, i_phase in enumerate(self.measurement.c):
            for j, i_st in enumerate(i_phase):
                i_st.set(data.timeseries[i][j])


class SamplingTime(ttk.Labelframe):
    "Pre-scripted Frame for Metadata"

    def __init__(self, master=None, text=None):
        super().__init__(master, text=text)
        self.grid()
        self.create_widgets()

    def create_widgets(self):
        st_label = ttk.Label(self, text="Number of sampling times")
        st_label.grid(row=0, column=0)
        self.n_st = tk.IntVar()
        self.n_st.set(5)
        st_entry = ttk.Entry(self, width=2, textvariable=self.n_st)
        st_entry.grid(row=0, column=2)


class Recipe(ttk.Labelframe):
    "Pre-scripted frame for recipe input"

    def __init__(self, master=None, text=None):
        super().__init__(master, text=text)
        self.grid()
        self.create_widgets()

    def create_widgets(self):
        self.entries = dict()
        _, self.entries["mass"] = label_entry_combo(self, "Total mass [g]", (0, 0))
        _, self.entries["cb"] = label_entry_combo(self, "Ratio cocoabutter [%]", (1, 0))
        _, self.entries["cp"] = label_entry_combo(
            self, "Ration cocoa particles [%]", (2, 0)
        )
        _, self.entries["sp"] = label_entry_combo(
            self, "Ration sugar particles [%]", (3, 0)
        )
        _, self.entries["K_cbcp"] = label_entry_combo(self, "Kcbcp", (4, 0))
        _, self.entries["K_cbsp"] = label_entry_combo(self, "Kcbsp", (5, 0))
        _, self.entries["temp"] = label_entry_combo(self, "Temperature [°C]", (6, 0))
        _, self.entries["substance"] = label_entry_combo(
            self, "Substance", (7, 0), tk.StringVar
        )

        ttk.Button(self, text="Load CSV", command=self.load_from_csv).grid(
            row=8, column=0
        )
        self.csv_path = tk.StringVar()
        ttk.Entry(self, textvariable=self.csv_path).grid(row=8, column=1)
        self.json_path = tk.StringVar()
        ttk.Entry(self, textvariable=self.json_path).grid(row=10, column=1)

    def load_from_csv(self):
        csv_content = pd.read_csv(self.csv_path.get(), header=0)
        available_keys = csv_content.columns.tolist()
        for key in available_keys:
            if key in self.entries.keys():
                self.entries[key].set(csv_content[key].iloc[0])

    def parse_meta_info(self):
        meta_info = dict()
        for key, value in self.entries.items():
            try:
                meta_info[key] = [float(value.get())]
            except ValueError:
                meta_info[key] = [value.get()]
        return pd.DataFrame.from_dict(meta_info)

    def get_K(self):
        K_vars = (self.entries["K_cbcp"], self.entries["K_cbsp"])
        K = [i_K.get() for i_K in K_vars]
        if all([i_k == 0.0 for i_k in K]):
            K = None
        return K

    def get_fractions(self):
        fractions = {
            "cb": self.entries["cb"].get(),
            "cp": self.entries["cp"].get(),
            "sp": self.entries["sp"].get(),
        }
        return fractions

    def get_nonzero_phases(self):
        fractions = self.get_fractions()
        phases = [
            key
            for key, value in fractions.items()
            if value is not None and value > 1e-12
        ]
        return phases

    def contains_sugar(self):
        phases = self.get_nonzero_phases()
        return "sp" in phases

    def get_recipe(self):
        return {key: value.get() for key, value in self.entries.items()}

    def get_temp(self):
        return self.entries["temp"].get()

    def get_substance(self):
        return self.entries["substance"].get()

    def get_mass(self):
        return self.entries["mass"].get()


class Measurement(ttk.Labelframe):
    "Pre-scriped Frame for Metadata"

    def __init__(self, n_times, master=None, text=None):
        super().__init__(master=master, text=text)
        self.master = master
        self.draw_frame(n_times, [True, True, True])
        self.grid()

    def draw_frame(self, n_times, is_phase):
        self.subframe = ttk.Frame(self)
        self.subframe.grid(row=0, column=0)
        all_phases = ("cb", "cp", "sp")
        self.valid_phases = [
            phase for phase, selector in zip(all_phases, is_phase) if selector is True
        ]

        # Draw labels
        ttk.Label(self.subframe, text="Time in h").grid(row=0, column=0)
        for i_phase, phase in enumerate(self.valid_phases):
            ttk.Label(self.subframe, text=phase).grid(row=i_phase + 1, column=0)

        self.t = [tk.DoubleVar() for _ in range(n_times)]
        for i_time in range(n_times):
            ttk.Entry(self.subframe, textvariable=self.t[i_time]).grid(
                row=0, column=i_time + 1
            )
        self.c = [
            [tk.DoubleVar() for i_time in range(n_times)]
            for i_phase in range(len(self.valid_phases))
        ]
        for i_phase in range(len(self.valid_phases)):
            for i_time in range(n_times):
                ttk.Entry(self.subframe, textvariable=self.c[i_phase][i_time]).grid(
                    row=i_phase + 1, column=i_time + 1
                )
        self.create_widgets()
        self.grid()

    def create_widgets(self):
        button_to_csv = ttk.Button(
            self.subframe, text="table2csv", command=self.entries_to_csv
        )
        button_to_csv.grid(row=4, column=0)

        button_plot = ttk.Button(self.subframe, text="Plot", command=self.plot_entries)
        button_plot.grid(row=4, column=1)

        button_from_csv = ttk.Button(
            self.subframe, text="Load CSV", command=self.load_from_csv
        )
        button_from_csv.grid(row=4, column=2)
        self.path_to_csv = tk.StringVar()
        entry_csv = ttk.Entry(self.subframe, textvariable=self.path_to_csv)
        entry_csv.grid(row=4, column=3)

    def read_entries(self):
        t = list(map(lambda i_t: float(i_t.get()), self.t))
        c = list(
            map(lambda i_phase: map(lambda i_c: float(i_c.get()), i_phase), self.c)
        )
        c = [list(element) for element in c]
        return t, c

    def get_sampling_times(self):
        t, c = self.read_entries()
        return t

    def get_timeseries(self):
        t, c = self.read_entries()
        return c

    def entries_to_df(self, tc_tuple):
        t, c = tc_tuple
        df = pd.DataFrame(c, columns=t, index=pd.Index(self.valid_phases))
        return df

    def entries_to_csv(self):
        self.entries_to_df(self.read_entries()).to_csv("default_path.csv")

    def plot_entries(self):
        ax = self.entries_to_df(self.read_entries()).T.plot()
        ax.set_xlabel("Time in h")
        ax.set_ylabel("Concentration")
        fig = ax.get_figure()
        fig.show()

    def load_from_csv(self):
        n_t, n_p = (len(self.t), len(self.c))
        path_to_csv = self.path_to_csv.get()
        if path_to_csv != "":
            csv_in = pd.read_csv(path_to_csv, header=0)
            assert csv_in.shape == (n_p, n_t)
            phases = csv_in.values.tolist()
            for i_phase in range(len(self.c)):
                for i_c in range(len(self.c[i_phase])):
                    self.c[i_phase][i_c].set(phases[i_phase][i_c])


###############################################################################


class ExperimentWindow(tk.Toplevel):
    "Launch a separate window for an experiment"

    def __init__(self, master=None, exp=None, data=None):
        super().__init__(master=master)
        self.title("Experiment")
        self.exp = exp
        self.data = data
        self.exp.gui_data = self.data
        self.grid()
        param_interface = ParamEstimInterface(master=self, exp=self.exp)
        param_interface.grid(row=0, column=1)


class ParamEstimInterface(ttk.Labelframe):
    "Interface for parameter estimation"

    def __init__(self, master=None, exp=None):
        super().__init__(master=master)
        self.exp = exp
        self.grid()
        self.bounds_frame = BoundsInterface(master=self, exp=self.exp)
        self.bounds_frame.grid(row=0, column=0)
        self.forward_frame = ForwardSimInterface(master=self)
        self.forward_frame.grid(row=0, column=1)

        ttk.Button(
            self, text="Run parameter estimation", command=self.execute_analysis_suite
        ).grid(row=1, column=0)
        ttk.Button(
            self, text="Run Forward Simulation", command=self.run_forward_sim
        ).grid(row=1, column=1)

    def execute_analysis_suite(self):
        analyses = [
            a(self.exp.meta_data, self.gather_analysis_kwargs())
            for a in self.bounds_frame.get_to_run()
        ]
        result = [analysis.run_analysis() for analysis in analyses]
        [res["fig"].show() for res in result]

    def gather_analysis_kwargs(self):
        result = {
            "GlobalParameterEstimation": {
                "bounds": self.bounds_frame.get_bounds(),
                "partition_coeff": self.gather_partition_coeffs(),
            },
            "ResidualMap": {
                "bounds": self.bounds_frame.get_bounds(),
                "partition_coeff": self.gather_partition_coeffs(),
            },
        }
        return result

    def gather_partition_coeffs(self):
        if self.exp.partition_coeffs is None:
            result = self.exp.pick_partition_coefficients()
        else:
            result = self.exp.partition_coeffs
        return result

    def run_forward_sim(self):
        mtcs = self.forward_frame.get_mtcs()
        sim_out = self.exp.run_forward_sim(mtcs)
        sim_out()


class BoundsInterface(ttk.Frame):
    def __init__(self, master=None, exp=None):
        super().__init__(master=master)
        self.grid()
        self.exp = exp
        self.create_widgets()

    def l_u_combo(self, txt, grid_pos):
        l, u = tk.DoubleVar(), tk.DoubleVar()
        ttk.Label(self, text=txt).grid(row=grid_pos[0], column=grid_pos[1])
        ttk.Entry(self, textvariable=l).grid(row=grid_pos[0], column=grid_pos[1] + 1)
        ttk.Entry(self, textvariable=u).grid(row=grid_pos[0], column=grid_pos[1] + 2)
        return l, u

    def create_checkbutton(self, name, pos):
        ttk.Label(self, text=name).grid(row=pos[0], column=pos[1])
        is_pressed = tk.BooleanVar()
        ttk.Checkbutton(self, variable=is_pressed).grid(row=pos[0] + 1, column=pos[1])
        return is_pressed

    def get_to_run(self):
        to_run_list = (
            AnalysisLib.GlobalParameterEstimation,
            AnalysisLib.ResidualMap,
            AnalysisLib.GlobalParameterEstimation,
        )
        checkbutton_state = map(lambda x: x.get(), self.checkbuttons)
        selector = zip(to_run_list, checkbutton_state)
        to_run = [id for id, is_enabled in selector if is_enabled is True]
        return to_run

    def get_bounds(self):
        bounds_out = [[i_bound.get() for i_bound in i_param] for i_param in self.bounds]
        return bounds_out

    def create_widgets(self):
        self.checkbuttons = [
            self.create_checkbutton("Global parameter estim.", (0, 0)),
            self.create_checkbutton("Residual Map", (0, 1)),
            self.create_checkbutton("Uncertainty analysis", (0, 2)),
        ]

        n_phases = len(self.exp.meta_data.phases)
        base_row_bounds = 3
        ttk.Label(self, text="lower bound").grid(row=base_row_bounds - 1, column=1)
        ttk.Label(self, text="upper bound").grid(row=base_row_bounds - 1, column=2)
        self.bounds = [
            self.l_u_combo(f"beta #{i_phase}", (base_row_bounds + i_phase, 0))
            for i_phase in range(n_phases)
        ]


class ForwardSimInterface(ttk.Frame):
    def __init__(self, master=None):
        super().__init__(master=master)
        self.grid()
        self.create_widgets()

    def param_entry(self, name, pos):
        ttk.Label(self, text=name).grid(row=pos[0], column=pos[1])
        par_var = tk.DoubleVar()
        ttk.Entry(self, textvariable=par_var).grid(row=pos[0], column=pos[1] + 1)
        return par_var

    def get_mtcs(self):
        mtcs = list(
            map(lambda x: x.get(), [self.beta_cbcp, self.beta_cbsp, self.beta_cba])
        )
        return mtcs

    def get_K(self):
        K = list(map(lambda x: x.get(), [self.K_cbcp]))
        return K

    def create_widgets(self):
        self.beta_cbcp = self.param_entry("beta_cbcp", (0, 0))
        self.beta_cbsp = self.param_entry("beta_cbsp", (1, 0))
        self.beta_cba = self.param_entry("beta_cba", (2, 0))


if __name__ == "__main__":
    pass
