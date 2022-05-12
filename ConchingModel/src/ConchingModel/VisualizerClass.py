# Module import
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.animation import FuncAnimation, writers

# Class definition


class VisualizerClass:
    """Handling of visualization of concentration profiles created,
    for example, by ConcheClass.runODE().
    Can be used to create static charts and dynamic visualizations.
    """

    def __init__(self):
        pass

    def read_data(self, t, data):
        """Reads the numpy array resulting from a run of runODE.
        Parameters:
            data (np.array, float):
                Result of runODE. Rows contain the concentration in the phases.
                The first row is always the cocoa butter.
                All following rows can be an arbitrary number
                of suspended phases or air. Usually the order will be:
            1. row: cocoa butter
            2. row: cocoa particles
            3. row: sugar particles
            4. row: air
        """

        self.t = t
        self.phase_data = data
        self.n_phases, _ = data.shape

    def make_lineplot(self):
        """Create a lineplot of the phases. The time is on the abscissa,
        the concentration on the ordinate.
        Parameters:

        Returns:
        fig, ax: Figure and axe objects created.
        """
        fig, ax1 = plt.subplots()
        ax1.set_xlabel("Time [h]")
        ax1.set_ylabel("Phase associated amount [mol]")
        ax2 = ax1.twinx()
        ax2.set_ylabel("Phase associated amount [mol]")

        for i_phase in range(0, self.n_phases - 1):
            curr_label = "Phase #{}".format(i_phase)
            if i_phase == 0:
                ax1.plot(
                    self.t, self.phase_data[i_phase], color="y", label="Cocoa butter"
                )
            else:
                ax2.plot(self.t, self.phase_data[i_phase], label=curr_label)

        ax1.set_ylim(ax2.get_ylim())

    def make_radar_chart(self, timepoint=None):
        """Draw a radar chart from data"""

        labels = ("Cocoa butter", "Cocoa particles", "Sugar particles")
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)

        if timepoint is None:
            radar_data = self.phase_data[0:3, :]
        else:
            radar_data = self.phase_data[0:3, timepoint]

        fig = plt.figure()
        ax = fig.add_subplot(111, polar=True)
        (line,) = ax.plot(angles, radar_data, "-", linewidth=2)
        (filler,) = ax.fill(angles, radar_data, alpha=0.25)
        ax.set_thetagrids(angles * 180 / np.pi, labels)
        ax.grid(True)
        ax.set_ylim((0, sum(self.phase_data[:, 0])))

    def make_animated_radar_chart(self, apply_transparency=True, save_dest="test.mp4"):
        """Make an animated radar chart from data."""
        # Setup
        sum_N0 = np.sum(self.phase_data[0:3, 0])
        n_phases, n_times = self.phase_data.shape
        cmap = cm.get_cmap("Blues")
        # Specific labels
        labels = ["Cocoa butter", "Cocoa particles", "Sugar particles"]
        # Start plots
        fig = plt.figure()
        ax = fig.add_subplot(111, polar=True)
        ax.set_ylabel(r"$\frac{N}{\sum{N_0}}$")
        (line,) = ax.plot([], [], linewidth=2)
        plt.tight_layout()

        def init():
            line.set_data([], [])
            return (line,)

        def animate(i_time):
            # Set angles and labels
            angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
            ax.set_thetagrids(angles * 180 / np.pi, labels)
            # Set current data and color
            y = self.phase_data[0:3, i_time]
            sum_N = np.sum(y)
            curr_color = cmap(sum_N / sum_N0)
            # Return to start
            angles = np.hstack((angles, angles[0]))
            y = np.hstack((y, y[0]))
            # Draw line
            line.set_data(angles, y)
            if apply_transparency is True:
                line.set_color(curr_color)
            return (line,)

        anim = FuncAnimation(fig, animate, init_func=init, frames=n_times, blit=True)

        writer = writers["ffmpeg"]
        writer = writer(fps=25, metadata=dict(artist="Me"), bitrate=1800)

        anim.save(save_dest, writer=writer, dpi=300)

    def make_animated_radar_chart_single_phase(
        self, labels, phase_data, save_dest="test_SP.mp4"
    ):
        """
        Make a radar chart with substances as labels.
        The radar chart only depicts the concentration in a single phase.

        Parameters:
        labels (list, string) = Names of the substances in question
        phase_data (np.array, float) = Concentration profile of multiple aroma
        components. Each aroma component occupys one row.
            Each column is a point in time
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, polar=True)
        ax.set_ylabel(r"$\frac{N}{\sum{N_0}}$")
        (line,) = ax.plot([], [], linewidth=2)
        plt.tight_layout()
        _, n_times = phase_data.shape

        def init():
            line.set_data([], [])
            return (line,)

        def animate(i_time):
            # Set angle and labels
            angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
            ax.set_thetagrids(angles * 180 / np.pi, labels)
            # Extract current data
            y = phase_data[:, i_time]
            # Add return point to angles and data
            angles = np.hstack((angles, angles[0]))
            y = np.hstack((y, y[0]))
            # Update data
            line.set_data(angles, y)
            return (line,)

        anim = FuncAnimation(fig, animate, init_func=init, frames=n_times, blit=True)

        writer = writers["ffmpeg"]
        writer = writer(fps=25, metadata=dict(artist="Me"), bitrate=1800)

        anim.save(save_dest, writer=writer, dpi=300)

    def make_animated_aroma_profile_spider(self, data, save_dest="anim.mp4"):
        """Construct an aroma profile from the data provided.
        Parameters:
        data (np.array, floats): Concentration of component in phases.
        Dimension 0 are substances:
        Acetic acid,
        Benzaldehyde,
        TMP,
        Phenylethanol,
        Linalool.
        Dimension 1 are phases:
        Cocoa butter,
        Cocoa particles,
        Sugar particles,
        Air
        Dimension 2 are points in time.
        Exclude air from calculations, does not contribute to aroma.
        Columns are points in time.
        """

        # Aroma descriptors
        LABELS = ["Vanilla", "Caramel", "Fruit", "Sour", "Honey", "Roasted", "Flowery"]

        # Process data
        data_without_air = data[:, 0:3, :]
        data_summed = np.sum(data_without_air, axis=1)
        n_substances, _ = data_summed.shape

        # Aroma correlation matrix
        aroma_matrix = np.random.randn(len(LABELS), n_substances)

        # Calculate aroma profile at every timestep
        aroma_profile = np.dot(aroma_matrix, data_summed)
        aroma_profile = np.abs(aroma_profile)
        _, n_times = aroma_profile.shape

        # Start plotting
        fig = plt.figure()
        ax = fig.add_subplot(111, polar=True)
        ax.set_ylim((0, aroma_profile.max()))
        ax.set_yticks([])
        (line,) = ax.plot([], [], linewidth=2)
        plt.tight_layout()

        def init():
            line.set_data([], [])
            return (line,)

        def animate(i_time):
            # Set angle and labels
            angles = np.linspace(0, 2 * np.pi, len(LABELS), endpoint=False)
            ax.set_thetagrids(angles * 180 / np.pi, LABELS)
            # Extract current data
            y = aroma_profile[:, i_time]
            # Add return point to angles and data
            angles = np.hstack((angles, angles[0]))
            y = np.hstack((y, y[0]))
            # Update data
            line.set_data(angles, y)
            return (line,)

        anim = FuncAnimation(fig, animate, init_func=init, frames=n_times, blit=True)

        writer = writers["ffmpeg"]
        writer = writer(fps=25, metadata=dict(artist="Me"), bitrate=1800)

        anim.save(save_dest, writer=writer, dpi=300)

        pass

    @staticmethod
    def visualize_PLP(out_path):
        import os

        import numpy as np

        dirname = os.path.dirname(__file__)
        filelist = [file for file in os.listdir(dirname) if "csv" in file.split(".")]

        for ifile, file in enumerate(filelist):
            currpath = os.path.join(dirname, file)
            data = np.loadtxt(currpath, dtype=np.float)
            x = data[:, 0]
            y = data[:, 1]

            fig, ax = plt.subplots()
            ax.plot(x, y)
            ax.set_xlabel("Parameter value")
            ax.set_ylabel("SSR")
            fig.tight_layout()
            fig.savefig(f"{out_path}/{file}.png")


def viz_forward_sim(gen_fun):
    "This wrapper can be used to return a figure instead of np.ndarray"

    def run_sim_wrapper(exp, substance, mtc):
        sim_out = gen_fun(exp, substance, mtc)
        fig, ax = plt.subplots()
        ax.plot(sim_out.T, "o")
        ax.set_xticks(exp.get_sampling_times())
        ax.set_xlabel("Time in h")
        ax.set_ylabel(r"c in $\mu g/g$")
        ax.set_title(substance)
        plt.legend(labels=exp.get_types())
        return fig

    return run_sim_wrapper


def vis_residual_contour_map(grid: np.ndarray, jout: np.ndarray) -> plt.figure:
    assert grid.ndim > 1
    assert not any(np.isinf(jout)) or not any(np.isnan(jout))

    fig, ax = plt.subplots()
    beta_1, beta_2 = grid
    ax.contour(X=beta_1, Y=beta_2, Z=jout, cmap="gray")
    return fig


if __name__ == "__main__":
    pass
