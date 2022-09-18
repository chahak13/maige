"""
First version for generativeart.

Owner: Chahak Mehta (chahakmehta013 [at] gmail [dot] com)
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import tqdm
from matplotlib.animation import FuncAnimation

from maige.functions import ExpressionTree


class Generator:
    """
    Base class to generate images.

    This class defines various methods that are used in generating the
    images.
    """

    def __init__(
        self,
        pointcolor="#000000",
        background="#ffffff",
        xfunc=None,
        yfunc=None,
        projection="rectilinear",
        xrange=None,
        yrange=None,
        fig=None,
        ax=None,
        seed=None,
    ):
        """Set variables for the base class Generator."""
        self.pointcolor = pointcolor
        self.background = background
        self.xfunc = xfunc
        self.yfunc = yfunc
        self.projection = projection
        self.xrange = (
            np.arange(-np.pi, np.pi, 0.01) if xrange is None else xrange
        )
        self.yrange = (
            np.arange(-np.pi, np.pi, 0.01) if yrange is None else yrange
        )
        self.fig = fig
        self.ax = ax
        self.seed = seed
        if isinstance(self.seed, str):
            self.random_state = json.load(self.seed)["random_state"]
            self.rng = np.random.default_rng()
            self.rng.__setstate__(self.random_state)
        elif isinstance(self.seed, int) or self.seed is None:
            self.rng = np.random.default_rng(self.seed)
            self.random_state = self.rng.__getstate__()
        else:
            raise ValueError(
                f"Random seed for Generator should be an int or a"
                f"state dictionary, found {type(self.seed)}"
            )

    def _create_fig(self, **kwargs):
        fig, ax = (
            plt.subplots(figsize=(7, 7), **kwargs)
            if self.fig is None or self.ax is None
            else (self.fig, self.ax)
        )
        ax.set_axis_off()
        ax.set_facecolor(self.background)
        fig.set_facecolor(self.background)
        ax.set_xticks([])
        ax.set_yticks([])
        return fig, ax

    def _create_mesh(self, **kwargs):
        X, Y = np.meshgrid(self.xrange, self.yrange)
        return X, Y

    def _x_function(self, X, Y):
        """
        Return changed for x values of points based on the function.

        Arguments:
        mesh: np.meshgrid Grid

        Returns:
        np.array: Changed X values
        """
        tree = ExpressionTree(self.rng, depth=0, max_depth=5, variables=[X, Y])
        tree.generate_function(None, 2, 1, None, None, None)
        return tree

    def _y_function(self, X, Y):
        """
        Return changed for y values of points based on the function.

        Arguments:
        mesh: np.meshgrid Grid

        Returns:
        np.array: Changed Y values
        """
        tree = ExpressionTree(self.rng, depth=0, max_depth=5, variables=[X, Y])
        tree.generate_function(None, 2, 1, None, None, None)
        return tree

    def _save_info(self, filetype, filepath):
        info = {
            "user_seed": self.seed,
            "random_state": self.random_state,
            "filetype": filetype,
            "filepath": filepath,
            "x_function": self.xfunc.get_expr_string()
            if isinstance(self.xfunc, ExpressionTree)
            else None,
            "y_function": self.yfunc.get_expr_string()
            if isinstance(self.yfunc, ExpressionTree)
            else None,
        }
        basename, filename = os.path.split(os.path.abspath(filepath))
        filename, extension = filename.rsplit(".")
        json_filepath = os.path.join(basename, filename + ".json")
        json.dump(info, open(json_filepath, "w"), indent=4)
        print(f"Stored run info at {json_filepath}")

    def generate_image(self, filepath="", **kwargs):
        """Generate the image and save it."""
        fig, ax = self._create_fig(subplot_kw={"projection": self.projection})
        X, Y = self._create_mesh()
        if not self.xfunc:
            self.xfunc = self._x_function(X, Y)
            x_res = self.xfunc.execute()
        else:
            x_res = self.xfunc(X, Y).real

        if not self.yfunc:
            self.yfunc = self._y_function(X, Y)
            y_res = self.yfunc.execute()
        else:
            y_res = self.yfunc(X, Y).real

        ax.scatter(x_res, y_res, c=self.pointcolor, s=0.2, alpha=0.05)
        if not filepath:
            filepath = (
                f"{self.random_state['bit_generator']}"
                f"_{self.random_state['state']['state']}.png"
            )

        self._save_info("image", filepath)
        fig.savefig(filepath)
        return fig, ax

    def generate_animation(
        self, filepath="./examples/temp.mp4", init_cond="linear", **kwargs
    ):
        """Generate animation for the given formulae."""
        X, Y = self._create_mesh()
        if not self.xfunc:
            self.xfunc = self._x_function(X, Y)
            x_res = self.xfunc.execute()
        else:
            x_res = self.xfunc(X, Y)

        if not self.yfunc:
            self.yfunc = self._y_function(X, Y)
            y_res = self.yfunc.execute()
        else:
            y_res = self.yfunc(X, Y)

        x_points = x_res.reshape(-1, 1)
        y_points = y_res.reshape(-1, 1)
        if init_cond == "linear":
            initial_points = np.stack(
                [
                    np.arange(x_points.shape[0]).reshape(-1, 1),
                    np.zeros(x_points.shape),
                ]
            )
        elif init_cond == "uniform":
            low = np.min(self.xrange)
            high = np.max(self.xrange)
            initial_points = np.stack(
                [
                    self.rng.uniform(low, high, x_points.shape),
                    self.rng.uniform(low, high, y_points.shape),
                ]
            )
        else:
            raise ValueError(
                "Invalid value of init_cond for points."
                " Use value from {linear, uniform}"
            )
        final_points = np.stack([x_points, y_points])

        slopes = (final_points[1, :] - initial_points[1, :]) / (
            final_points[0, :] - initial_points[0, :]
        )
        intercepts = initial_points[1, :] - slopes * initial_points[0, :]

        fig, ax = self._create_fig(subplot_kw={"projection": self.projection})
        scat = ax.scatter(
            final_points[0, :],
            final_points[1, :],
            c=self.pointcolor,
            s=0.2,
            alpha=0.05,
        )

        dx = (final_points[0, :] - initial_points[0, :]) / len(self.yrange)

        def generate_data():
            while True:
                i = yield
                if i < len(self.yrange):
                    points = initial_points[0, :] + (i + 1) * dx
                    lines = slopes * points + intercepts
                    data = np.stack([points, lines]).squeeze()
                else:
                    data = final_points.squeeze()
                yield data

        generator = generate_data()
        pbar = tqdm.tqdm()

        def animate(i):
            next(generator)
            data = generator.send(i)
            scat.set_offsets(data.T)
            pbar.update()
            pbar.refresh()
            return (scat,)

        ani = FuncAnimation(
            fig=fig, func=animate, blit=True, frames=len(self.yrange) + 50
        )
        if not filepath:
            filepath = (
                f"{self.random_state['bit_generator']}"
                f"_{self.random_state['state']['state']}.mp4"
            )
        ani.save(filepath, writer="ffmpeg", fps=24)
        self._save_info("video", filepath)
