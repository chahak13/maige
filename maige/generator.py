# MIT License.

# Copyright (c) 2022 Chahak Mehta

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Define the base Generator class.

The public interface of the library is via the Generator class that sets up all
the figure properties and handles the plotting.
"""

import json
import os

from typing import Callable, Union
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

    Usage example:

    >>> from maige.generator import Generator
    >>> g = Generator()
    >>> g.generate_image("/path/to/store/image.png")
    """

    def __init__(
        self,
        pointcolor: str = "#000000",
        background: str = "#ffffff",
        xfunc: Callable = None,
        yfunc: Callable = None,
        projection: str = "rectilinear",
        xrange: np.ndarray = None,
        yrange: np.ndarray = None,
        fig=None,
        ax=None,
        seed: Union[int, str] = None,
    ):
        """Create the base class to initialize the image generator.

        Args:
            pointcolor: str
                Hex string used as color for the points in the figure.

            background: str
                Hex string used to set the color for the background of
                the figure.

            xfunc: Callable
                Function used to transform the x-coordinates. This is an
                optional argument. If not provided, a random function is
                generated using an Expression Tree.

            yfunc: Callable
                Function to transform the y-coordinates. This is an optional
                argument like xfunc and a random function is generated if
                one is not provided.

            projection: str
                A string to determine the projection used for the figure.
                It accepts any string accepted by matplotlib.pyplot.figure's
                `projection` argument.

            xrange: np.ndarray
                A numpy array of x-coordinates to calculate xfunc on.

            yrange: np.ndarray
                A numpy array of y-coordinates to calculate yfunc on.

            fig: matplotlib.figure.Figure
                Matplolib figure to use for plotting. __Optional__

            ax: matplotlib.axes.Axes
                Matplotlib axes to use for plotting. __Optional__

            seed: int or str
                Random seed to determine state for the numpy random generator.
                If `int`, the integer is passed directly as seed to
                `numpy.random.default_rng`. If `str`, it expects a path to
                a json file that has a key called 'random_state' that can be
                used by the `numpy.random.Generator.__setstate__` to set the
                random state.
        """
        self._pointcolor = pointcolor
        self._background = background
        self._xfunc = xfunc
        self._yfunc = yfunc
        self._projection = projection
        self._xrange = (
            np.arange(-np.pi, np.pi, 0.01) if xrange is None else xrange
        )
        self._yrange = (
            np.arange(-np.pi, np.pi, 0.01) if yrange is None else yrange
        )
        self.fig = fig
        self.ax = ax
        self.seed = seed
        if isinstance(self.seed, str):
            self.random_state = json.load(open(self.seed, "r"))["random_state"]
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

    def __create_fig(self, **kwargs):
        fig, ax = (
            plt.subplots(figsize=(7, 7), **kwargs)
            if self.fig is None or self.ax is None
            else (self.fig, self.ax)
        )
        ax.set_axis_off()
        ax.set_facecolor(self._background)
        fig.set_facecolor(self._background)
        ax.set_xticks([])
        ax.set_yticks([])
        return fig, ax

    def __create_mesh(self, **kwargs):
        X, Y = np.meshgrid(self._xrange, self._yrange)
        return X, Y

    def __generate_x_func(self, X, Y):
        """
        Return changed for x values of points based on the function.

        Args:
            mesh: np.meshgrid Grid

        Returns:
            np.array: Changed X values
        """
        tree = ExpressionTree(self.rng, depth=0, max_depth=5, variables=[X, Y])
        tree.generate_function(None, 2, 1, None, None, None)
        return tree

    def __generate_y_func(self, X, Y):
        """
        Return changed for y values of points based on the function.

        Args:
            mesh: np.meshgrid Grid

        Returns:
            np.array: Changed Y values
        """
        tree = ExpressionTree(self.rng, depth=0, max_depth=5, variables=[X, Y])
        tree.generate_function(None, 2, 1, None, None, None)
        return tree

    def __save_info(self, filetype, filepath):
        info = {
            "user_seed": self.seed,
            "random_state": self.random_state,
            "filetype": filetype,
            "filepath": filepath,
            "x_function": self._xfunc.get_expr_string()
            if isinstance(self._xfunc, ExpressionTree)
            else None,
            "y_function": self._yfunc.get_expr_string()
            if isinstance(self._yfunc, ExpressionTree)
            else None,
        }
        basename, filename = os.path.split(os.path.abspath(filepath))
        filename, extension = filename.rsplit(".")
        json_filepath = os.path.join(basename, filename + ".json")
        json.dump(info, open(json_filepath, "w"), indent=4)
        print(f"Stored run info at {json_filepath}")

    def generate_image(self, filepath="", **kwargs):
        """Generate the image and save it.

        Args:
            filepath: str
                Path to store the final image at. If not provided, a filename
                is generated based on the random generator and the seed.

            **kwargs:
                Keyword arguments passed onwards to matplotlib's scatter
                function.
        """
        fig, ax = self.__create_fig(subplot_kw={"projection": self._projection})
        X, Y = self.__create_mesh()
        if not self._xfunc:
            self._xfunc = self.__generate_x_func(X, Y)
            x_res = self._xfunc.execute()
        else:
            x_res = self._xfunc(X, Y).real

        if not self._yfunc:
            self._yfunc = self.__generate_y_func(X, Y)
            y_res = self._yfunc.execute()
        else:
            y_res = self._yfunc(X, Y).real

        ax.scatter(
            x_res, y_res, c=self._pointcolor, s=0.2, alpha=0.05, **kwargs
        )
        if not filepath:
            filepath = (
                f"{self.random_state['bit_generator']}"
                f"_{self.random_state['state']['state']}.png"
            )

        self.__save_info("image", filepath)
        fig.savefig(filepath, dpi=350)
        return fig, ax

    def generate_animation(
        self, filepath="./examples/temp.mp4", init_cond="linear", **kwargs
    ):
        """Generate animation for the given formulae.

        Args:
            filepath: str
                Path to store the final image at. If not provided, a filename
                is generated based on the random generator and the seed.

            init_cond: str
                Initial condition to start the animation from. Can be one of
                "linear" or "uniform". Default is set to "linear".

            **kwargs:
                Keyword arguments passed onwards to matplotlib's scatter
                function.
        """
        X, Y = self.__create_mesh()
        if not self._xfunc:
            self._xfunc = self.__generate_x_func(X, Y)
            x_res = self._xfunc.execute()
        else:
            x_res = self._xfunc(X, Y)

        if not self._yfunc:
            self._yfunc = self.__generate_y_func(X, Y)
            y_res = self._yfunc.execute()
        else:
            y_res = self._yfunc(X, Y)

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
            low = np.min(self._xrange)
            high = np.max(self._xrange)
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

        fig, ax = self.__create_fig(subplot_kw={"projection": self._projection})
        scat = ax.scatter(
            final_points[0, :],
            final_points[1, :],
            c=self._pointcolor,
            s=0.2,
            alpha=0.05,
        )

        dx = (final_points[0, :] - initial_points[0, :]) / len(self._yrange)

        def __generate_data():
            while True:
                i = yield
                if i < len(self._yrange):
                    points = initial_points[0, :] + (i + 1) * dx
                    lines = slopes * points + intercepts
                    data = np.stack([points, lines]).squeeze()
                else:
                    data = final_points.squeeze()
                yield data

        generator = __generate_data()
        pbar = tqdm.tqdm()

        def __animate(i):
            next(generator)
            data = generator.send(i)
            scat.set_offsets(data.T)
            pbar.update()
            pbar.refresh()
            return (scat,)

        ani = FuncAnimation(
            fig=fig, func=__animate, blit=True, frames=len(self._yrange) + 50
        )
        if not filepath:
            filepath = (
                f"{self.random_state['bit_generator']}"
                f"_{self.random_state['state']['state']}.mp4"
            )
        ani.save(filepath, writer="ffmpeg", fps=24)
        self.__save_info("video", filepath)
