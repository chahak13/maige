"""
First version for generativeart.

Owner: Chahak Mehta (chahakmehta013 [at] gmail [dot] com)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation


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
        projection="polar",
        xrange=None,
        yrange=None,
        fig=None,
        ax=None,
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
        return X + np.cos(Y)

    def _y_function(self, X, Y):
        """
        Return changed for y values of points based on the function.

        Arguments:
        mesh: np.meshgrid Grid

        Returns:
        np.array: Changed Y values
        """
        return Y + np.sin(Y) ** 2

    def generate_image(self, filepath="", **kwargs):
        """Generate the image and save it."""
        fig, ax = self._create_fig(subplot_kw={"projection": self.projection})
        X, Y = self._create_mesh()
        if not self.xfunc:
            x_res = self._x_function(X, Y)
        else:
            x_res = self.xfunc(X, Y)

        if not self.yfunc:
            y_res = self._y_function(X, Y)
        else:
            y_res = self.yfunc(X, Y)

        ax.scatter(x_res, y_res, c=self.pointcolor, s=0.2, alpha=0.05)
        if filepath:
            fig.savefig(filepath)
        return fig, ax

    def generate_animation(self, filepath="./examples/temp.mp4", **kwargs):
        """Generate animation for the given formulae."""
        initial_points = np.linspace(0, np.max(self.yrange), len(self.yrange))

        X, Y = self._create_mesh()
        if not self.xfunc:
            x_res = self._x_function(X, Y)
        else:
            x_res = self.xfunc(X, Y)

        if not self.yfunc:
            y_res = self._y_function(X, Y)
        else:
            y_res = self.yfunc(X, Y)

        x_points = x_res.reshape(-1, 1)
        y_points = y_res.reshape(-1, 1)
        initial_points = np.stack(
            [np.zeros(x_points.shape), np.zeros(y_points.shape)]
        )
        final_points = np.stack([x_points, y_points])

        slopes = (final_points[1, :] - initial_points[1, :]) / (
            final_points[0, :] - initial_points[0, :]
        )
        intercepts = initial_points[1, :] - slopes * initial_points[0, :]

        points = np.stack(
            [
                np.linspace(
                    initial_points[0, i], final_points[0, i], len(self.yrange)
                )
                for i in range(y_points.shape[0])
            ]
        ).squeeze()
        lines = slopes * points + intercepts

        fig, ax = self._create_fig(subplot_kw={"projection": self.projection})
        ax.set_axis_off()
        ax.set_facecolor(self.background)
        fig.set_facecolor(self.background)
        ax.set_xticks([])
        ax.set_yticks([])
        ims = [
            [
                ax.scatter(
                    points[:, frame], lines[:, frame], c="k", s=0.2, alpha=0.05
                )
            ]
            for frame in range(len(self.yrange))
        ]
        ani = ArtistAnimation(fig, ims, interval=15)
        print("Saving animation")
        ani.save(filepath, writer="ffmpeg")
