"""
First version for generativeart.

Owner: Chahak Mehta (chahakmehta013 [at] gmail [dot] com)
"""

import numpy as np
import matplotlib.pyplot as plt


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
        X, Y = mesh
        return Y + np.sin(Y)**2

    def generate(self, filepath=""):
        """Generate the image and save it."""
        fig, ax = self._create_fig(subplot_kw={"projection": self.projection})
        mesh = self._create_mesh()
        if not self.xfunc:
            X = self._x_function(mesh)
        else:
            X = self.xfunc(mesh)

        if not self.yfunc:
            Y = self._y_function(mesh)
        else:
            Y = self.yfunc(mesh)

        ax.scatter(X, Y, c='k', s=0.3)
        if filepath:
            fig.savefig(filepath)
        return fig, ax
