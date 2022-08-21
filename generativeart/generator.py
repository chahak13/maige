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

    def __init__(self, facecolor="#ffffff", xfunc=None, yfunc=None, projection="polar"):
        """Set variables for the base class Generator."""
        self.facecolor = facecolor
        self.xfunc = xfunc
        self.yfunc = yfunc
        self.projection = projection

    def _create_fig(self, **kwargs):
        fig, ax = plt.subplots(figsize=(7, 7), **kwargs)
        ax.set_axis_off()
        ax.set_facecolor(self.facecolor)
        fig.set_facecolor(self.facecolor)
        ax.set_xticks([])
        ax.set_yticks([])
        return fig, ax

    def _create_mesh(self, **kwargs):
        x = np.arange(-np.pi, np.pi, 0.1)
        y = np.arange(-np.pi, np.pi, 0.1)
        X, Y = np.meshgrid(x, y)
        return X, Y

    def _x_function(self, mesh):
        """
        Return changed for x values of points based on the function.

        Arguments:
        mesh: np.meshgrid Grid

        Returns:
        np.array: Changed X values
        """
        X, Y = mesh
        return X + np.cos(Y)

    def _y_function(self, mesh):
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
