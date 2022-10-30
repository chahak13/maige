import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from maige.generator import Generator
from PIL import Image


class PhotoGenerator(Generator):
    def __init__(self, projection="rectilinear"):
        super().__init__(projection=projection)

    def generate_image(self, photofilepath):
        img = Image.open(photofilepath)
        img = img.convert("L")
        npimg = np.asarray(img).T
        npimg = npimg[::-1, ::-1] / 255
        aspect_ratio = npimg.shape[1] / npimg.shape[0]
        thresh = 0
        cover = 0
        while cover < 0.4:
            thresh += 0.01
            nonzero_coords = (npimg < thresh).nonzero()
            cover = nonzero_coords[0].shape[0] / npimg.size
        thresh -= 0.01
        nonzero_coords = (npimg < thresh).nonzero()
        fig, ax = self._create_fig(figsize=(7, 7 * aspect_ratio))
        ax.scatter(nonzero_coords[0], nonzero_coords[1], c="k", s=1)
        return fig, ax

    def generate_animation(
        self,
        photofilepath,
        savefilepath="./examples/temp.mp4",
        init_cond="linear",
        **kwargs,
    ):
        """Generate animation for the given formulae.

        Args:
            filepath: str
                Path to store the final image at. If not provided, a filename
                is generated based on the random generator and the seed.

            init_cond: str
                Initial condition to start the animation from. Can be one of
                "linear" or "uniform". Default is set to "linear".

                __linear__: When the value is set to "linear", the initial
                position of the points is set as x=x for all x in the range,
                and y=0.

                __uniform__: When the value is set to "uniform", the initial
                position of the points is set at random points selected from
                a uniform distribution across the coordinate ranges.

            **kwargs:
                Keyword arguments passed onwards to matplotlib's scatter
                function.
        """
        # X, Y = self.__create_mesh()
        # if not self._xfunc:
        #     self._xfunc = self.__generate_x_func(X, Y)
        #     x_res = self._xfunc.execute()
        # else:
        #     x_res = self._xfunc(X, Y)

        # if not self._yfunc:
        #     self._yfunc = self.__generate_y_func(X, Y)
        #     y_res = self._yfunc.execute()
        # else:
        #     y_res = self._yfunc(X, Y)

        img = Image.open(photofilepath)
        img = img.convert("L")
        npimg = np.asarray(img).T
        npimg = npimg[::-1, ::-1]

        thresh = 0
        cover = 0
        threshold = 0.3
        with tqdm.tqdm(total=threshold) as pbar:
            while cover < threshold:
                thresh += 0.01
                nonzero_coords = (npimg < thresh).nonzero()
                cover = nonzero_coords[0].shape[0] / npimg.size
                pbar.update(cover - pbar.n)
        thresh -= 0.01
        nonzero_coords = (npimg < thresh).nonzero()
        self._xrange = np.arange(npimg.shape[0])
        self._yrange = np.arange(npimg.shape[1])

        x_points = nonzero_coords[0].reshape(-1, 1)
        y_points = nonzero_coords[1].reshape(-1, 1)
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

        # breakpoint()
        slopes = (final_points[1, :] - initial_points[1, :]) / (
            final_points[0, :] - initial_points[0, :] + 0.0000001
        )
        intercepts = initial_points[1, :] - slopes * initial_points[0, :]

        # z = np.sqrt(initial_points[0, :] ** 2 + initial_points[1, :] ** 2)
        # norm = plt.Normalize(np.min(z), np.max(z))
        aspect_ratio = npimg.shape[1] / npimg.shape[0]
        figsize = (7, 7 * aspect_ratio)
        fig, ax = self._create_fig(
            figsize=figsize, subplot_kw={"projection": self._projection}
        )
        # fig, ax = plt.subplots()
        scat = ax.scatter(
            final_points[0, :],
            final_points[1, :],
            c=self._pointcolor,
            # c=z,
            # norm=norm,
            # cmap="viridis",
            s=0.2,
            # alpha=0.05,
        )

        # import matplotlib

        # cmap = matplotlib.cm.winter
        # colors = cmap(
        #     norm(initial_points[0, :] ** 2 + initial_points[1, :] ** 2)
        # )
        # scat.set_color(
        #     cmap(norm(initial_points[0, :] ** 2 + initial_points[1, :] ** 2))
        # )
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
        pbar = tqdm.tqdm(total=len(self._yrange) + 122)

        def __animate(i):
            next(generator)
            data = generator.send(i)
            scat.set_offsets(data.T)
            # scat.set_color(cmap(norm(data[0, :] ** 2 + data[1, :] ** 2)))
            # scat.set_color(colors)
            pbar.update()
            pbar.refresh()
            return (scat,)

        ani = FuncAnimation(
            fig=fig, func=__animate, blit=True, frames=len(self._yrange) + 120
        )
        if not savefilepath:
            savefilepath = (
                f"{self.random_state['bit_generator']}"
                f"_{self.random_state['state']['state']}.mp4"
            )
        ani.save(
            savefilepath,
            writer="ffmpeg",
            fps=60,
        )
        # self.__save_info("video", filepath)


if __name__ == "__main__":
    g = PhotoGenerator(projection="rectilinear")
    # fig, ax = g.generate_image("/home/chahak/Pictures/result_RGB.jpg")
    # fig, ax = g.generate_image("/home/chahak/Pictures/chahak.jpg")
    # plt.show()
    g.generate_animation(
        "/home/chahak/Pictures/starry_night.jpg",
        "./examples/starry_night_tqdm.mp4",
        init_cond="uniform",
    )
