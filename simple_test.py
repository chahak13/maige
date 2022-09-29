import matplotlib.pyplot as plt
import numpy as np

from maige.generator import Generator

# xfunc = lambda X, Y: X + np.arctan(Y) + np.log(np.abs(Y))
# yfunc = lambda X, Y: Y + np.sin(X) + np.log((X + Y) ** 2)
# xfunc = lambda X, Y: X + np.sin(X**2)
# yfunc = lambda X, Y: Y + np.sin(Y**2)
xfunc = lambda X, Y: X - Y - 1 * 0
yfunc = lambda X, Y: np.sinc(X) - 6
xrange = np.arange(-np.pi, np.pi, 0.01)
yrange = np.arange(-np.pi, np.pi, 0.01)
# yrange = np.linspace(np.pi, np.pi / 2, 2)
# g = Generator(pointcolor="#FF1818", xfunc=xfunc, yfunc=yfunc)
g = Generator(
    pointcolor="#000000",
    projection="polar",
    # xfunc=xfunc,
    # yfunc=yfunc,
    xrange=xrange,
    yrange=yrange,
    seed=2,
)
# fig, ax = g.generate_image()
g.generate_animation(filepath="./examples/color_test.mp4", init_cond="uniform")
# del g
# xrange = np.arange(-np.pi + 1, np.pi + 1, 0.01)
# yrange = np.arange(-np.pi + 1, np.pi + 1, 0.01)
# g2 = Generator(
#     pointcolor="#3120E0",
#     xfunc=xfunc,
#     yfunc=yfunc,
#     xrange=xrange,
#     yrange=yrange,
# )
# ims2 = g2.generate_animation()
# g2._save_anim(ims1 + ims2)
# plt.show()
# fig.savefig("./examples/curiosity_1.png", dpi=450)
