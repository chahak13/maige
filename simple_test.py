import numpy as np
import matplotlib.pyplot as plt

from generativeart.generator import Generator

xfunc = lambda X, Y: X + Y * np.sin(X**2)
yfunc = lambda X, Y: Y + X * np.cos(Y**2)
# g = Generator(pointcolor="#FF1818", xfunc=xfunc, yfunc=yfunc)
g = Generator(pointcolor="#000000", projection="rectilinear")
# fig, ax = g.generate_image()
g.generate_animation(filepath="./examples/t.mp4")
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
