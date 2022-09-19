# maige
Maige is a package to generate images using mathematical functions, written in pure python. It uses numpy and matplotlib to create images and animations for different formulae. A user can provide their own functions that they would like to plot or use the inbuilt random function generator to generate outputs. A random algebraic function is generated using an Expression Tree that uses various functions provided by numpy. The user can also provide a set of points that they want to plot, instead of using the default set of points. Since the library is built on matplotlib, it can project the output in any projection that is accepted by `matplotlib`.

## Usage
### Basic Usage

``` python
from maige.generator import Generator
g = Generator()
g.generate_image()
```

<!-- ![](./examples/ghost.png) -->
<img src="https://raw.githubusercontent.com/chahak13/maige/main/examples/ghost.png" width="50%">

### Projection

``` python
from maige.generator import Generator
g = Generator(projection="rectilinear")
g.generate_image()
```

<!-- ![](./examples/rectilinear.png) -->
<img src="https://raw.githubusercontent.com/chahak13/maige/main/examples/rectilinear.png" width="50%">

### Custom Function

``` python
from maige.generator import Generator
xfunc = lambda X, Y: X**2 * np.sin(Y**2)
yfunc = lambda X, Y: Y**3 - np.cos(X**2)
g = Generator(
    projection="rectilinear",
    xfunc=xfunc,
    yfunc=yfunc,
)
fig, ax = g.generate_image("./examples/custom_func.png")
```
<!-- ![](./examples/custom_func.png) -->
<img src="https://raw.githubusercontent.com/chahak13/maige/main/examples/custom_func.png" width="50%">

### Custom Range

``` python
xfunc = lambda X, Y: X**2 * np.sin(Y**2)
yfunc = lambda X, Y: Y**3 - np.cos(X**2)
xrange = np.arange(0, np.pi, 0.01)
yrange = np.arange(0, np.pi, 0.01)
g = Generator(
    projection="polar",
    xfunc=xfunc,
    yfunc=yfunc,
    xrange=xrange,
    yrange=yrange,
)
fig, ax = g.generate_image("./examples/custom_range.png")
```

<!-- ![](./examples/custom_range.png) -->
<img src="https://raw.githubusercontent.com/chahak13/maige/main/examples/custom_range.png" width="50%">

### Color

``` python
g = Generator(
    pointcolor="#000000",
    background="#FA7070",
    projection="polar",
)
fig, ax = g.generate_image("./examples/custom_color.png")
```
<!-- ![](./examples/custom_color.png) -->
<img src="https://raw.githubusercontent.com/chahak13/maige/main/examples/custom_color.png" width="50%">

### Animation

``` python
xfunc = lambda X, Y: X**2 * np.sin(Y**2)
yfunc = lambda X, Y: Y**3 - np.cos(X**2)
xrange = np.arange(0, np.pi, 0.01)
yrange = np.arange(0, np.pi, 0.01)
g = Generator(
    pointcolor="#ffffff",
    background="#000000",
    projection="polar",
    xfunc=xfunc,
    yfunc=yfunc,
)
g.generate_animation("./examples/anim_example.gif", init_cond="uniform")
```

<!-- ![](./examples/anim_example_compressed.gif) -->
<img src="https://raw.githubusercontent.com/chahak13/maige/main/examples/anim_example_compressed.gif" width="50%">

### Reproducibility

Images and animations can be reproduced by using the JSON stored on the first creation. One can also pass an integer seed to reproduce the same designs over multiple runs.
``` python
from maige.generator import Generator

g = Generator(
    seed="./examples/rectilinear.json",
)
fig, ax = g.generate_image("./examples/rectilinear_2.png")
```

## Installation

### PyPI
`maige` can be installed directly from PyPI by using `pip` or `pipenv`

``` shell
pip install maige
```
or

``` shell
pipenv install maige
```


### Source
`Pipfile` and `requirements.txt` are provided to install virtual environment using `pipenv` or `pip`. This can be done by following steps:

``` shell
$ git clone https://github.com/chahak13/maige.git
$ cd maige
$ pipenv shell
$ pipenv install
```

If you're `pip` instead of `pipenv`, change the last command to `pip install -r requirements.txt`.
