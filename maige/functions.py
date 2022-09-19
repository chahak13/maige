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
import numpy as np


def _power1(x):
    return x


def _power2(x):
    return x**2


def _power3(x):
    return x**3


def _arcsin(x):
    x = np.interp(x, (x.min(), x.max()), (-1, 1))
    return np.arcsin(x)


def _arccos(x):
    x = np.interp(x, (x.min(), x.max()), (-1, 1))
    return np.arccos(x)


def _log(x):
    if np.any(x < 0):
        x = np.abs(x) + 1e-6
    return np.log(x)


def _sqrt(x):
    if np.any(x < 0):
        x = np.abs(x)
    return np.sqrt(x)


def _constants(rng=None):
    if rng is None:
        rng = np.random.default_rng()
    return rng.integers(0, 4)


_unary_functions = [
    np.sin,
    np.cos,
    np.tan,
    _arcsin,
    _arccos,
    np.arctan,
    np.sinh,
    np.cosh,
    np.tanh,
    # np.arcsinh,
    # np.arccosh,
    # np.arctanh,
    np.floor,
    np.ceil,
    np.exp,
    _log,
    np.sinc,
    _sqrt,
    _power1,
    _power2,
    _power3,
]

_binary_functions = [
    np.add,
    np.subtract,
    np.multiply,
]


class ExpressionTree:
    """Generate exp tree."""

    def __init__(self, rng=None, depth=0, max_depth=5, variables=[]):
        """Initialize exp tree."""
        self.max_depth = max_depth
        self.depth = depth
        self.rng = rng if rng is not None else np.random.default_rng()
        self.variables = variables
        self.method = None
        self.children = []

    def generate_function(
        self,
        parent,
        node_type,
        weight,
        method,
        var_index,
        value,
    ):
        """Generate function."""
        self.node_type = (
            node_type
            if node_type
            else self.rng.choice(
                np.arange(0, 4), 1, p=[0.001, 0.333, 0.333, 0.333]
            )
        )
        self.weight = weight if weight else _constants(self.rng)
        self.children = []

        if self.depth == self.max_depth:
            self.node_type = 1
        if self.node_type == 0:  # Constant
            self.expr = (
                value
                if value
                else np.ones(self.variables[0].shape) * _constants(self.rng)
            )
        elif self.node_type == 1:  # variable
            self.index = (
                var_index
                if var_index is not None
                else self.rng.integers(0, len(self.variables))
            )
            self.expr = self.variables[self.index]
            # self.expr = index
        elif self.node_type == 2:  # Binary
            self.method = (
                method
                if method is not None
                else self.rng.choice(_binary_functions)
            )
            self.children = [
                ExpressionTree(
                    rng=self.rng,
                    depth=self.depth + 1,
                    max_depth=self.max_depth,
                    variables=self.variables,
                ).generate_function(
                    parent=self,
                    node_type=None,
                    weight=None,
                    method=None,
                    var_index=None,
                    value=None,
                ),
                ExpressionTree(
                    rng=self.rng,
                    depth=self.depth + 1,
                    max_depth=self.max_depth,
                    variables=self.variables,
                ).generate_function(
                    parent=self,
                    node_type=None,
                    weight=None,
                    method=None,
                    var_index=None,
                    value=None,
                ),
            ]
        elif self.node_type == 3:  # Unary
            self.method = (
                method
                if method is not None
                else self.rng.choice(_unary_functions)
            )
            self.children = [
                ExpressionTree(
                    rng=self.rng,
                    depth=self.depth + 1,
                    max_depth=self.max_depth,
                    variables=self.variables,
                ).generate_function(
                    parent=self,
                    node_type=None,
                    weight=None,
                    method=None,
                    var_index=None,
                    value=None,
                ),
            ]
        return self

    def execute(self):
        """Execute exp tree."""
        inputs = []
        for child in self.children:
            inputs.append(child.execute())

        if self.node_type == 0 or self.node_type == 1:
            return self.expr * self.weight
        elif self.node_type == 2:
            return self.method(inputs[0], inputs[1]).real
        elif self.node_type == 3:
            return self.method(inputs[0]).real

    def get_expr_string(self):
        """Get expression in string form."""
        string = ""
        inputs = []
        for child in self.children:
            inputs.append(child.get_expr_string())

        if self.node_type == 0:
            return string + f"{self.expr[0, 0]} * {self.weight}"
        elif self.node_type == 1:
            return string + f"{'x' if self.index==0 else 'y'}"
        elif self.node_type == 2:
            return string + f"{self.method.__name__}({inputs[0]}, {inputs[1]})"
        elif self.node_type == 3:
            return string + f"{self.method.__name__}({inputs[0]})"


if __name__ == "__main__":
    variables = [1, 2]
    rng = np.random.default_rng()
    tree = ExpressionTree(rng, depth=0, max_depth=2, variables=variables)
    tree.generate_function(None, 2, 1, None, None, None)
    expr = tree.execute()
    print(expr)
