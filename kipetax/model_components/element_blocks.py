"""
Ordered, name-keyed collections of model elements.

A ReactionModel keeps one collection per element kind (``r.parameters``,
``r.components``, ...). Each collection creates its elements, looks them up by
name, and iterates over them in the order they were declared.
"""
from kipetax.model_components.model_components import (
    ModelAlgebraic, ModelComponent, ModelConstant, ModelParameter, ModelState,
)


class ElementCollection:
    """Elements of a single kind, keyed by name in declaration order.

    Subclasses set ``element_type`` to the element class they hold.
    """

    element_type = None

    def __init__(self):
        self._elements = {}

    def add_element(self, name, **options):
        """Create an element of this collection's type and store it.

        Re-declaring an existing name replaces the earlier element.
        """
        element = self.element_type(name, **options)
        self._elements[element.name] = element
        return element

    @property
    def names(self):
        """Element names, in declaration order."""
        return list(self._elements)

    @property
    def values(self):
        """``{name: value}`` for every element."""
        return {name: el.value for name, el in self._elements.items()}

    def __getitem__(self, name):
        return self._elements[name]

    def __contains__(self, name):
        return name in self._elements

    def __iter__(self):
        return iter(self._elements.values())

    def __len__(self):
        return len(self._elements)

    def __str__(self):
        title = type(self).__name__.replace('Block', 's')
        rows = [f"{title}:"]
        rows += [f"  {el.name} = {el.value}" for el in self]
        return '\n'.join(rows) + '\n'

    __repr__ = __str__


class ParameterBlock(ElementCollection):
    """Kinetic parameters (``r.parameters``)."""
    element_type = ModelParameter


class ComponentBlock(ElementCollection):
    """Chemical species (``r.components``)."""
    element_type = ModelComponent


class StateBlock(ElementCollection):
    """Non-concentration differential states (``r.states``)."""
    element_type = ModelState


class AlgebraicBlock(ElementCollection):
    """Algebraic variables (``r.algebraics``)."""
    element_type = ModelAlgebraic


class ConstantBlock(ElementCollection):
    """Fixed constants (``r.constants``)."""
    element_type = ModelConstant
