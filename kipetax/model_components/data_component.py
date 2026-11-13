"""
Measured (non-spectral) datasets attached to a ReactionModel.

Each dataset is a DataFrame indexed by time with one column per measured
variable. Spectral D-matrices are held separately by
:class:`~kipetax.model_components.spectral_handler.SpectralData`.
"""


class DataSet:
    """One named table of measurements."""

    def __init__(self, name, category=None, data=None, file=None):
        self.name = name
        self.category = category
        self.data = data
        self.file = file

    @property
    def shape(self):
        return None if self.data is None else self.data.shape

    def __repr__(self):
        return (f"DataSet({self.name!r}, category={self.category!r}, "
                f"shape={self.shape})")


class DataBlock:
    """The datasets of one model, keyed by name in the order added."""

    def __init__(self):
        self._sets = {}

    def add_dataset(self, dataset):
        """Store ``dataset``, replacing any earlier one with the same name."""
        self._sets[dataset.name] = dataset

    @property
    def names(self):
        return list(self._sets)

    def __getitem__(self, name):
        return self._sets[name]

    def __iter__(self):
        return iter(self._sets.values())

    def __len__(self):
        return len(self._sets)

    def __repr__(self):
        lines = ['DataBlock:'] + [f'  {ds!r}' for ds in self]
        return '\n'.join(lines)

    __str__ = __repr__
