"""
Settings for kipetax (Pyomo-free, pure Python).
"""


class _AttrDict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value


class Settings:
    """Container for kipetax model and solver settings."""

    def __init__(self, category='model'):
        self.general = _AttrDict(
            simulation_times=None,
            concentration_lb=0.0,
            spectral_lb=0.0,
            add_volume_terms=False,
            scale_parameters=False,
            nfe=20,
            ncp=3,
            scheme='radau',
        )
        self.collocation = _AttrDict(
            nfe=20,
            ncp=3,
            scheme='radau',
        )
        self.simulator = _AttrDict(
            method='collocation',
        )
        self.parameter_estimator = _AttrDict(
            tee=False,
        )
        self.variance_estimator = _AttrDict(
            max_iter=20,
            tolerance=1e-4,
        )

    def __str__(self):
        s = 'Settings\n'
        for section in ['general', 'collocation', 'simulator',
                        'parameter_estimator', 'variance_estimator']:
            if hasattr(self, section):
                s += f'\n{section.capitalize()} Settings:\n'
                for k, v in getattr(self, section).items():
                    s += f'  {k:30s}: {v}\n'
        return s

    def __repr__(self):
        return self.__str__()
