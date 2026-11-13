"""
MixtureModel for kipetax.

The non-additive interaction motif of this project appears once more with no
time axis at all: property prediction from a blend. Each component has a property
(an FTIR-like spectrum, a viscosity) and blending mixes them NON-additively. The
same ``w_i w_j`` object that was an adduct band in the spectral kinetics
(:meth:`ReactionModel.interaction`) is here a pairwise blend-interaction term:

    property(blend) = sum_i w_i P_i  +  sum_{i<j} w_i w_j Gamma_ij

For a spectrum ``P_i`` is a pure-component spectrum and ``Gamma_ij`` an
interaction band that appears only for an incompatible pair. For viscosity the
same form is the Grunberg-Nissan mixing rule on ``ln eta``: ``P_i = ln eta_i``
and ``Gamma_ij = d_ij`` a scalar interaction parameter.

Both models are LINEAR in their parameters given the blend fractions, and they
share one design matrix ``[w_i | w_i w_j]``, so calibration is an ordinary
least-squares fit and prediction of a new recipe is a matrix product. This class
is therefore small and deliberately not a dynamic model: it does not integrate
anything. It is here so the interaction idea has one home across the dynamic and
static cases.
"""
from itertools import combinations

import numpy as np


class MixtureModel:
    """Linear-in-parameters blend-property model with pairwise interactions.

    Parameters
    ----------
    n_components : int
        Number of pure components.
    interaction : bool
        If True (default) include the ``w_i w_j`` pairwise columns; if False the
        model is purely additive (Kay's rule / ideal mixing).

    Examples
    --------
    >>> mm = MixtureModel(n_components=4)
    >>> mm.fit(W_train, D_train)        # D_train: (n_blends, n_wavelengths)
    >>> D_pred = mm.predict(W_test)
    >>> mm.interaction_props            # (n_pairs, n_wavelengths): the Gamma_ij
    >>> w = mm.infer_composition(a_spectrum)   # inverse: spectrum -> fractions
    """

    def __init__(self, n_components, interaction=True):
        self.n_components = int(n_components)
        self.interaction = bool(interaction)
        self.pairs = list(combinations(range(self.n_components), 2))
        self.component_props = None       # (n_components, n_features)
        self.interaction_props = None     # (n_pairs, n_features) or None
        self._coef = None                 # stacked (n_components [+ n_pairs], n_features)

    # -- design --

    def design(self, W):
        """Blend-fraction design matrix ``[w_i | w_i w_j]``.

        Parameters
        ----------
        W : ndarray, shape (n_blends, n_components)
            Blend fractions (rows need not be normalized, but usually sum to 1).

        Returns
        -------
        ndarray, shape (n_blends, n_components [+ n_pairs])
        """
        W = np.atleast_2d(np.asarray(W, dtype=float))
        cols = [W]
        if self.interaction:
            cols.append(np.stack([W[:, i] * W[:, j] for (i, j) in self.pairs],
                                 axis=1))
        return np.concatenate(cols, axis=1)

    # -- calibration --

    def fit(self, W, Y):
        """Least-squares calibration.

        Parameters
        ----------
        W : ndarray, shape (n_blends, n_components)
        Y : ndarray, shape (n_blends,) or (n_blends, n_features)
            Measured blend property (a scalar per blend, or a spectrum).

        Returns
        -------
        self
        """
        G = self.design(W)
        Y = np.asarray(Y, dtype=float)
        coef, *_ = np.linalg.lstsq(G, Y, rcond=None)
        self._coef = coef
        nc = self.n_components
        self.component_props = coef[:nc]
        self.interaction_props = coef[nc:] if self.interaction else None
        return self

    def predict(self, W):
        """Predict the blend property for new fractions."""
        if self._coef is None:
            raise ValueError("call fit() first")
        return self.design(W) @ self._coef

    def interaction_parameter(self, i, j):
        """The fitted interaction property for the pair (i, j).

        For a scalar-property model this is the Grunberg-Nissan ``d_ij``; for a
        spectral model it is the interaction band ``Gamma_ij`` (a vector).
        """
        if not self.interaction:
            raise ValueError("this model has no interaction terms")
        key = (i, j) if i < j else (j, i)
        return self.interaction_props[self.pairs.index(key)]

    # -- inverse --

    def infer_composition(self, y_target, normalize=True):
        """Infer blend fractions from a measured property (the inverse problem).

        Solves ``min_w || design(w) @ coef - y_target ||^2`` subject to
        ``w >= 0`` and ``sum(w) = 1``. Only meaningful for a rich property such
        as a spectrum, where the fractions are identifiable.

        Parameters
        ----------
        y_target : ndarray, shape (n_features,)
            A single measured property vector.
        normalize : bool
            Enforce the simplex constraint ``sum(w) = 1``.

        Returns
        -------
        ndarray, shape (n_components,)
            The recovered blend fractions.
        """
        from scipy.optimize import minimize

        if self._coef is None:
            raise ValueError("call fit() first")
        y_target = np.asarray(y_target, dtype=float).squeeze()
        if y_target.ndim != 1:
            raise ValueError("infer_composition expects a single property "
                             f"vector; got shape {y_target.shape}")
        nc = self.n_components

        # Warm start: additive least squares onto the component properties.
        w0 = np.clip(np.linalg.lstsq(self.component_props.T, y_target,
                                     rcond=None)[0], 0, None)
        s = w0.sum()
        w0 = w0 / s if s > 1e-9 else np.full(nc, 1.0 / nc)

        def obj(w):
            return float(np.sum((self.design(w) @ self._coef - y_target) ** 2))

        constraints = ([{'type': 'eq', 'fun': lambda w: w.sum() - 1.0}]
                       if normalize else [])
        res = minimize(obj, w0, method='SLSQP', bounds=[(0, 1)] * nc,
                       constraints=constraints)
        return res.x
