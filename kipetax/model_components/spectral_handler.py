"""
Spectral data handler for kipetax (Pyomo-free).

Manages spectral (D-matrix) data with optional preprocessing methods.
"""
import numpy as np
import pandas as pd


class SpectralData:
    """Handler for spectral absorption data (D matrix)."""

    def __init__(self, name, file=None, remove_negatives=False):
        self.name = name
        self.file = file
        self.D = None  # DataFrame: rows=times, cols=wavelengths
        self._remove_negatives = remove_negatives

    def add_data(self, dataframe):
        """Add a DataFrame as spectral data."""
        self.D = dataframe.copy()
        if self._remove_negatives:
            self.D = self.D.clip(lower=0)

    @property
    def times(self):
        if self.D is not None:
            return self.D.index.tolist()
        return []

    @property
    def wavelengths(self):
        if self.D is not None:
            return self.D.columns.tolist()
        return []

    def msc(self, reference=None):
        """Multiplicative scatter correction."""
        if self.D is None:
            raise ValueError("No data loaded")
        D = self.D.values
        if reference is None:
            reference = D.mean(axis=0)
        corrected = np.zeros_like(D)
        for i, row in enumerate(D):
            coeffs = np.polyfit(reference, row, 1)
            corrected[i] = (row - coeffs[1]) / coeffs[0]
        self.D = pd.DataFrame(corrected, index=self.D.index, columns=self.D.columns)
        return self

    def snv(self):
        """Standard normal variate normalization."""
        if self.D is None:
            raise ValueError("No data loaded")
        D = self.D.values
        mean = D.mean(axis=1, keepdims=True)
        std = D.std(axis=1, keepdims=True)
        corrected = (D - mean) / std
        self.D = pd.DataFrame(corrected, index=self.D.index, columns=self.D.columns)
        return self

    def savitzky_golay(self, window_length=11, polyorder=2, deriv=0):
        """Apply Savitzky-Golay filter."""
        from scipy.signal import savgol_filter
        if self.D is None:
            raise ValueError("No data loaded")
        D = self.D.values
        corrected = savgol_filter(D, window_length=window_length,
                                  polyorder=polyorder, deriv=deriv, axis=1)
        self.D = pd.DataFrame(corrected, index=self.D.index, columns=self.D.columns)
        return self

    def __repr__(self):
        shape = self.D.shape if self.D is not None else None
        return f"SpectralData(name={self.name!r}, shape={shape})"
