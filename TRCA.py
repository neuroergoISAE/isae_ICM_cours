"""Classification of SSVEP EEG signals based on the Task-Related Component Analysis method [1]. 

Reference:

    [1] - M. Nakanishi, Y. Wang, X. Chen, Y. -T. Wang, X. Gao, and T.-P. Jung,
          "Enhancing detection of SSVEPs for a high-speed brain speller using 
          task-related component analysis",
          IEEE Trans. Biomed. Eng, 65(1):104-112, 2018.
    
Code based on the Matlab (Matlab sucks) implementation from authors of [1]
(https://github.com/mnakanishi/TRCA-SSVEP). 

Authors: Ludovic Darmet,Juan Jesús Torre,  Giuseppe Ferraro
Mail: Ludovic.DARMET@isae-supaero.fr, Juan-jesus.TORRE-TRESOLS@isae-supaero.fr,
      Giuseppe.FERRARO@isae-supaero.fr
"""


import numpy as np
import scipy.linalg as linalg
import scipy.signal as scp

from sklearn.base import BaseEstimator, ClassifierMixin
from pyriemann.estimation import Covariances
from sklearn.utils.validation import  check_is_fitted

def filterbank(data, sfreq, idx_fb, peaks):
    """
    Filter bank design for decomposing EEG data into sub-band components [1]

    Parameters
    ----------

    data: np.array, shape (trials, channels, samples) or (channels, samples)
        EEG data to be processed

    sfreq: int
        Sampling frequency of the data.

    idx_fb: int
        Index of filters in filter bank analysis

    peaks : list of len (n_classes)
        Frequencies corresponding to the SSVEP components.

    Returns
    -------

    y: np.array, shape (trials, channels, samples)
        Sub-band components decomposed by a filter bank

    Reference:
      [1] M. Nakanishi, Y. Wang, X. Chen, Y. -T. Wang, X. Gao, and T.-P. Jung,
          "Enhancing detection of SSVEPs for a high-speed brain speller using
           task-related component analysis",
          IEEE Trans. Biomed. Eng, 65(1):104-112, 2018.

    Code based on the Matlab implementation from authors of [1]
    (https://github.com/mnakanishi/TRCA-SSVEP).
    """

    # Calibration data comes in batches of trials
    if data.ndim == 3:
        num_chans = data.shape[1]
        num_trials = data.shape[0]

    # Testdata come with only one trial at the time
    elif data.ndim == 2:
        num_chans = data.shape[0]
        num_trials = 1

    sfreq = sfreq / 2

    min_freq = np.min(peaks)
    max_freq = np.max(peaks)
    if max_freq < 40:
        top = 90
    else:
        top = 115
    diff = min_freq
    # Lowcut frequencies for the pass band (depends on the frequencies of SSVEP)
    # No more than 3dB loss in the passband
    passband = [min_freq + x * diff for x in range(7)]

    # At least 40db attenuation in the stopband
    if min_freq - 4 > 0:
        stopband = [
            min_freq - 4 + x * (diff - 2) if x < 3 else min_freq - 4 + x * diff
            for x in range(7)
        ]
    else:
        stopband = [2 + x * (diff - 2) if x < 3 else 2 + x * diff for x in range(7)]

    Wp = [passband[idx_fb] / sfreq, top / sfreq]
    Ws = [stopband[idx_fb] / sfreq, (top + 5) / sfreq]
    N, Wn = scp.cheb1ord(Wp, Ws, 3, 40)  # Chebyshev type I filter order selection.

    B, A = scp.cheby1(N, 0.5, Wn, btype="bandpass")  #  Chebyshev type I filter design

    y = np.zeros(data.shape)
    if num_trials == 1:  # For testdata
        for ch_i in range(num_chans):
            try:
                # The arguments 'axis=0, padtype='odd', padlen=3*(max(len(B),len(A))-1)' correspond
                # to Matlab filtfilt (https://dsp.stackexchange.com/a/47945)
                y[ch_i, :] = scp.filtfilt(
                    B,
                    A,
                    data[ch_i, :],
                    axis=0,
                    padtype="odd",
                    padlen=3 * (max(len(B), len(A)) - 1),
                )
            except Exception as e:
                print(e)
                print(f'Error on channel {ch_i}')
    else:
        for trial_i in range(num_trials):  # Filter each trial sequentially
            for ch_i in range(num_chans):  # Filter each channel sequentially
                y[trial_i, ch_i, :] = scp.filtfilt(
                    B,
                    A,
                    data[trial_i, ch_i, :],
                    axis=0,
                    padtype="odd",
                    padlen=3 * (max(len(B), len(A)) - 1),
                )
    return y

class TRCA(BaseEstimator, ClassifierMixin):
    """
    Parameters
    ----------

    sfreq : float
        Sampling frequency of the data to be analyzed.

     peaks : list of len (n_classes)
        Frequencies corresponding to the SSVEP components. These are
        necessary to construct the filterbank.

    peaks : list of len (n_classes)
        Frequencies corresponding to the SSVEP components. These are
        necessary to design the filterbank bands.

    n_fbands : int, default=5
        Number of sub-bands considered for filterbank analysis.

    downsample: int, default=1
        Factor by which downsample the data. A downsample value of N will result
        on a sampling frequency of (sfreq // N) by taking one sample every N of
        the original data. In the original TRCA paper [1] data are at 250Hz.

    is_ensemble: bool, default=False
        If True, predict on new data using the Ensemble-TRCA method described
        in [1].

    regul : str
        For both methods, regularization to use for covariance matrices estimations.
        Consider 'sch', 'lwf', 'oas' or 'scm' for no regularization.
        In the original implementation from TRCA paper [1], no regularization
        is used. So method='original' and regul='scm' is similar to original
        implementation.


    Attributes
    ----------

    fb_coefs : list of len (n_fb)
        Alpha coefficients for the fusion of the filterbank sub-bands.

    classes_ : ndarray of shape (n_class,)
        Array with the class labels extracted at fit time.

    n_class : int
        Number of unique labels/classes.

    templates_ : ndarray of shape (n_class, n_bands, n_channels, n_samples)
        Template data obtained by averaging all training trials for a given
        class. Each class templates is divided in n_fbands sub-bands extracted
        from the filterbank approach.

    weights_ : ndarray of shape (n_fbands, n_class, n_channels)
        Weight coefficients for the different electrodes which are used
        as spatial filters for the data.

    x_train_ : ndarray of shape ((n_trials, self.n_class, n_samples)
        Calibration data filtered with the first filter of filterbank and
        with spatial filter. Only used to compute TrustScore.
        See:  https://arxiv.org/abs/1805.11783

    """

    def __init__(
        self,
        sfreq,
        peaks,
        n_fbands=5,
        downsample=1,
        is_ensemble=True,
        regul="sch"
    ):
        self.peaks = peaks
        self.n_fbands = n_fbands
        self.downsample = downsample
        self.sfreq = sfreq / self.downsample
        self.peaks = peaks
        self.is_ensemble = is_ensemble
        self.fb_coefs = [(x + 1) ** (-1.25) + 0.25 for x in range(self.n_fbands)]
        self.regul = regul

    def _compute_trca(self, data):
        """
        Computation of TRCA spatial filters.

        Parameters
        ----------

        data: np.array, shape (trials, channels, samples)
            Training data

        Returns
        -------

        W: np.array, shape (channels)
            Weight coefficients for electrodes which can be used as
            a spatial filter.
        """

        # Check if X is a single trial (test data) or not
        if data.ndim == 2:
            data = data[np.newaxis, ...]

        # Get data shape
        n_trials, n_channels, n_samples = data.shape

        X = np.concatenate((data, data), axis=1)

        # Initialize S matrix
        S = np.zeros((n_channels, n_channels))

        # Estimate covariance between every trial and the rest of the trials (excluding itself)
        for trial_i in range(n_trials - 1):
            x1 = np.squeeze(data[trial_i, :, :])

            # Mean centering for the selected trial
            x1 -= np.mean(x1, 0)

            # Select a second trial that is different
            for trial_j in range(trial_i + 1, n_trials):
                x2 = np.squeeze(data[trial_j, :, :])

                # Mean centering for the selected trial
                x2 -= np.mean(x2, 0)

                # # Put the two trials together
                X = np.concatenate((x1, x2))

                if n_channels == 1:
                    X = X.reshape((n_channels, len(X)))

                # Regularized covariance estimate
                cov = Covariances(estimator=self.regul).fit_transform(
                    X[np.newaxis, ...]
                )
                cov = np.squeeze(cov)

                # Compute empirical covariance betwwen the two selected trials and sum it
                if n_channels > 1:
                    S = (
                        S
                        + cov[:n_channels, n_channels:]
                        + cov[n_channels:, :n_channels]
                    )

                else:
                    S = S + cov + cov

        # Concatenate all the trials
        UX = np.zeros((n_channels, n_samples * n_trials))

        for trial_n in range(n_trials):
            UX[:, trial_n * n_samples : (trial_n + 1) * n_samples] = data[
                trial_n, :, :
            ]

        # Mean centering
        UX -= np.mean(UX, 1)[:, None]
        cov = Covariances(estimator=self.regul).fit_transform(UX[np.newaxis, ...])
        Q = np.squeeze(cov)

        
        # Compute eigenvalues and vectors
        lambdas, W = linalg.eig(S, Q, left=True, right=False)

        # Sort eigenvectors by eigenvalue
        arr1inds = lambdas.argsort()
        W = W[:, arr1inds[::-1]]

        return W[:, 0], W

    def fit(self, X, y):
        """
        Extract spatial filters and templates from the given calibration data.

        Parameters
        ----------

        X : ndarray of shape (n_trials, n_channels, n_samples)
            Training data. Trials are grouped by class, divided in n_fbands bands by
            the filterbank approach and then used to calculate weight vectors and
            templates for each class and band.

        y : ndarray of shape (n_trials,)
            Label vector in respect to X.

        Returns
        -------

        self: CCA object
            Instance of classifier.
        """
        # Downsample data
        X = X[:, :, :: self.downsample]

        # Get shape of X and labels
        n_trials, n_channels, n_samples = X.shape

        self.classes_ = np.unique(y)
        self.n_class = len(self.classes_)

        # Initialize the final arrays
        self.templates_ = np.zeros((self.n_class, self.n_fbands, n_channels, n_samples))
        self.weights_ = np.zeros((self.n_fbands, self.n_class, n_channels))
        for class_idx in self.classes_:
            cal_data = X[y == class_idx]  # Select data with a specific label
            # Filterbank approach
            for band_n in range(self.n_fbands):
                # Filter the data and compute TRCA
                filter_data = filterbank(cal_data, self.sfreq, band_n, self.peaks)
                w_best, _ = self._compute_trca(filter_data)

                # Get template by averaging trials and take the best filter for this band
                self.templates_[class_idx, band_n, :, :] = np.mean(filter_data, axis=0)
                self.weights_[band_n, class_idx, :] = w_best

        return self

    def predict(self, X, t_min=0., t_max=None):
        """
        Make predictions on unseen data. The new data observation X will be filtered
        with weights previously extracted and compared to the templates to assess
        similarity with each of them and select a class based on the maximal value.

        Parameters
        ----------

        X : ndarray of shape (n_trials, n_channels, n_samples)
            Testing data. This will be divided in self.n_fbands using the filter- bank approach,
            then it will be transformed by the different spatial filters and compared to the
            previously fit templates according to the selected method for analysis (ensemble or
            not). Finally, correlation scores for all sub-bands of each class will be combined,
            resulting on a single correlation score per class, from which the maximal one is
            identified as the predicted class of the data.

        t_min : int or float, default = 0.
            Time from where to slice the template for correlation. When using testing data that does
            not start from time zero, this argument needs to specify the time from which the testing
            epochs start. Defaults to 0 to start from the beginning of the template.

        t_max : int, float or None, default = None
            End time to slice the template for correlation. When using testing data that does not use
            the full template, this argument needs to specify the point the testing epochs reach, respective
            to the template. Defaults to None to use the template from t_min until the end.
        Returns
        -------

        y_pred : ndarray of shape (n_trials,)
            Prediction vector in respect to X.
        """

        # Check is fit had been called
        check_is_fitted(self)

        # Check if X is a single trial or not
        if X.ndim == 2:
            X = X[np.newaxis, ...]

        # Downsample data
        X = X[:, :, :: self.downsample]

        # Get test data shape
        n_trials, _, n_samples = X.shape

        # Get t_min and t_max (and expected length)
        t_min = int(t_min * self.sfreq)
        if t_max is None:
            t_max = n_samples
        else:
            t_max = int(t_max * self.sfreq)

        expected_len = t_max - t_min

        # Initialize pred array
        y_pred = []

        for trial_n in range(n_trials):
            # Pick trial
            test_data = X[trial_n, :, t_min:t_max]

            # Check the picked trial has the desired len
            if not test_data.shape[-1] == expected_len:
                raise ValueError(f"Invalid trial test length. Expecting an array of {expected_len} samples"
                                 f" from {t_min} to {t_max} respective to the template (sfreq {self.sfreq})")

            # Initialize correlations array
            corr_array = np.zeros((self.n_fbands, self.n_class))

            # Filter the data in the corresponding band
            for band_n in range(self.n_fbands):
                filter_data = filterbank(test_data, self.sfreq, band_n, self.peaks)

                # Compute correlation with all the templates and bands
                for class_idx in range(self.n_class):
                    # Get the corresponding template
                    template = np.squeeze(self.templates_[class_idx, band_n, :, t_min:t_max])

                    if self.is_ensemble:
                        w = np.squeeze(
                            self.weights_[band_n, :, :]
                        ).T  # (n_class, n_channel)
                    else:
                        w = np.squeeze(
                            self.weights_[band_n, class_idx, :]
                        ).T  # (n_channel,)

                    # Compute 2D correlation of spatially filtered testdata with ref
                    r = np.corrcoef(
                        np.dot(filter_data.T, w).flatten(),
                        np.dot(template.T, w).flatten(),
                    )
                    corr_array[band_n, class_idx] = r[0, 1]

            # Fusion for the filterbank analysis
            rho = np.dot(self.fb_coefs, corr_array)

            # Select the maximal value and append to preddictions
            tau = np.argmax(rho)
            y_pred.append(tau)

        return y_pred
