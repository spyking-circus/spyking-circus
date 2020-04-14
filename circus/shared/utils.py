# -*- coding: utf-8 -*-
import warnings, logging
warnings.filterwarnings("ignore")
import os, sys, time, types, tqdm
import numpy as np
import scipy.sparse as sp
from math import log, sqrt
from scipy import linalg
import scipy.interpolate
from scipy.stats import gamma
import numpy, os, tempfile
import scipy.linalg, scipy.optimize, cPickle, socket, tempfile, shutil, scipy.ndimage.filters, scipy.signal
import six

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import h5py

# Warning: be careful while importing other circus modules (avoid circular imports)!
from circus.shared.mpi import gather_array, all_gather_array, comm, SHARED_MEMORY
from circus.shared.messages import print_and_log

logger = logging.getLogger(__name__)

import circus
from distutils.version import StrictVersion
from scipy.optimize import brenth, minimize


def largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    indices = np.argpartition(ary, -n)[-n:]
    indices = indices[np.argsort(-ary[indices])]
    return indices

def test_patch_for_similarities(params, extension):

    import circus.shared.files as io

    file_out_suff = params.get('data', 'file_out_suff')
    template_file = file_out_suff + '.templates%s.hdf5' % extension
    if os.path.exists(template_file):
        version = io.load_data(params, 'version', extension)
    else:
        print_and_log(['No templates found! Check suffix?'], 'error', logger)
        sys.exit(0)

    if version is not None:
        if StrictVersion(version) >= StrictVersion('0.6.0'):
            return True
    else:
        print_and_log(["Version is below 0.6.0"], 'debug', logger)
        return False


def test_if_support(params, extension):
    file_out_suff = params.get('data', 'file_out_suff')
    if os.path.exists(file_out_suff + '.templates%s.hdf5' % extension):
        myfile = h5py.File(file_out_suff + '.templates%s.hdf5' % extension, 'r', libver='earliest')
        return 'supports' in myfile
    else:
        return False

def test_if_purity(params, extension):
    file_out_suff = params.get('data', 'file_out_suff')
    if os.path.exists(file_out_suff + '.templates%s.hdf5' % extension):
        myfile = h5py.File(file_out_suff + '.templates%s.hdf5' % extension, 'r', libver='earliest')
        return 'purity' in myfile
    else:
        return False


def indices_for_dead_times(start, end):
    lens = end - start
    np.cumsum(lens, out=lens)
    i = np.ones(lens[-1], dtype=numpy.int64)
    i[0] = start[0]
    i[lens[:-1]] += start[1:]
    i[lens[:-1]] -= end[:-1]
    np.cumsum(i, out=i)
    return i


def apply_patch_for_similarities(params, extension):

    if not test_patch_for_similarities(params, extension):

        import circus.shared.files as io

        file_out_suff = params.get('data', 'file_out_suff')
        hdf5_compress = params.getboolean('data', 'hdf5_compress')
        blosc_compress = params.getboolean('data', 'blosc_compress')
        N_tm = io.load_data(params, 'nb_templates', extension)
        N_half = int(N_tm // 2)
        N_t = params.getint('detection', 'N_t')
        duration = 2 * N_t - 1

        if comm.rank == 0:
            print_and_log(["Fixing overlaps from 0.5.XX..."], 'default', logger)

        maxlag = numpy.zeros((N_half, N_half), dtype=numpy.int32)
        maxoverlap = numpy.zeros((N_half, N_half), dtype=numpy.float32)

        to_explore = numpy.arange(N_half - 1)[comm.rank::comm.size]

        if comm.rank == 0:
            to_explore = get_tqdm_progressbar(params, to_explore)

        if not SHARED_MEMORY:
            over_x, over_y, over_data, over_shape = io.load_data(params, 'overlaps-raw', extension=extension)
        else:
            over_x, over_y, over_data, over_shape = io.load_data_memshared(params, 'overlaps-raw', extension=extension)
            
        for i in to_explore:

            idx = numpy.where((over_x >= i*N_tm+i+1) & (over_x < (i*N_tm+N_half)))[0]
            local_x = over_x[idx] - (i*N_tm+i+1)
            data = numpy.zeros((N_half - (i + 1), duration), dtype=numpy.float32)
            data[local_x, over_y[idx]] = over_data[idx]
            maxlag[i, i+1:] = N_t - numpy.argmax(data, 1)
            maxlag[i+1:, i] = -maxlag[i, i+1:]
            maxoverlap[i, i+1:] = numpy.max(data, 1)
            maxoverlap[i+1:, i] = maxoverlap[i, i+1:]

        # Now we need to sync everything across nodes.
        maxlag = gather_array(maxlag, comm, 0, 1, 'int32', compress=blosc_compress)

        if comm.rank == 0:
            maxlag = maxlag.reshape(comm.size, N_half, N_half)
            maxlag = numpy.sum(maxlag, 0)

        maxoverlap = gather_array(maxoverlap, comm, 0, 1, 'float32', compress=blosc_compress)
        if comm.rank == 0:
            maxoverlap = maxoverlap.reshape(comm.size, N_half, N_half)
            maxoverlap = numpy.sum(maxoverlap, 0)

        if comm.rank == 0:
            myfile2 = h5py.File(file_out_suff + '.templates%s.hdf5' % extension, 'r+', libver='earliest')

            for key in ['maxoverlap', 'maxlag', 'version']:
                if key in myfile2.keys():
                    myfile2.pop(key)

            myfile2.create_dataset('version', data=numpy.array(circus.__version__.split('.'), dtype=numpy.int32))
            if hdf5_compress:
                myfile2.create_dataset('maxlag',  data=maxlag, compression='gzip')
                myfile2.create_dataset('maxoverlap', data=maxoverlap, compression='gzip')
            else:
                myfile2.create_dataset('maxlag',  data=maxlag)
                myfile2.create_dataset('maxoverlap', data=maxoverlap)
            myfile2.close()


def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        sys.stdout.flush()
        choice = raw_input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")


def get_shared_memory_flag(params):
    """Get parallel HDF5 flag.

    Argument
    --------
    params: dict
        Dictionary of parameters.

    Return
    ------
    flag: bool
        True if parallel HDF5 is available and the user want to use it.
    """
    flag = SHARED_MEMORY and params.getboolean('data', 'shared_memory')

    return flag


def get_parallel_hdf5_flag(params):
    """Get parallel HDF5 flag.

    Argument
    --------
    params: dict
        Dictionary of parameters.

    Return
    ------
    flag: bool
        True if parallel HDF5 is available and the user want to use it.
    """

    flag = h5py.get_config().mpi and params.getboolean('data', 'parallel_hdf5')

    return flag


def purge(file, pattern):
    dir = os.path.dirname(os.path.abspath(file))
    for f in os.listdir(dir):
        if f.find(pattern) > -1:
            os.remove(os.path.join(dir, f))
    if comm.rank == 0:
        print_and_log(['Removing %s for directory %s' % (pattern, dir)], 'debug', logger)


def get_tqdm_progressbar(params, iterator):
    sys.stderr.flush()
    show_bars = params.getboolean('data', 'status_bars')
    if show_bars:
        return tqdm.tqdm(iterator, bar_format='{desc}{percentage:3.0f}%|{bar}|[{elapsed}<{remaining}, {rate_fmt}]', ncols=66)
    else:
        return iterator


def get_whitening_matrix(X, fudge=1e-15):
    sigma = np.dot(X.T, X) / X.shape[0]
    u, s, _ = linalg.svd(sigma)
    W = np.dot(np.dot(u, np.diag(1. / np.sqrt(s + fudge))), u.T)
    return W


def check_is_fitted(estimator, attributes, msg=None, all_or_any=all):
    """Perform is_fitted validation for estimator.
    Checks if the estimator is fitted by verifying the presence of
    "all_or_any" of the passed attributes and raises a NotFittedError with the
    given message.
    Parameters
    ----------
    estimator : estimator instance.
        estimator instance for which the check is performed.
    attributes : attribute name(s) given as string or a list/tuple of strings
        Eg. : ["coef_", "estimator_", ...], "coef_"
    msg : string
        The default error message is, "This %(name)s instance is not fitted
        yet. Call 'fit' with appropriate arguments before using this method."
        For custom messages if "%(name)s" is present in the message string,
        it is substituted for the estimator name.
        Eg. : "Estimator, %(name)s, must be fitted before sparsifying".
    all_or_any : callable, {all, any}, default all
        Specify whether all or any of the given attributes must exist.
    """
    if msg is None:
        msg = ("This %(name)s instance is not fitted yet. Call 'fit' with "
               "appropriate arguments before using this method.")

    if not hasattr(estimator, 'fit'):
        raise TypeError("%s is not an estimator instance." % estimator)

    if not isinstance(attributes, (list, tuple)):
        attributes = [attributes]

    if not all_or_any([hasattr(estimator, attr) for attr in attributes]):
        raise NotFittedError(msg % {'name': type(estimator).__name__})


def _shape_repr(shape):
    """Return a platform independent representation of an array shape
    Under Python 2, the `long` type introduces an 'L' suffix when using the
    default %r format for tuples of integers (typically used to store the shape
    of an array).
    Under Windows 64 bit (and Python 2), the `long` type is used by default
    in numpy shapes even when the integer dimensions are well below 32 bit.
    The platform specific type causes string messages or doctests to change
    from one platform to another which is not desirable.
    Under Python 3, there is no more `long` type so the `L` suffix is never
    introduced in string representation.
    >>> _shape_repr((1, 2))
    '(1, 2)'
    >>> one = 2 ** 64 / 2 ** 64  # force an upcast to `long` under Python 2
    >>> _shape_repr((one, 2 * one))
    '(1, 2)'
    >>> _shape_repr((1,))
    '(1,)'
    >>> _shape_repr(())
    '()'
    """
    if len(shape) == 0:
        return "()"
    joined = ", ".join("%d" % e for e in shape)
    if len(shape) == 1:
        # special notation for singleton tuples
        joined += ','
    return "(%s)" % joined


def _num_samples(x):
    """Return number of samples in array-like x."""
    if hasattr(x, 'fit'):
        # Don't get num_samples from an ensembles length!
        raise TypeError('Expected sequence or array-like, got '
                        'estimator %s' % x)
    if not hasattr(x, '__len__') and not hasattr(x, 'shape'):
        if hasattr(x, '__array__'):
            x = np.asarray(x)
        else:
            raise TypeError("Expected sequence or array-like, got %s" %
                            type(x))
    if hasattr(x, 'shape'):
        if len(x.shape) == 0:
            raise TypeError("Singleton array %r cannot be considered"
                            " a valid collection." % x)
        return x.shape[0]
    else:
        return len(x)


def check_consistent_length(*arrays):
    """Check that all arrays have consistent first dimensions.
    Checks whether all objects in arrays have the same shape or length.
    Parameters
    ----------
    *arrays : list or tuple of input objects.
        Objects that will be checked for consistent length.
    """

    uniques = np.unique([_num_samples(X) for X in arrays if X is not None])
    if len(uniques) > 1:
        raise ValueError("Found arrays with inconsistent numbers of samples: "
                         "%s" % str(uniques))


def _assert_all_finite(X):
    """Like assert_all_finite, but only for ndarray."""
    X = np.asanyarray(X)
    # First try an O(n) time, O(1) space solution for the common case that
    # everything is finite; fall back to O(n) space np.isfinite to prevent
    # false positives from overflow in sum method.
    if (X.dtype.char in np.typecodes['AllFloat'] and not np.isfinite(X.sum())
            and not np.isfinite(X).all()):
        raise ValueError("Input contains NaN, infinity"
                         " or a value too large for %r." % X.dtype)


def as_float_array(X, copy=True, force_all_finite=True):
    """Converts an array-like to an array of floats
    The new dtype will be np.float32 or np.float64, depending on the original
    type. The function can create a copy or modify the argument depending
    on the argument copy.
    Parameters
    ----------
    X : {array-like, sparse matrix}
    copy : bool, optional
        If True, a copy of X will be created. If False, a copy may still be
        returned if X's dtype is not a floating point type.
    force_all_finite : boolean (default=True)
        Whether to raise an error on np.inf and np.nan in X.
    Returns
    -------
    XT : {array, sparse matrix}
        An array of type np.float
    """
    if isinstance(X, np.matrix) or (not isinstance(X, np.ndarray)
                                    and not sp.issparse(X)):
        return check_array(X, ['csr', 'csc', 'coo'], dtype=np.float64,
                           copy=copy, force_all_finite=force_all_finite,
                           ensure_2d=False)
    elif sp.issparse(X) and X.dtype in [np.float32, np.float64]:
        return X.copy() if copy else X
    elif X.dtype in [np.float32, np.float64]:  # is numpy array
        return X.copy('F' if X.flags['F_CONTIGUOUS'] else 'C') if copy else X
    else:
        return X.astype(np.float32 if X.dtype == np.int32 else np.float64)


def check_array(array, accept_sparse=None, dtype="numeric", order=None,
                copy=False, force_all_finite=True, ensure_2d=True,
                allow_nd=False, ensure_min_samples=1, ensure_min_features=1,
                warn_on_dtype=False, estimator=None):
    """Input validation on an array, list, sparse matrix or similar.
    By default, the input is converted to an at least 2nd numpy array.
    If the dtype of the array is object, attempt converting to float,
    raising on failure.
    Parameters
    ----------
    array : object
        Input object to check / convert.
    accept_sparse : string, list of string or None (default=None)
        String[s] representing allowed sparse matrix formats, such as 'csc',
        'csr', etc.  None means that sparse matrix input will raise an error.
        If the input is sparse but not in the allowed format, it will be
        converted to the first listed format.
    dtype : string, type, list of types or None (default="numeric")
        Data type of result. If None, the dtype of the input is preserved.
        If "numeric", dtype is preserved unless array.dtype is object.
        If dtype is a list of types, conversion on the first type is only
        performed if the dtype of the input is not in the list.
    order : 'F', 'C' or None (default=None)
        Whether an array will be forced to be fortran or c-style.
    copy : boolean (default=False)
        Whether a forced copy will be triggered. If copy=False, a copy might
        be triggered by a conversion.
    force_all_finite : boolean (default=True)
        Whether to raise an error on np.inf and np.nan in X.
    ensure_2d : boolean (default=True)
        Whether to make X at least 2d.
    allow_nd : boolean (default=False)
        Whether to allow X.ndim > 2.
    ensure_min_samples : int (default=1)
        Make sure that the array has a minimum number of samples in its first
        axis (rows for a 2D array). Setting to 0 disables this check.
    ensure_min_features : int (default=1)
        Make sure that the 2D array has some minimum number of features
        (columns). The default value of 1 rejects empty datasets.
        This check is only enforced when the input data has effectively 2
        dimensions or is originally 1D and ``ensure_2d`` is True. Setting to 0
        disables this check.
    warn_on_dtype : boolean (default=False)
        Raise DataConversionWarning if the dtype of the input data structure
        does not match the requested dtype, causing a memory copy.
    estimator : str or estimator instance (default=None)
        If passed, include the name of the estimator in warning messages.
    Returns
    -------
    X_converted : object
        The converted and validated X.
    """
    if isinstance(accept_sparse, str):
        accept_sparse = [accept_sparse]

    # store whether originally we wanted numeric dtype
    dtype_numeric = dtype == "numeric"

    dtype_orig = getattr(array, "dtype", None)
    if not hasattr(dtype_orig, 'kind'):
        # not a data type (e.g. a column named dtype in a pandas DataFrame)
        dtype_orig = None

    if dtype_numeric:
        if dtype_orig is not None and dtype_orig.kind == "O":
            # if input is object, convert to float.
            dtype = np.float64
        else:
            dtype = None

    if isinstance(dtype, (list, tuple)):
        if dtype_orig is not None and dtype_orig in dtype:
            # no dtype conversion required
            dtype = None
        else:
            # dtype conversion required. Let's select the first element of the
            # list of accepted types.
            dtype = dtype[0]

    if estimator is not None:
        if isinstance(estimator, six.string_types):
            estimator_name = estimator
        else:
            estimator_name = estimator.__class__.__name__
    else:
        estimator_name = "Estimator"
    context = " by %s" % estimator_name if estimator is not None else ""

    if sp.issparse(array):
        array = _ensure_sparse_format(array, accept_sparse, dtype, copy,
                                      force_all_finite)
    else:
        array = np.array(array, dtype=dtype, order=order, copy=copy)

        if ensure_2d:
            if array.ndim == 1:
                if ensure_min_samples >= 2:
                    raise ValueError("%s expects at least 2 samples provided "
                                     "in a 2 dimensional array-like input"
                                     % estimator_name)
                warnings.warn(
                    "Passing 1d arrays as data is deprecated in 0.17 and will"
                    "raise ValueError in 0.19. Reshape your data either using "
                    "X.reshape(-1, 1) if your data has a single feature or "
                    "X.reshape(1, -1) if it contains a single sample.",
                    DeprecationWarning)
            array = np.atleast_2d(array)
            # To ensure that array flags are maintained
            array = np.array(array, dtype=dtype, order=order, copy=copy)

        # make sure we acually converted to numeric:
        if dtype_numeric and array.dtype.kind == "O":
            array = array.astype(np.float64)
        if not allow_nd and array.ndim >= 3:
            raise ValueError("Found array with dim %d. %s expected <= 2."
                             % (array.ndim, estimator_name))
        if force_all_finite:
            _assert_all_finite(array)

    shape_repr = _shape_repr(array.shape)
    if ensure_min_samples > 0:
        n_samples = _num_samples(array)
        if n_samples < ensure_min_samples:
            raise ValueError("Found array with %d sample(s) (shape=%s) while a"
                             " minimum of %d is required%s."
                             % (n_samples, shape_repr, ensure_min_samples,
                                context))

    if ensure_min_features > 0 and array.ndim == 2:
        n_features = array.shape[1]
        if n_features < ensure_min_features:
            raise ValueError("Found array with %d feature(s) (shape=%s) while"
                             " a minimum of %d is required%s."
                             % (n_features, shape_repr, ensure_min_features,
                                context))

    if warn_on_dtype and dtype_orig is not None and array.dtype != dtype_orig:
        msg = ("Data with input dtype %s was converted to %s%s."
               % (dtype_orig, array.dtype, context))
        warnings.warn(msg, DataConversionWarning)
    return array


class PCA(object):
    """Principal component analysis (PCA)
    Linear dimensionality reduction using Singular Value Decomposition of the
    data and keeping only the most significant singular vectors to project the
    data to a lower dimensional space.
    This implementation uses the scipy.linalg implementation of the singular
    value decomposition. It only works for dense arrays and is not scalable to
    large dimensional data.
    The time complexity of this implementation is ``O(n ** 3)`` assuming
    n ~ n_samples ~ n_features.
    Read more in the :ref:`User Guide <PCA>`.
    Parameters
    ----------
    n_components : int, None or string
        Number of components to keep.
        if n_components is not set all components are kept::
            n_components == min(n_samples, n_features)
        if n_components == 'mle', Minka\'s MLE is used to guess the dimension
        if ``0 < n_components < 1``, select the number of components such that
        the amount of variance that needs to be explained is greater than the
        percentage specified by n_components
    copy : bool
        If False, data passed to fit are overwritten and running
        fit(X).transform(X) will not yield the expected results,
        use fit_transform(X) instead.
    whiten : bool, optional
        When True (False by default) the `components_` vectors are divided
        by n_samples times singular values to ensure uncorrelated outputs
        with unit component-wise variances.
        Whitening will remove some information from the transformed signal
        (the relative variance scales of the components) but can sometime
        improve the predictive accuracy of the downstream estimators by
        making there data respect some hard-wired assumptions.
    Attributes
    ----------
    components_ : array, [n_components, n_features]
        Principal axes in feature space, representing the directions of
        maximum variance in the data.
    explained_variance_ratio_ : array, [n_components]
        Percentage of variance explained by each of the selected components.
        If ``n_components`` is not set then all components are stored and the
        sum of explained variances is equal to 1.0
    mean_ : array, [n_features]
        Per-feature empirical mean, estimated from the training set.
    n_components_ : int
        The estimated number of components. Relevant when n_components is set
        to 'mle' or a number between 0 and 1 to select using explained
        variance.
    noise_variance_ : float
        The estimated noise covariance following the Probabilistic PCA model
        from Tipping and Bishop 1999. See "Pattern Recognition and
        Machine Learning" by C. Bishop, 12.2.1 p. 574 or
        http://www.miketipping.com/papers/met-mppca.pdf. It is required to
        computed the estimated data covariance and score samples.
    Notes
    -----
    For n_components='mle', this class uses the method of `Thomas P. Minka:
    Automatic Choice of Dimensionality for PCA. NIPS 2000: 598-604`
    Implements the probabilistic PCA model from:
    M. Tipping and C. Bishop, Probabilistic Principal Component Analysis,
    Journal of the Royal Statistical Society, Series B, 61, Part 3, pp. 611-622
    via the score and score_samples methods.
    See http://www.miketipping.com/papers/met-mppca.pdf
    Due to implementation subtleties of the Singular Value Decomposition (SVD),
    which is used in this implementation, running fit twice on the same matrix
    can lead to principal components with signs flipped (change in direction).
    For this reason, it is important to always use the same estimator object to
    transform data in a consistent fashion.
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.decomposition import PCA
    >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    >>> pca = PCA(n_components=2)
    >>> pca.fit(X)
    PCA(copy=True, n_components=2, whiten=False)
    >>> print(pca.explained_variance_ratio_) # doctest: +ELLIPSIS
    [ 0.99244...  0.00755...]
    See also
    --------
    RandomizedPCA
    KernelPCA
    SparsePCA
    TruncatedSVD
    """
    def __init__(self, n_components=None, copy=True, whiten=False):
        self.n_components = n_components
        self.copy = copy
        self.whiten = whiten

    def fit(self, X, y=None):
        """Fit the model with X.
        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Training data, where n_samples in the number of samples
            and n_features is the number of features.
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self._fit(X)
        return self

    def fit_transform(self, X, y=None):
        """Fit the model with X and apply the dimensionality reduction on X.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """
        U, S, V = self._fit(X)
        U = U[:, :int(self.n_components_)]

        if self.whiten:
            # X_new = X * V / S * sqrt(n_samples) = U * sqrt(n_samples)
            U *= sqrt(X.shape[0])
        else:
            # X_new = X * V = U * S * V^T * V = U * S
            U *= S[:int(self.n_components_)]

        return U

    def _fit(self, X):
        """Fit the model on X
        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.
        Returns
        -------
        U, s, V : ndarrays
            The SVD of the input data, copied and centered when
            requested.
        """
        X = check_array(X)
        n_samples, n_features = X.shape
        X = as_float_array(X, copy=self.copy)
        # Center data
        self.mean_ = np.mean(X, axis=0)
        X -= self.mean_
        U, S, V = linalg.svd(X, full_matrices=False)
        explained_variance_ = (S ** 2) / n_samples
        explained_variance_ratio_ = (explained_variance_ /
                                     explained_variance_.sum())

        components_ = V

        n_components = self.n_components
        if n_components is None:
            n_components = n_features
        elif n_components == 'mle':
            if n_samples < n_features:
                raise ValueError("n_components='mle' is only supported "
                                 "if n_samples >= n_features")

            n_components = _infer_dimension_(explained_variance_,
                                             n_samples, n_features)
        elif not 0 <= n_components <= n_features:
            raise ValueError("n_components=%r invalid for n_features=%d"
                             % (n_components, n_features))

        if 0 < n_components < 1.0:
            # number of components for which the cumulated explained variance
            # percentage is superior to the desired threshold
            ratio_cumsum = explained_variance_ratio_.cumsum()
            n_components = np.sum(ratio_cumsum < n_components) + 1

        # Fix DepreciationWarning (i.e. do not index using non-integer numbers)
        n_components = int(n_components)

        # Compute noise covariance using Probabilistic PCA model
        # The sigma2 maximum likelihood (cf. eq. 12.46)
        if n_components < min(n_features, n_samples):
            self.noise_variance_ = explained_variance_[int(n_components):].mean()
        else:
            self.noise_variance_ = 0.

        # store n_samples to revert whitening when getting covariance
        self.n_samples_ = n_samples

        self.components_ = components_[:int(n_components)]
        self.explained_variance_ = explained_variance_[:int(n_components)]
        explained_variance_ratio_ = explained_variance_ratio_[:int(n_components)]
        self.explained_variance_ratio_ = explained_variance_ratio_
        self.n_components_ = n_components

        return U, S, V

    def get_covariance(self):
        """Compute data covariance with the generative model.
        ``cov = components_.T * S**2 * components_ + sigma2 * eye(n_features)``
        where  S**2 contains the explained variances.
        Returns
        -------
        cov : array, shape=(n_features, n_features)
            Estimated covariance of data.
        """
        components_ = self.components_
        exp_var = self.explained_variance_
        if self.whiten:
            components_ = components_ * np.sqrt(exp_var[:, np.newaxis])
        exp_var_diff = np.maximum(exp_var - self.noise_variance_, 0.)
        cov = np.dot(components_.T * exp_var_diff, components_)
        cov.flat[::len(cov) + 1] += self.noise_variance_  # modify diag inplace
        return cov

    def get_precision(self):
        """Compute data precision matrix with the generative model.
        Equals the inverse of the covariance but computed with
        the matrix inversion lemma for efficiency.
        Returns
        -------
        precision : array, shape=(n_features, n_features)
            Estimated precision of data.
        """
        n_features = self.components_.shape[1]

        # handle corner cases first
        if self.n_components_ == 0:
            return np.eye(n_features) / self.noise_variance_
        if self.n_components_ == n_features:
            return linalg.inv(self.get_covariance())

        # Get precision using matrix inversion lemma
        components_ = self.components_
        exp_var = self.explained_variance_
        exp_var_diff = np.maximum(exp_var - self.noise_variance_, 0.)
        precision = np.dot(components_, components_.T) / self.noise_variance_
        precision.flat[::len(precision) + 1] += 1. / exp_var_diff
        precision = np.dot(components_.T,
                           np.dot(linalg.inv(precision), components_))
        precision /= -(self.noise_variance_ ** 2)
        precision.flat[::len(precision) + 1] += 1. / self.noise_variance_
        return precision

    def transform(self, X):
        """Apply the dimensionality reduction on X.
        X is projected on the first principal components previous extracted
        from a training set.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data, where n_samples is the number of samples
            and n_features is the number of features.
        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """
        check_is_fitted(self, 'mean_')

        X = check_array(X)
        if self.mean_ is not None:
            X = X - self.mean_
        X_transformed = np.dot(X, self.components_.T)
        if self.whiten:
            X_transformed /= np.sqrt(self.explained_variance_)
        return X_transformed

    def inverse_transform(self, X):
        """Transform data back to its original space, i.e.,
        return an input X_original whose transform would be X
        Parameters
        ----------
        X : array-like, shape (n_samples, n_components)
            New data, where n_samples is the number of samples
            and n_components is the number of components.
        Returns
        -------
        X_original array-like, shape (n_samples, n_features)
        """
        check_is_fitted(self, 'mean_')

        if self.whiten:
            return fast_dot(
                X,
                np.sqrt(self.explained_variance_[:, np.newaxis]) *
                self.components_) + self.mean_
        else:
            return fast_dot(X, self.components_) + self.mean_

    def score_samples(self, X):
        """Return the log-likelihood of each sample
        See. "Pattern Recognition and Machine Learning"
        by C. Bishop, 12.2.1 p. 574
        or http://www.miketipping.com/papers/met-mppca.pdf
        Parameters
        ----------
        X: array, shape(n_samples, n_features)
            The data.
        Returns
        -------
        ll: array, shape (n_samples,)
            Log-likelihood of each sample under the current model
        """
        check_is_fitted(self, 'mean_')

        X = check_array(X)
        Xr = X - self.mean_
        n_features = X.shape[1]
        log_like = np.zeros(X.shape[0])
        precision = self.get_precision()
        log_like = -.5 * (Xr * (np.dot(Xr, precision))).sum(axis=1)
        log_like -= .5 * (n_features * log(2. * np.pi)
                          - fast_logdet(precision))
        return log_like

    def score(self, X, y=None):
        """Return the average log-likelihood of all samples
        See. "Pattern Recognition and Machine Learning"
        by C. Bishop, 12.2.1 p. 574
        or http://www.miketipping.com/papers/met-mppca.pdf
        Parameters
        ----------
        X: array, shape(n_samples, n_features)
            The data.
        Returns
        -------
        ll: float
            Average log-likelihood of the samples under the current model
        """
        return np.mean(self.score_samples(X))


def maxstuff(X):
    index = 0
    maxi = X[0]
    for i in range(1, len(X)):
        if X[i] > maxi:
            maxi = X[i]
            index = i
    return maxi, index


def interpolation(Z, X, F):
    """
    """
    try:
        indices = X.searchsorted(Z)
    except BaseException:
        print(X)
        print(Z)
    return -(F[indices] * (X[indices - 1] - Z) - F[indices - 1] *
             (X[indices] - Z)) / (X[indices] - X[indices - 1])


def gcm(X, left, right):
    F = numpy.arange(0, 1, 1. / len(X))
    GCM = numpy.array(left)
    while left < right:
        slopes_left = (F[(left + 1):(right + 1)] - F[left]) / \
            (X[(left + 1):(right + 1)] - X[left])
        left += 1 + slopes_left.argmin()
        GCM = numpy.append(GCM, left)
    return GCM


def lcm(X, left, right):
    F = numpy.arange(0, 1, 1. / len(X))
    LCM = numpy.array(right)
    while left < right:
        slopes_right = (F[left:right] - F[right]) / (X[left:right] - X[right])
        right = left + slopes_right.argmin()
        LCM = numpy.append(right, LCM)
    return LCM


def dip_threshold(n, p_value):
    k = 21.642
    theta = 1.84157e-2/numpy.sqrt(n)
    return gamma.ppf(1.-p_value, a=k, scale=theta)


def dip(X):
    try:
        X = numpy.sort(X)
        F = numpy.arange(0, 1, 1. / X.shape[0]) + 1. / X.shape[0]
        left = 0
        right = len(X) - 1
        D = 0
        d = 1
        while True:
            GCM = gcm(X, left, right)
            LCM = lcm(X, left, right)

            Lg = interpolation(X[GCM], X[LCM], F[LCM])
            Gl = interpolation(X[LCM], X[GCM], F[GCM])

            gap_g, gap_g_index = maxstuff(numpy.abs(F[GCM] - Lg))
            gap_l, gap_l_index = maxstuff(numpy.abs(F[LCM] - Gl))

            if gap_g > gap_l:
                d = gap_g
                left_ = GCM[gap_g_index]
                right_ = LCM[LCM.searchsorted(GCM[gap_g_index])]
            else:
                d = gap_l
                left_ = GCM[GCM.searchsorted(LCM[gap_l_index]) - 1]
                right_ = LCM[gap_l_index]
            if d <= D:
                return D / 2.
            else:
                sup_l = numpy.abs(interpolation(
                    X[left:(left_ + 1)], X[GCM], F[GCM]) - F[left:(left_ + 1)]).max()
                sup_r = numpy.abs(interpolation(
                    X[right_:(right + 1)], X[LCM], F[LCM]) - F[right_:(right + 1)]).max()
                D = max([D, sup_l, sup_r])
                left = left_
                right = right_
    except Exception:
        return numpy.inf


PVAL_A = 0.4785
PVAL_B = 0.1946
PVAL_C = 2.0287


def decision_bound(p_value, n, d):
    """
    Compute the decision bound q according to the desired p-value. The test would
    be significant if |phi-1|>q.

    Parameters
    ----------

    p_value: float
        between 0 and 1 (this is the probability to be in the uniform case)
    n: int
        the number of observations
    d: int
        the dimension
    """
    return PVAL_A * (p_value - PVAL_B * numpy.log(1 - p_value)) * \
        (PVAL_C + numpy.log(d)) / numpy.sqrt(n)


def p_value(phi, n, d):
    """
    Compute the p-value of a test

    Parameters
    ----------

    phi: float
        the folding statistics
    n: int
        the number of observations
    d: int
        the dimension
    """
    try:
        def obj_fun(p):
            return numpy.abs(phi - 1.) - decision_bound(1 - p, n, d)
        p_val = brenth(obj_fun, 0., 1.)
    except BaseException:
        p_val = numpy.exp(-numpy.abs(phi - 1.) * numpy.sqrt(n) / (PVAL_C + numpy.log(d)))
    return p_val


def diagonal(X):
    """
    Returns the diagonal of the smallest hypercube including the dataset X

    Parameters
    ----------

    X: numpy.ndarray
        a d by n matrix (n observations in dimension d)
    """
    return numpy.linalg.norm(X.max(1)-X.min(1))


def markov_coeff(X, X_reduced):
    """
    Computes the Markov coefficient

    Parameters
    ----------

    X: numpy.ndarray
        a d by n matrix (n observations in dimension d)
    X_reduced: numpy.ndarray
        the 1 by n matrix equals to ||X-s*(X)||
    """
    return (X_reduced/diagonal(X)).mean()


def markov_bound(d):
    """
    Returns the bound on the Markov coefficient

    Parameters
    ----------

    d: int
        dimension of the feature space
    """
    return numpy.sqrt(d) / (2. * (d + 1.))


def batch_folding_test_with_MPA(X, with_markov=False):
    """
    Perform statically the folding test of unimodality (pure python) with a
    Markov Ex Post Analysis

    Parameters
    ----------

    X: numpy.ndarray
        a d by n matrix (n observations in dimension d)
    """
    try:
        n, p = X.shape
    except BaseException:
        X = X.reshape(1, len(X))
        n, p = X.shape

    if n > p:  # if lines are observations, we transpose it
        X = X.T
        dim = p
        n_obs = n
    else:
        dim = n
        n_obs = p

    X_square_norm = (X * X).sum(axis=0)  # |X|²
    mat_cov = np.cov(X).reshape(dim, dim)  # cov(X)
    trace = np.trace(mat_cov)  # Tr(cov(X))

    try:
        cov_norm = np.cov(X, X_square_norm)[
            :-1, -1].reshape(-1, 1)  # cov(X,|X|²)
        pivot = 0.5 * np.linalg.solve(mat_cov, cov_norm)
    except numpy.linalg.LinAlgError:
        pivot = minimize(lambda s: np.power(
            X.T - s, 2).sum(axis=1).var(), x0=X.mean(axis=1)).x.reshape(-1, 1)

    X_reduced = np.linalg.norm(X-pivot, axis=0)  # |X-s*|
    phi = pow(1. + dim, 2) * X_reduced.var(ddof=1) / trace
    unimodal = (phi >= 1.)
    if not with_markov:
        return unimodal, p_value(phi, n_obs, dim), phi, None
    else:
        if unimodal:
            mc = markov_coeff(X, X_reduced)
            if mc>markov_bound(dim):
                unimodal = False
        else:
            mc = None
        return unimodal, p_value(phi, n_obs, dim), phi, mc


def nd_bhatta_dist(X1, X2):

    mu_1 = numpy.mean(X1, 1)
    mu_2 = numpy.mean(X2, 1)
    ms = mu_1 - mu_2

    cov_1 = numpy.cov(X1)
    cov_2 = numpy.cov(X2)
    cov = (cov_1 + cov_2)/2

    det_1 = numpy.linalg.det(cov_1)
    det_2 = numpy.linalg.det(cov_2)
    det = numpy.linalg.det(cov)

    dist = (1/8.)*numpy.dot(numpy.dot(ms.T, numpy.linalg.inv(cov)), ms) + 0.5*numpy.log(det/numpy.sqrt(det_1*det_2))
    return dist


import numpy as np
from math import sqrt
from scipy.stats import gaussian_kde


def bhatta_dist(X1, X2, method='continuous', n_steps=50, bounds=None):
    # Calculate the Bhattacharyya distance between X1 and X2. X1 and X2 should be 1D numpy arrays representing the same
    # feature in two separate classes. 

    def get_density(x, cov_factor=0.1):
        # Produces a continuous density function for the data in 'x'.
        # Some benefit may be gained from adjusting the cov_factor.
        density = gaussian_kde(x)
        density.covariance_factor = lambda:cov_factor
        density._compute_covariance()
        return density

    # Combine X1 and X2, we'll use it later:
    cX = np.concatenate((X1, X2))

    if method == 'noiseless':
        # # This method works well when the feature is qualitative (rather than quantitative). Each unique value is
        # # treated as an individual bin.
        uX = np.unique(cX)
        A1 = len(X1) * (max(cX)-min(cX)) / len(uX)
        A2 = len(X2) * (max(cX)-min(cX)) / len(uX)
        bht = 0
        for x in uX:
            p1 = (X1 == x).sum() / A1
            p2 = (X2 == x).sum() / A2
            bht += sqrt(p1*p2) * (max(cX)-min(cX))/len(uX)

    elif method == 'hist':
        # # Bin the values into a hardcoded number of bins (this is sensitive to N_BINS).
        N_BINS = 10
        # Bin the values:
        h1 = np.histogram(X1, bins=N_BINS, range=(min(cX), max(cX)), density=True)[0]
        h2 = np.histogram(X2, bins=N_BINS, range=(min(cX), max(cX)), density=True)[0]
        # Calc coeff from bin densities:
        bht = 0
        for i in range(N_BINS):
            p1 = h1[i]
            p2 = h2[i]
            bht += sqrt(p1*p2) * (max(cX)-min(cX))/N_BINS

    elif method == 'autohist':
        # # Bin the values into bins automatically set by np.histogram:
        # Create bins from the combined sets:
        # bins = np.histogram(cX, bins='fd')[1]
        bins = np.histogram(cX, bins='doane')[1]  # Seems to work better
        # bins = np.histogram(cX, bins='auto')[1]

        h1 = np.histogram(X1,bins=bins, density=True)[0]
        h2 = np.histogram(X2,bins=bins, density=True)[0]

        # Calc coeff from bin densities:
        bht = 0
        for i in range(len(h1)):
            p1 = h1[i]
            p2 = h2[i]
            bht += sqrt(p1*p2) * (max(cX)-min(cX))/len(h1)

    elif method == 'continuous':
        # # Use a continuous density function to calculate the coefficient (This is the most consistent, but also slightly slow):
        # Get density functions:
        d1 = get_density(X1)
        d2 = get_density(X2)
        # Calc coeff:
        if bounds is None:
            bounds = (min(cX), max(cX))

        xs = np.linspace(bounds[0], bounds[1], n_steps)
        bht = 0
        for x in xs:
            bht += sqrt(d1(x) * d2(x))

        bht *= (bounds[1]-bounds[0])/n_steps

    else:
        raise ValueError("The value of the 'method' parameter does not match any known method")

    # # Lastly, convert the coefficient into distance:
    if bht == 0:
        return float('Inf')
    else:
        return -np.log(bht)
