import numpy as np
from scipy.linalg import logm
import itertools
from einops import rearrange

def norm_l(vec):
    return vec / (np.sum(vec) + 1e-16)

def norm_r(vec):
    return vec/(vec[0,0] + 1e-16)

def purify(mat, tol = 1e-10):
    mat[np.abs(mat) < tol] = 0
    return mat

def pprint(array):
    print(purify(array))

class EMachine:
    """
    EMachine
    Representation of a matrix-product (tensor) instrument / emission machine
    with utilities for canonicalization, spectral analysis, statistical and
    quantum information measures, observable correlations, and MPO manipulation.
    This class wraps a collection of site tensors A (derived from an instrument
    tensor T) and provides methods to compute transfer operators, canonical
    gauges, stationary statistics, and to build/inspect parent Hamiltonians
    and MPO decompositions used in 1D quantum/classical models.

    Parameters
    ----------
    T : ndarray, shape (m, d, d)
        Instrument tensor (or list/array of Kraus-like/transfer operators) with
        m observable symbols and a bond (auxiliary) dimension d. Expected to be
        convertible to a NumPy array. The class constructs the local operator
        set A = sqrt(T) (elementwise or via a sensible square-root convention).
        The code expects T to be such that np.sqrt(T) (or an equivalent) yields
        well-shaped matrices; the user is responsible for providing a compatible
        representation.
    Attributes
    ----------
    T : ndarray, shape (m, d, d)
        Original instrument tensor provided at initialization.
    A : ndarray, shape (m, d, d)
        Local site matrices derived from T (square-root or otherwise).
    dim : int
        Bond (matrix) dimension of each A (alias for d).
    mdim : int
        Number of observable symbols (alias for m).
    E : ndarray, shape (d**2, d**2)
        Reshaped transfer operator (environment/transfer matrix) built from
        outer products of the A matrices and arranged for linear-algebraic
        manipulations on vectorized operators.
    B : ndarray, shape (d, d) or (num_states, num_observations)
        Classical/emission matrix (sum over symbol-indexed slices of T).
    eig_r, eig_l : tuple
        Raw eigen-decompositions (eigenvalues, eigenvectors) computed for E
        (right and left respectively). These may be complex-valued.
    eig_r_mat, eig_l_mat : ndarray, shape (d, d)
        Reshaped/normalized dominant right/left eigenmatrices used as boundary
        conditions for matrix-product evaluations.
    w_r, w_l : ndarray, shape (d, d)
        Cholesky factors (or similar) for eig_r_mat and eig_l_mat used to
        construct canonical gauge transformations.
    U, lam_v, V : ndarray
        SVD factors used in the canonicalization; lam_v are singular values.
    gam, lam : ndarray
        Canonical MPS/MPO tensor (gauge transformed) and diagonal singular
        value matrix (lam = diag(lam_v)).
    eig_l_classical, eig_r_classical : tuple
        Eigen-decompositions of the classical transition matrix (sum over T).
    l : None or other
        Placeholder attribute for external use; not set by class internals.
    Public methods
    --------------
    __repr__():
        Human-readable representation showing primary dimensions.
    can_r():
        Return the canonical right boundary (self.gam @ self.lam). Useful to
        build a canonical EMachine from its square (e.g., EMachine(can_r()**2)).
    can_l():
        Return the canonical left boundary (self.lam @ self.gam).
    unitary():
        Apply a similarity transform by w_r to each A and return the Hermitian
        conjugate of inv(w_r) @ A @ w_r for each symbol. Returns an array of
        shape (m, d, d).
    density():
        Construct the density matrix rho = sum_i eig_l_mat[i,i] * outer(w_r[i], w_r[i]).
        Returns a (d, d) ndarray. This is the effective stationary density in
        the canonical basis encoded by the eigen-objects.
    quantum_statistical_memory():
        Von Neumann entropy of the density matrix: -Tr(rho log2 rho). Returns
        a float (bits). Requires a numerical logm routine (scipy.linalg.logm).
    topological_memory():
        Log2 of the rank of the density matrix (bits). Returns np.log2(rank).
    statistical_memory():
        Shannon-like entropy of the left-eigenmatrix diagonal via a matrix-log
        expression (uses logm on eig_l_mat). Returns a float (bits).
    correlation_length():
        Estimate of correlation length from the leading and subleading
        eigenvalues of the transfer operator E:
        -1 / log2(|second_largest_eigenvalue|). Returns float or np.inf if no
        subleading eigenvalue exists.
    propagator(iterations, state):
        Classical stochastic sampler that evolves an initial classical state
        distribution for a number of steps according to T and emits sampled
        symbols. Parameters:
          state : 1D ndarray (length = dim), initial probability vector.
        Returns a 1D integer array of emitted symbols. This method uses
        numpy.random.choice — the output is stochastic and depends on RNG seed.
    state_distribution(n):
        Compute the classical transition matrix (sum over T) raised to the
        power n using eigendecomposition: V diag(lambda**n) V^{-1}. Returns
        the (mdim x mdim) matrix representing the n-step transition.
    emission_distribution(n):
        Compute the distribution over observable symbols at time n by applying
        the (n-1)-step state transition to the emission matrix B:
          state_dist(n-1) @ B
        Returns a 1D array of probabilities over symbols.
    mean(f):
        Expectation value of observable f (array indexed by observation symbols)
        under the stationary distribution implied by eig_l_mat diagonal.
    variance(f):
        Variance of observable f under the stationary distribution.
    covariance(f, n):
        Finite-sample covariance estimator for observable f at separation n.
        This method uses spectral decompositions of the classical transition
        matrix to form intermediate matrices and returns a scalar estimate.
        Note: the implementation uses several matrix inverses and eigen-data;
        ensure arguments are valid and matrices are invertible where required.
    asymptotic_variance(f, n):
        Asymptotic variance combining variance/n and covariance contributions:
          variance(f)/n + 2 * covariance(f, n).
    measure(output):
        Matrix-product evaluation of the weight/probability of a symbol string
        `output` (e.g., "00110") using the boundary matrices eig_l_mat and
        eig_r_mat and the A tensors. Returns a scalar (complex or real).
    measure_2(output):
        Alternative evaluation using the reshaped E operator (vectorized
        operator-space representation). Produces the same measure as measure()
        in exact arithmetic.
    measure_can(output):
        Rebuilds a canonical EMachine from the canonical right object
        (EMachine(self.can_r()**2)) and evaluates the measure for `output`.
        This performs additional construction and prints the underlying T of
        the canonical machine (debugging side-effect).
    observable(O, l):
        Multi-site observable correlation: builds a modified transfer operator
        incorporating the single-site observable O (an m x m operator on symbol
        space) and evaluates its l-th power between left/right eigenmatrices.
        Returns a scalar expectation.
    correlation(O, l):
        Two-point correlation function of observable O at separation l. Returns
        the connected correlation (covariance) using the transfer operator E.
    interaction_set(n):
        Generate dictionary mapping each length-n sequence (tuple of symbols)
        to the interaction matrix M = A[x_0] @ A[x_1] @ ... @ A[x_{n-1}]. Keys
        are tuples of ints in range(m). Useful for enumerating all sequence
        interactions on a block of length n.
    interaction_rank(n):
        Compute the linear rank of the collection of interaction matrices for
        length n by flattening each interaction matrix into a vector and
        computing np.linalg.matrix_rank on the stacked vectors.
    ground_space(l):
        Compute an orthonormal (reduced-QR) basis for the span of single-site
        operator overlaps with the set of l-site interaction matrices. Returns
        a 2-D array whose rows are orthonormal vectors in the mdim**l-dimensional
        local Hilbert space (shape roughly (r, mdim**l)). This basis is used to
        form parent Hamiltonian projectors.
    parent_hamiltonian(l, reshape=True):
        Return the local parent Hamiltonian term (projector onto the orthogonal
        complement of the l-site ground-space). If reshape is False return a
        matrix of shape (mdim**l, mdim**l). If reshape is True return a tensor
        with axes ordered as (a0, b0, a1, b1, ..., a_{l-1}, b_{l-1}) suitable
        for per-site bra/ket indexing. Requires ground_space(l) to produce
        orthonormal row vectors.
    decompose_mpo(mpo, l):
        Decompose a full l-site MPO (square matrix shape (mdim**l, mdim**l)) into
        a list of local MPO site tensors via iterative SVDs. The returned list
        contains l site tensors with conventional MPO index order. No rank
        truncation is performed; memory/bond dimensions may grow quickly.
    mpo_action(mpo_sites):
        Contract a list of MPO site tensors with the instance's A-tensors and
        return the assembled multi-index tensor representing the MPO acting on
        the physical bond legs. The returned shape is (dim, b0, b1, ..., b_{L-1}, dim)
        where bk are MPO bond dimensions (typically mdim). The routine expects
        each site tensor to be indexable by symbols 0..mdim-1 as its first axis.
    - This class mixes classical (stochastic) and quantum (matrix-product)
      constructions. The user is responsible for providing consistent inputs
      (compatible shapes, probability normalization when appropriate).
    - Many routines rely on eigendecompositions and matrix inverses; numerical
      instabilities can occur for nearly-degenerate spectra or ill-conditioned
      matrices. Small regularization (eps diagonals, SVD truncation) may be
      required in practice.
    - The implementation assumes availability of:
        - NumPy (array and linear algebra primitives)
        - scipy.linalg.logm or equivalent for matrix logarithms used in entropies
        - einops.rearrange for reshaping MPOs and tensors
      If these are not available the dependent methods will fail.
    - Several methods (propagator, measure_can) have side-effects or use RNG:
      propagator returns stochastic samples and measure_can prints debugging
      information.
    Exceptions
    ----------
    Typical exceptions raised by methods include:
    - ValueError: when shapes or input arguments are incompatible.
    - numpy.linalg.LinAlgError: when matrix inversion/Cholesky/SVD fails due to
      singular or ill-conditioned matrices.
    - IndexError: when symbol indices are out of bounds for the provided A/T.
    # Construct and inspect
    em = EMachine(T)                # T: ndarray with shape (m, d, d)
    print(em)                       # EMachine(dim=d, A_shape=(m, d, d))
    # Stationary and information quantities
    rho = em.density()
    S_q = em.quantum_statistical_memory()
    S_top = em.topological_memory()
    # Evaluate the probability (weight) of a short output string
    p = em.measure("0101")
    # Build a 2-site parent Hamiltonian tensor
    h2 = em.parent_hamiltonian(2, reshape=True)
    # Decompose and reassemble an MPO (requires einops)
    mpo_sites = em.decompose_mpo(some_mpo_matrix, l=3)
    assembled = em.mpo_action(mpo_sites)
    See Also
    - NumPy: ndarray, linalg.eig, linalg.svd, linalg.cholesky
    - SciPy: scipy.linalg.logm for matrix logarithm when computing entropies
    - einops.rearrange for tensor reshaping in MPO routines
    """
    def __init__(self, A):
        self.A = A.astype(np.complex128)
        self.T = A**2
        self.dim = self.A.shape[1]
        self.mdim = self.A.shape[0]
        self.E = np.tensordot(self.A, self.A.conj(), axes = ([0],[0])).transpose(0,2,1,3).reshape(self.dim**2, self.dim**2)
        self.B = np.sum(self.T.transpose(2, 1, 0) , axis = 0)

        self.eig_r = np.linalg.eig(self.E)
        leading_idx_r = np.argmax(np.abs(self.eig_r[0]))
        self.eig_r_mat = norm_r(rearrange(self.eig_r[1][:, leading_idx_r], '(a b) -> a b', a=self.dim, b=self.dim))
        self.eig_l = np.linalg.eig(self.E.T)
        leading_idx_l = np.argmax(np.abs(self.eig_l[0]))
        self.eig_l_mat = norm_l(rearrange(self.eig_l[1][:, leading_idx_l], '(a b) -> a b', a=self.dim, b=self.dim))
        self.w_r = np.linalg.cholesky(self.eig_r_mat + np.eye(self.dim)*1e-12)
        self.w_l = np.linalg.cholesky(self.eig_l_mat + np.eye(self.dim)*1e-12)

        self.U, self.lam_v, self.V = np.linalg.svd(self.w_l @ self.w_r)

        self.gam = self.V @ np.linalg.inv(self.w_r) @ self.A @ np.linalg.inv(self.w_l) @ self.U
        self.lam = np.diag(self.lam_v)

        self.Q = np.sum(self.T, axis=0)
        self.eig_l_classical = np.linalg.eig(self.Q.T)
        self.eig_r_classical = np.linalg.eig(self.Q)

        p_i = self.eig_l_mat.diagonal()
        p_x = 1/(self.eig_l_mat.diagonal() @ self.B)
        p_x_mesh, p_i_mesh = np.meshgrid(p_x, p_i)
        self.D = p_x_mesh * p_i_mesh * self.B
    def __repr__(self):
        return f"EMachine(dim={self.dim}, A_shape={self.A.shape})"

    def can_r(self):
        """
        Compute and return an EMachine built from the right canonical A matrix.
        This method computes the right canonical A matrix as the matrix product of the
        instance attributes `gam` and `lam` (i.e. A_can = self.gam @ self.lam) and
        returns a new EMachine initialized with that matrix.

        Parameters
        ----------
        self : EMachine
            The instance whose `gam` and `lam` attributes are used. Both must be
            array-like objects with shapes compatible for matrix multiplication
            (i.e., self.gam.shape[1] == self.lam.shape[0]).
        Returns
        -------
        EMachine
            A new EMachine constructed from the canonical A matrix A_can.
        Raises
        ------
        AttributeError
            If `self` does not have `gam` or `lam` attributes.
        ValueError
            If the shapes of `gam` and `lam` are not aligned for matrix multiplication.
        Notes
        -----
        This method does not modify the calling instance; it returns a new EMachine.
        """

        A_can =  self.gam @ self.lam
        return EMachine(A_can)
    
    def can_l(self):
        """
        Compute and return an EMachine built from the left canonical A matrix.
        This method computes the left canonical A matrix as the matrix product of the
        instance attributes `lam` and `gam` (i.e. A_can = self.lam @ self.gam) and
        returns a new EMachine initialized with that matrix.

        Parameters
        ----------
        self : EMachine
            The instance whose `gam` and `lam` attributes are used. Both must be
            array-like objects with shapes compatible for matrix multiplication
            (i.e., self.gam.shape[1] == self.lam.shape[0]).
        Returns
        -------
        EMachine
            A new EMachine constructed from the canonical A matrix A_can.
        Raises
        ------
        AttributeError
            If `self` does not have `gam` or `lam` attributes.
        ValueError
            If the shapes of `gam` and `lam` are not aligned for matrix multiplication.
        Notes
        -----
        This method does not modify the calling instance; it returns a new EMachine.
        """

        A_can = self.lam @ self.gam
        return EMachine(A_can)
    
    def unitary(self):
        """
        Return the Hermitian (conjugate-transpose) similarity transforms of matrices in self.A
        using the operator self.w_r.
        For each matrix a in self.A, this method computes
            (inv(self.w_r) @ a @ self.w_r).conj().T
        and collects the results into a single numpy.ndarray.
        
        Parameters
        ----------
        self : object
            Expected to provide the attributes:
            - w_r : (n, n) array_like
                A square, invertible matrix used for the similarity transform.
            - A : iterable of (n, n) array_like
                An iterable (e.g. list or array) of square matrices to be transformed.
        Returns
        -------
        ndarray
            A numpy array of shape (m, n, n) where m = len(self.A). Each entry is the
            Hermitian (conjugate-transpose) of inv(w_r) @ a @ w_r. The returned dtype
            will typically be complex if any inputs are complex.
        Raises
        ------
        numpy.linalg.LinAlgError
            If self.w_r is singular and cannot be inverted.
        ValueError
            If the matrices in self.A are not compatible in shape with self.w_r.
        Notes
        -----
        This operation performs a similarity transform followed by a conjugate transpose
        for each matrix in self.A. No in-place modification of self.w_r or the elements
        of self.A is performed.
        Examples
        --------
        Assuming self.w_r has shape (n, n) and self.A is a list of m matrices of shape (n, n),
        the result will have shape (m, n, n):
            result = self.unitary()
        """
        
        return np.array([np.conj(np.linalg.inv(self.w_r) @ a @ self.w_r).T for a in self.A])

    def density(self):
        """
        Compute the density-like quantity w_r† @ eig_l_mat @ w_r.
        This method returns the conjugate-transpose product of the instance's
        right-eigenvector(s) with the stored left-eigenvector matrix (or density-like
        operator) according to:
            result = w_r.T.conj() @ eig_l_mat @ w_r

        Parameters
        ----------
        self : object
            Instance expected to provide the attributes:
            - w_r: array_like, shape (N,) or (N, K)
              Right eigenvector(s). If 1-D, treated as a single vector; if 2-D,
              columns are treated as separate vectors.
            - eig_l_mat: array_like, shape (N, N)
              Left-eigenvector matrix or an operator to be sandwiched between w_r† and w_r.
        Returns
        -------
        complex or numpy.ndarray
            If w_r is 1-D, returns a scalar complex value (the overlap/expectation).
            If w_r is 2-D with K columns, returns a (K, K) array giving the matrix of
            overlaps w_r† @ eig_l_mat @ w_r.
        Raises
        ------
        AttributeError
            If required attributes (w_r or eig_l_mat) are missing on the instance.
        Notes
        -----
        - The operation uses the Hermitian transpose of w_r (conjugate transpose).
        - The returned object represents an overlap, expectation value, or reduced
          density depending on the interpretation of eig_l_mat and w_r.
        """

        return self.w_r.T.conj() @ self.eig_l_mat @ self.w_r
    
    def quantum_statistical_memory(self):
        """
        Compute the quantum statistical memory, defined as the von Neumann entropy of the density matrix.

        Returns
        -------
        float
            The quantum statistical memory in bits.
        Notes
        -----
        - Uses the density matrix computed by self.density().
        - The von Neumann entropy is calculated as -Tr(rho log2 rho).
        """
        return -np.trace(self.density() @ logm(self.density()) / np.log(2))
    
    def topological_memory(self):
        """
        Compute the topological memory, defined as the log2 of the rank of the density matrix.

        Returns
        -------
        float
            The topological memory in bits.
        Notes
        -----
        - Uses the density matrix computed by self.density().
        - The rank is computed using numpy.linalg.matrix_rank.
        """
        return np.log2(np.linalg.matrix_rank(self.density()))
    
    def statistical_memory(self):
        """
        Compute the statistical memory, defined as the Shannon entropy of the diagonal of the left eigenmatrix.

        Returns
        -------
        float
            The statistical memory in bits.
        Notes
        -----
        - Uses the diagonal of self.eig_l_mat.
        - The Shannon entropy is calculated as -Tr(p log2 p) for the diagonal probabilities p.
        """
        return np.trace(-self.eig_l_mat * logm(self.eig_l_mat) / np.log(2))
    
    def correlation_length(self):
        """
        Calculate the correlation length from the eigenvalues of the E-matrix.

        Returns
        -------
        float
            The estimated correlation length.
        Notes
        -----
        - Uses the eigenvalues of the E-matrix stored in self.E.
        - The correlation length is computed as -1 / log2(|second largest eigenvalue|).
        """
        eigvals = np.linalg.eigvals(self.E)
        leading_eigval = np.max(np.abs(eigvals))
        subleading_eigvals = eigvals[np.abs(eigvals) < leading_eigval]
        if len(subleading_eigvals) == 0:
            return np.inf
        second_largest = np.max(np.abs(subleading_eigvals))
        return -1 / np.log2(np.abs(second_largest))
    
    def propagator(self, iterations, state):
        """
        Stochastic sampler that evolves a classical state vector using the
        instrument T for a number of iterations and returns the emitted symbol sequence.

        Parameters
        ----------
        iterations : int
            Number of time steps to evolve the state.
        state : numpy.ndarray, shape (d,)
            Initial classical state vector of dimension d.
        Returns
        -------
        numpy.ndarray
            1-D array of emitted symbols (0 or 1) of length equal to `iterations`.
        Notes
        -----
        - The method assumes self.T is an array of shape (m, d, d) where m is the number of symbols.
        - The input state should be a valid probability distribution (non-negative, sums to 1).
        - The output symbols are sampled according to the probabilities derived from the state evolution.
        """
        stoch = np.array([], dtype=int)
        for _ in range(iterations):
            state_arr = np.array([np.dot(t.T, state) for t in self.T])
            prob = np.sum(state_arr, axis=1)
            output = np.random.choice([0, 1], p=prob/prob.sum())
            stoch = np.append(stoch, output)
            state_num = np.random.choice([0, 1], p=state_arr[output]/(state_arr[output].sum()))
            state = np.zeros(self.dim)
            state[state_num] = 1
        return stoch

    def state_distribution(self, n):
        """
        Compute the classical transition matrix raised to power n via eigendecomposition.

        Parameters
        ----------
        n : int
            Power to which the classical transition matrix is raised.
        Returns
        -------
        numpy.ndarray
            2-D array representing the classical transition matrix raised to power n.
        Notes
        -----
        - The classical transition matrix is obtained by summing self.T over the first axis.
        - The method uses eigendecomposition for efficient computation of the matrix power.
        """
        trans_classical = np.sum(self.T, axis=0)
        eigval, eigvec = np.linalg.eig(trans_classical)
        return eigvec @ np.diag(eigval**n) @ np.linalg.inv(eigvec)

    def emission_distribution(self, n):
        """
        Compute the emission (observation) distribution at time step n.

        Parameters
        ----------
        n : int
            Time index for which to compute the emission distribution. This method
            uses the state distribution at time n-1 to produce the distribution over
            observable symbols at time n.
        Returns
        -------
        numpy.ndarray
            1-D array of shape (num_observations,) representing the probability
            distribution over observation symbols at time n. Computation follows
                emission_dist = state_distribution(n-1) @ self.B
            where self.B is the emission matrix with shape (num_states, num_observations)
            and state_distribution(n-1) returns a 1-D array of length num_states.
        Raises
        ------
        ValueError
            If n < 1 or if the shapes of the state distribution and emission matrix
            are incompatible.
        Notes
        -----
        - The method assumes emission probabilities in self.B are conditioned on hidden
          states (rows correspond to states, columns to observation symbols).
        - Returned array sums to 1 up to numerical precision if inputs are valid
          probability distributions.
        """
        state_dist = self.state_distribution(n-1)
        return state_dist @ self.B

    def mean(self, f):
        """
        Compute the expectation value of observable f of the emitted distribution
        with respect to the stationary distribution.

        Parameters
        ----------
        f : numpy.ndarray, shape (num_observations,)
            Observable function defined over the observation symbols.
        Returns
        -------
        float
            The expectation value of f of the emitted distributionunder the stationary distribution.
        Notes
        -----
        - The stationary distribution is given by the diagonal of self.eig_l_mat.
        """
        return np.sum(self.eig_l_mat.diagonal() @ self.B * f)

    def variance(self, f):
        """
        Compute the variance of observable f of the emitted distribution
        with respect to the stationary distribution.

        Parameters
        ----------
        f : numpy.ndarray, shape (num_observations,)
            Observable function defined over the observation symbols.
        Returns
        -------
        float
            The variance of f of the emitted distribution under the stationary distribution.
        Notes
        -----
        - The stationary distribution is given by the diagonal of self.eig_l_mat.
        """
        mean_f = self.mean(f)
        return np.dot(self.eig_l_mat.diagonal() @ self.B, (f - mean_f)**2)

    def D_asym(self, init_state, n):
        """
        Compute the matrix D used in the relation probability calculation.

        Parameters
        ----------
        init_state : numpy.ndarray
            Initial state distribution.
        n : int
            Time step for the calculation.

        Returns
        -------
        numpy.ndarray
            The computed matrix D.
        """
        p_i = init_state @ np.linalg.matrix_power(self.Q, n)
        p_x = 1/(self.eig_l_mat.diagonal() @ self.B)
        p_x_mesh, p_i_mesh = np.meshgrid(p_x, p_i)
        return (p_x_mesh * p_i_mesh * self.B).T

    def covariance(self, f, n):
        """
        Compute the finite-sample covariance of observable f at separation n
        with respect to the stationary distribution.

        Parameters
        ----------
        f : numpy.ndarray, shape (num_observations,)
            Observable function defined over the observation symbols.
        n : int
            Separation distance for covariance computation.
        Returns
        -------
        float
            The covariance of f at separation n under the stationary distribution.
        Notes
        -----
        - The stationary distribution is given by the diagonal of self.eig_l_mat.
        """
        mean_f = self.mean(f)
        stat_prob = self.eig_l_mat.diagonal()
        f_centered = f - mean_f
        func_1 = lambda x: (x - x**n)/(1-x)
        func_2 = lambda x: (x/(1-x)*(1-n*x**(n-1)) + x**2*(1-x**(n - 1))/(1 - x**2))
        mat_1 = np.zeros((self.dim, self.dim))
        mat_2 = np.zeros((self.dim, self.dim))
        for n, val in enumerate(self.eig_r_classical[0]):
            if int(val) != 1:
                mat_1 += func_1(val) * np.outer(self.eig_r_classical[1][:, n], self.eig_l_classical[1][:, n])
                mat_2 += func_2(val) * np.outer(self.eig_r_classical[1][:, n], self.eig_l_classical[1][:, n])
        return 2/n * np.dot(stat_prob @ self.B * f_centered, f_centered @ np.linalg.inv(self.B) @ mat_1 @ self.B) - 2/n**2 * np.dot(stat_prob @ self.B * f_centered, f_centered @ np.linalg.inv(self.B) @ mat_2 @ self.B)

    def asymptotic_variance(self, f):
        """
        Compute the asymptotic variance of observable f combining variance and covariance contributions.

        Parameters
        ----------
        f : numpy.ndarray, shape (num_observations,)
            Observable function defined over the observation symbols.
        n : int
            Sample size for asymptotic variance computation.
        Returns
        -------
        float
            The asymptotic variance of f under the stationary distribution.
        Notes
        -----
        - The stationary distribution is given by the diagonal of self.eig_l_mat.
        """
        mean_f = self.mean(f)
        stat_prob = self.eig_l_mat.diagonal()
        var = np.dot(stat_prob @ self.B, (f - mean_f)**2)
        f_centered = f - mean_f
        f_mat = np.outer(f, f)
        func = lambda x: 1/(1-x)
        mat = np.zeros((self.dim, self.dim))
        for n, val in enumerate(self.eig_r_classical[0]):
            if int(val+1e-10) != 1:
                norm = np.dot(self.eig_r_classical[1][:, n], self.eig_l_classical[1][:, n])
                mat += func(val) * np.outer(self.eig_r_classical[1][:, n], self.eig_l_classical[1][:, n]) / norm
        c_mat = stat_prob @ self.T @ mat @ self.B        
        cov = np.trace(f_mat @ c_mat)
        return var + 2 * cov

    def relation_probability(self, init_state, t, s):
        """
        Compute the relation probability matrix between system states at two time steps.
        This method evaluates two state-probability vectors at times s and t (with t >= s),
        then returns their outer product as a relation-probability matrix.

        Parameters
        ----------
        init_state : array_like
            Initial state probability vector (1D). Expected to be compatible with matrix
            multiplication by self.Q and self.B (e.g. shape (n,) or (1, n)).
        t : int
            Final time index (must be an integer >= 0).
        s : int
            Intermediate time index (0 <= s <= t).
        Returns
        -------
        numpy.ndarray
            2D array of shape (n, n) equal to outer(p_t, p_s), where
            p_s = init_state @ (self.Q ** s) @ self.B
            and p_t is obtained by further propagating (and modifying) the intermediate
            state:
            p_t = (init_state @ (self.Q ** s) @ self.B @ self.D(init_state, s))
                  @ (self.Q ** (t - s)) @ self.B
        Notes
        -----
        - The method uses numpy.linalg.matrix_power for powers of self.Q.
        - self.D must be a callable that returns an operator/matrix compatible with the
          intermediate multiplication (shapes must align).
        - A ValueError or NumPy broadcast/multiplication error may occur if t < s or if
          input/attribute shapes are incompatible.
        """
        p_t_1 = init_state @ np.linalg.matrix_power(self.Q, s) @ self.B @ self.D(init_state, s)
        p_t_2 = p_t_1 @ np.linalg.matrix_power(self.Q, t - s) @ self.B
        p_s = init_state @ np.linalg.matrix_power(self.Q, s) @ self.B
        return np.outer(p_t_2, p_s)
    
    def time_centered_dist(self, f, init_state, t):
        return f - init_state @ np.linalg.matrix_power(self.Q, t) @ self.B @ f
    
    def time_variance(self, f, init_state, n):
            var = 0
            cov = 0
            tvar = np.zeros(n)
            for t in range(n):
                var += np.dot(init_state @ np.linalg.matrix_power(self.Q, t) @ self.B, self.time_centered_dist(f, init_state, t)**2)
                for s in range(t):
                    cov += self.time_centered_dist(f, init_state, t) @ self.relation_probability(init_state, t, s) @ self.time_centered_dist(f, init_state, s)
                tvar[t] = (var + 2 * cov)/(t+1)**2
            return tvar
    
    def measure(self, output: str):
        """
        Compute the probability (or weight) of observing a given symbol string `output`
        using the matrix product representation.

        Parameters
        ----------
        output : str
            A string of symbols (e.g., '00110') representing the observed output sequence.
        Returns
        -------
        float
            The computed measure (probability or weight) of the observed output sequence.
        Notes
        -----
        - The method constructs a matrix product from the sequence of symbols in `output`
        - It uses the left and right eigenmatrices to evaluate the trace of the resulting product.
        """
        out_ls = list(output)
        mat_prod = np.diag(np.ones(self.dim))
        for out in reversed(out_ls):
            mat_prod = mat_prod @ np.conj(self.A[int(out)].T)
        mat_prod = mat_prod @ self.eig_l_mat
        for out in out_ls:
            mat_prod = mat_prod @ self.A[int(out)]
        mat_prod = mat_prod @ self.eig_r_mat
        return np.trace(mat_prod)
    
    def measure_2(self, output: str):
        """
        Alternative evaluation of measure(output) using the reshaped E operator.

        Parameters
        ----------
        output : str
            A string of symbols (e.g., '00110') representing the observed output sequence.
        Returns
        -------
        float
            The computed measure (probability or weight) of the observed output sequence.
        Notes
        -----
        - The method constructs a matrix product from the sequence of symbols in `output`
        - It uses the left and right eigenmatrices to evaluate the trace of the resulting product.
        """
        out_ls = list(output)
        mat_prod = np.diag(np.ones(self.dim**2))
        for out in out_ls:
            mat_prod = mat_prod @ np.tensordot(self.A[int(out)], np.conj(self.A[int(out)].T), axes = 0).transpose(0,2,1,3).reshape(self.dim**2, self.dim**2)
        return np.dot(self.eig_l_mat.reshape(self.dim**2), mat_prod @ self.eig_r_mat.reshape(self.dim**2))
    
    def measure_can(self, output: str):
        """
        Compute measure for the canonical gauge (rebuilds an EMachine in canonical form).

        Parameters
        ----------
        output : str
            A string of symbols (e.g., '00110') representing the observed output sequence.
        Returns
        -------
        float
            The computed measure (probability or weight) of the observed output sequence.
        Notes
        -----
        - The method constructs a matrix product from the sequence of symbols in `output`
        - It uses the left and right eigenmatrices to evaluate the trace of the resulting product
        """
        em_can = EMachine(self.can_r()**2)
        out_ls = list(output)
        mat_prod = np.diag(np.ones(self.dim))
        for out in reversed(out_ls):
            mat_prod = mat_prod @ np.conj(em_can.A[int(out)].T)
        mat_prod = mat_prod @ em_can.eig_l_mat
        for out in out_ls:
            mat_prod = mat_prod @ em_can.A[int(out)]
        return np.trace(mat_prod)
    
    def observable(self, O, l):
        """
        Compute the expectation value of a single-site observable propagated by the transfer operator.
        This method constructs the transfer matrix with a single-site operator insertion, raises that
        transfer matrix to the integer power l, and contracts with the stored left and right dominant
        eigenvectors to produce a scalar expectation value.
        
        Parameters
        ----------
        O : array_like, shape (p, p)
            Single-site operator acting on the physical index of the MPS tensor self.A. The first axis
            of O must match the physical dimension of self.A.
        l : int
            Non-negative integer number of transfer-operator steps to propagate (i.e. the exponent of
            the transfer matrix).
        Returns
        -------
        scalar
            A scalar (real or complex, depending on inputs) equal to
            <eig_left | (transfer_with_O)^l | eig_right>, where eig_left and eig_right are the stored
            left and right eigenmatrices (self.eig_l_mat, self.eig_r_mat) reshaped to vectors.
        Raises
        ------
        ValueError
            If l is negative, or if the shapes of O, self.A, or the eigenmodes are incompatible so that
            the required tensor contractions or reshapes cannot be performed.
        Notes
        -----
        - Expected tensor conventions: self.A is typically a rank-3 MPS tensor with shape (p, D, D)
          (physical dimension p, bond dimension D). After contracting the physical index with O and
          contracting with another copy of self.A, the resulting object is reshaped into a (D^2, D^2)
          transfer matrix.
        - self.eig_l_mat and self.eig_r_mat are expected to be arrays compatible with reshape(dim**2,)
          where dim equals the bond dimension D used to form the transfer matrix.
        - Computing a dense matrix power of the transfer matrix can be expensive for large D; for large
          exponents or bond dimensions, consider diagonalization or iterative/fast exponentiation methods.
        Example
        -------
        Assuming self.A has shape (p, D, D) and self.dim == D:
            O = np.eye(p)
            value = self.observable(O, 3)
        """
        X = np.tensordot(self.A, O.astype(self.A.dtype), axes = ([0],[0]))
        site = rearrange(np.tensordot(X, self.A.conj(), axes = ([2],[0])), 'i1 j1 i2 j2 -> (i1 i2) (j1 j2)')
        return np.dot(self.eig_l_mat.reshape(self.dim**2), np.linalg.matrix_power(site, l) @ self.eig_r_mat.reshape(self.dim**2))

    def correlation(self, O, l):
        """
        Two-point correlation function of observable O at separation l.

        Parameters
        ----------
        O : numpy.ndarray, shape (m, m)
            Observable operator defined over the observation symbols.
        l : int
            Separation distance for correlation computation.
        Returns
        -------
        float
            The two-point correlation of O at separation l under the stationary distribution.
        Notes
        -----
        - The method constructs a modified transfer site incorporating the observable O.
        """
        X = np.tensordot(self.A, O.astype(self.A.dtype), axes = ([0],[0]))
        site = rearrange(np.tensordot(X, self.A.conj(), axes = ([2],[0])), 'i1 j1 i2 j2 -> (i1 i2) (j1 j2)')
        two_corr = np.dot(self.eig_l_mat.reshape(self.dim**2), site @ np.linalg.matrix_power(self.E, l-1) @ site @ self.eig_r_mat.reshape(self.dim**2))
        mean_sq = self.observable(O, 1)**2
        return two_corr - mean_sq
    
    def interaction_set(self, n):
        """
        Generate the set of interaction matrices for sequences of length n.

        Parameters
        ----------
        n : int
            Length of the sequences for which to generate interaction matrices.
        Returns
        -------
        dict
            A dictionary mapping each sequence (as a tuple of integers) to its corresponding
            interaction matrix (numpy.ndarray).
        Notes
        -----
        - The method constructs interaction matrices by multiplying the A matrices
          corresponding to each symbol in the sequence.
        """
        results = {}
        for xs in itertools.product(range(self.mdim), repeat=n):
            # Start with identity on the bond dimension
            M = np.eye(self.dim)
            for x in xs:
                M = M @ self.A[x]
            results[xs] = M
        return results
    
    def interaction_rank(self, n):
        """
        Compute the rank of the set of interaction matrices for sequences of length n.

        Parameters
        ----------
        n : int
            Length of the sequences for which to compute the interaction rank.
        Returns
        -------
        int
            The rank of the set of interaction matrices.
        Notes
        -----
        - The method constructs interaction matrices for all sequences of length n
          and computes the rank of the resulting set.
        """
        interaction_matrices = self.interaction_set(n).values()
        stacked_matrices = np.array([mat.flatten() for mat in interaction_matrices])
        return np.linalg.matrix_rank(stacked_matrices)
    
    def ground_space(self, l):
        """
        Compute an orthonormal basis (the "ground space") for the space of single-site operator overlaps
        with the set of l-site interaction matrices.

        Parameters
        ----------
        l : int
            Interaction length / number of sites used to build the interaction set via
            self.interaction_set(l).
        Returns
        -------
        numpy.ndarray
            2-D array whose rows are orthonormal vectors spanning the space generated by
            the overlaps tr(M @ B) where M runs over matrices in self.interaction_set(l)
            and B runs over the canonical single-site operator basis {E_{ij}}.
            The returned array has shape (r, self.mdim**l) with r = min(self.dim**2, self.mdim**l)
            (the number of basis vectors produced by the reduced QR); note that the
            effective linear rank of the underlying overlap matrix may be lower than r.
        Notes
        -----
        - The method depends on the following instance attributes:
          - self.interaction_set(l): returns a mapping of keys -> matrices for l sites.
          - self.dim: single-site Hilbert-space dimension (used to form the single-site basis).
          - self.mdim: local dimension used to index multi-site configurations (used to size columns).
          - self.A.dtype: used to set the dtype of intermediate arrays.
        - Algorithm outline:
          1. Form the canonical single-site operator basis E_{ij} (matrices with a 1 at (i,j)).
          2. For each interaction matrix M in the interaction set and each E_{ij}, compute
             the scalar overlap tr(M @ E_{ij}) and store it in a matrix of shape
             (self.dim**2, self.mdim**l).
          3. Perform a reduced QR decomposition of the transpose of that matrix to
             obtain an orthonormal basis; return the transpose so that basis vectors are rows.
        - Complexity: dominated by filling the overlap matrix and the QR decomposition;
          roughly O(self.dim**2 * self.mdim**l) memory for the overlap matrix and
          comparable time complexity plus QR cost.
        Examples
        --------
        # returns an array of orthonormal row vectors for the given interaction length
        basis = self.ground_space(l)
        """
        inter_set = self.interaction_set(l)
        counter = np.array([self.dim**i for i in range(l)])
        basis = []
        for i in range(self.dim):
            for j in range(self.dim):
                M = np.zeros((self.dim, self.dim), dtype=self.A.dtype)
                M[i, j] = 1
                basis.append(M)
        ground_space = np.zeros((self.dim**2, self.mdim**l), dtype=self.A.dtype)
        for n, b in enumerate(basis):
            for key, mat in inter_set.items():
                idx = np.sum(np.array(key) * counter)
                ground_space[n, idx] += np.trace(mat @ np.array(b))
        ground_space, _ = np.linalg.qr(ground_space.T)
        return ground_space.T

    def to_ground_space(self, l, M):
        """
        Project a given matrix M into the ground-space basis for interaction length l.

        Parameters
        ----------  
        l : int
            Interaction length / number of sites used to build the interaction set via
            self.interaction_set(l).
        M : np.ndarray
            Matrix to be projected into the ground-space basis.
        Returns
        -------
        np.ndarray
            1-D array representing the projection of M into the ground-space basis.
            The array has shape (self.mdim**l,) and dtype matching self.A.dtype.
        Notes
        -----
        - The method depends on the following instance attributes:
        - self.interaction_set(l): returns a mapping of keys -> matrices for l sites.
        - self.dim: single-site Hilbert-space dimension (used to form the single-site basis).
        - self.mdim: local dimension used to index multi-site configurations (used to size output).
        - self.A.dtype: used to set the dtype of the output array.
        """
        if M.shape != (self.dim, self.dim):
            raise ValueError(f"Input matrix M must have shape ({self.dim}, {self.dim})")
        inter_set = self.interaction_set(l)
        counter = np.array([self.dim**i for i in range(l)])
        gs_vector = np.zeros(self.mdim**l, dtype=self.A.dtype)
        for key, mat in inter_set.items():
            idx = np.sum(np.array(key) * counter)
            gs_vector[idx] += np.trace(M @ np.array(mat))
        return gs_vector

    def parent_hamiltonian(self, l, reshape = True):
        """
        Compute the local parent Hamiltonian term for a contiguous block of length `l`.
        The parent Hamiltonian returned is the projector onto the orthogonal complement of the
        ground-space for the l-site block. Given a matrix G whose rows are ground-state
        vectors expressed in the full local basis of dimension mdim**l, the projector is
        where G^† denotes the conjugate transpose of G. P annihilates vectors in the ground-space
        and equals 1 on vectors orthogonal to that span.
        
        Parameters
        ----------
        l : int
            Number of contiguous sites in the block. Must be a positive integer.
            If True (default), return the operator reshaped into a 2l-dimensional tensor with
            interleaved bra/ket site indices ordered as
                (a0, b0, a1, b1, ..., a_{l-1}, b_{l-1}).
            If False, return the operator as a 2D matrix of shape (mdim**l, mdim**l).
        Returns
        -------
        np.ndarray
            The parent Hamiltonian:
              - If reshape is False: a 2D Hermitian array of shape (mdim**l, mdim**l).
              - If reshape is True: a 2l-dimensional array with shape (mdim,)*2l and axes
                ordered as (a0, b0, a1, b1, ..., a_{l-1}, b_{l-1}).
            The array dtype follows the computation (typically complex when ground-space vectors
            are complex).
            If `l` is not a positive integer, or if the array returned by self.ground_space(l)
            has an incompatible shape (the method expects an array G with shape (k, mdim**l)
            so that G^† G yields an (mdim**l, mdim**l) operator).
        Notes
        -----
        - The implementation assumes self.ground_space(l) returns ground-state vectors arranged
          as rows (shape (n_ground, mdim**l)). If the returned vectors are not orthonormal,
          G^† G will not be a true orthogonal projector onto the span; in that case orthonormalize
          the ground-space (e.g., via QR or SVD) before forming the projector.
        - When reshape=True the method uses einops.rearrange to map the flat matrix into a tensor
          with per-site bra/ket indices interleaved. This is purely a reshaping of data (no alteration
          of operator semantics).
        - The resulting operator is a local term that penalizes components outside the ground-space
          on the specified l-site block and can be used to assemble an overall parent Hamiltonian.
        Assuming self.mdim == 2 and l == 2:
        - If reshape=False, the result is a 4x4 matrix (numpy array) acting on the 2-site Hilbert space.
        - If reshape=True, the result has shape (2,2,2,2) with axes ordered as (a0, b0, a1, b1),
          corresponding to the same 4x4 operator reshaped into a two-site tensor.
        """
        ground_space = self.ground_space(l)
        ham = np.eye(self.mdim**l) - np.conj(ground_space.T) @ ground_space
        if reshape:
            target = str()
            a_idx = np.array(["a"+str(n) for n in range(l)]) #a0 a1 a2 ...
            b_idx = np.array(["b"+str(n) for n in range(l)]) #b0 b1 b2 ...
            for pair in reversed([f"{a} {b}" for a, b in zip(a_idx, b_idx)]):
                print(pair)
                target += pair + ' '
            a_sc = str("(")
            b_sc = str("(")
            for a, b in zip(a_idx, b_idx):
                a_sc += a + ' '
                b_sc += b + ' '
            a_sc = a_sc.rstrip()
            b_sc = b_sc.rstrip()
            a_sc += ")"
            b_sc += ")"
            expression = a_sc + b_sc + ' -> ' + target

            slc = {i: self.mdim for i in np.hstack([a_idx, b_idx])}
            print("Hamiltonian reshaped expression:", expression)
            return np.array(rearrange(ham, expression, **slc))
        return ham

    def decompose_mpo(self, mpo, l):
        """
        Decompose a full MPO (matrix representation of an l-site operator) into a list
        of local MPO site tensors using iterative singular value decompositions (SVDs).
        
        Parameters
        ----------
        mpo : ndarray
            Square matrix representing the full MPO acting on l sites. Expected shape
            is (mdim**l, mdim**l), equivalently (mdim * mdim**(l-1), mdim * mdim**(l-1)),
            where self.mdim is the physical dimension per site. The function internally
            treats mpo as shaped '(i*h) x (j*w)' with i=j=self.mdim and h=w=self.mdim**(l-1).
        l : int
            Number of sites to decompose (the MPO length). Must be >= 2 for a meaningful
            decomposition.
        Returns
        -------
        mpo_sites : list of ndarray
            A list of length l containing the site tensors obtained from the iterative
            SVD splitting. Indexing/order and shapes produced by this implementation:
              - mpo_sites[0]    has shape (mdim, r1, mdim)      (3 indices: phys-left, bond, phys-right)
              - mpo_sites[k]    for 0 < k < l-1 has shape (mdim, rk, r{k+1}, mdim) (4 indices)
              - mpo_sites[-1]   has shape (mdim, r_{l-1}, mdim) (3 indices)
            Here r1, r2, ... are the SVD ranks (bond dimensions) produced at each split.
            Note: no rank truncation is performed — bond dimensions equal full SVD ranks.
        Raises
        ------
        ValueError
            If l is less than 2 or otherwise inconsistent with the shape of mpo, the
            decomposition is not meaningful (the implementation may print an error or
            fail early).
        numpy.linalg.LinAlgError
            May be raised by the underlying SVD calls for ill-conditioned inputs.
        Notes
        -----
        - The routine relies on self.mdim (the physical dimension per site) being set
          on the instance; that value is used to reshape and interpret the global mpo.
        - The algorithm performs a sequence of bipartitionings using einops.rearrange
          patterns and full SVDs to peel off one site at a time from the operator.
        - Because no truncation is applied, intermediate bond dimensions can grow
          quickly (up to mdim**k for some k), so memory use may be large for big l.
        - Requires numpy and einops.rearrange to be available.
        Example
        -------
        # After setting self.mdim and having mpo as a (mdim**l x mdim**l) ndarray:
        mpo_sites = self.decompose_mpo(mpo, l)
        # mpo_sites[0].shape -> (mdim, r1, mdim)
        # mpo_sites[1].shape -> (mdim, r1, r2, mdim)
        # ...
        # mpo_sites[-1].shape -> (mdim, r_{l-1}, mdim)
        """
        mat = mpo
        mpo_sites = []
        mat = rearrange(mat, '(i h)(j w) -> (i j)(h w)', i = self.mdim, j = self.mdim)
        u, s, vh = np.linalg.svd(mat, full_matrices=False)
        mpo_sites.append(np.array(rearrange(u, '(i j) c1 -> i c1 j', i = self.mdim, j = self.mdim)))
        s_dim = s.shape[0]
        mat = np.diag(s) @ vh
        for _ in range(l-2):
            mat = rearrange(mat, 'c1 (i h j w) -> (c1 i j)(h w)', c1 = s_dim, i = self.mdim, j = self.mdim, h = self.mdim**(l-2-_), w = self.mdim**(l-2-_))
            u, s, vh = np.linalg.svd(mat, full_matrices=False)
            mpo_sites.append(np.array(rearrange(u, '(c1 i j) c2 -> i c1 c2 j', c1 = s_dim, i = self.mdim, j = self.mdim)))
            s_dim = s.shape[0]
            mat = np.diag(s) @ vh
            if _ == l-3:
                mpo_sites.append(np.array(rearrange(mat, 'c1 (i j) -> i c1 j', c1 = s_dim, i = self.mdim, j = self.mdim)))
                return mpo_sites      
   
    def mpo_action(self, mpo_sites):
        """
        Contract a list of MPO site tensors with the instance's A-tensors and return the assembled tensor.

        Parameters
        ----------
        mpo_sites : Sequence[np.ndarray]
            Sequence of L site tensors (L = number of MPO sites). Each site tensor must be indexable
            on its first axis with indices 0..self.mdim-1 (i.e. site[i] must exist for i in that range).
            The routine forms, for each site, the sum over i of np.tensordot(self.A[i], site[i], axes=0)
            and then contracts these per-site results together along the MPO bond indices to produce a
            single multi-index tensor.

        Returns
        -------
        np.ndarray
            The assembled tensor with shape
            (self.dim, b0, b1, ..., b_{L-1}, self.dim),
            where each b_k has length self.mdim. In other words, the left physical index (i),
            L MPO bond indices, and the right physical index (j).

        Notes
        -----
        - The function only requires that for each i in 0..self.mdim-1, the expression
          np.tensordot(self.A[i], site[i], axes=0) is valid and yields an array whose first two
          axes correspond to the physical input/output dimensions self.dim x self.dim. Any remaining
          axes are treated as MPO bond / on-site physical axes and are contracted across sites.
        - The exact per-site internal shapes can vary between sites (e.g. different on-site physical
          dimensions), but the first axis length of each site must equal self.mdim and the per-site
          contractions must be mutually consistent so that the sequential matrix multiplications
          performed later succeed.
        - If input shapes are inconsistent the method will raise or print an error when attempting
          the inter-site contractions.

        Raises
        ------
        ValueError
            If mpo_sites does not behave like a sequence of site tensors indexed by 0..self.mdim-1
            or if subsequent contractions fail due to incompatible shapes.

        Example
        -------
        - If self.dim = d, self.mdim = m and mpo_sites is a list of L arrays where each site has
          site.shape[0] == m and site[i] yields a tensor whose first two axes are length d,
          then the returned array will have shape (d, m, m, ..., m, d) with L occurrences of m.
        """
        res_sites = []
        l = len(mpo_sites)
        for site in mpo_sites:
            mat = np.tensordot(self.A, site, axes = ([0],[0])) #(a1,i1,i2)x(a1,c1,b1) -> (i1,i2,c1,b1) or (a1,i1,i2)x(a1,c1,c2,b1) -> (i1,i2,c1,c2,b1)
            res_sites.append(mat)
        res_tsr = rearrange(res_sites[0], 'i1 i2 c1 b1 -> i1 b1 i2 c1', i1 = self.dim, i2 = self.dim, b1 = self.mdim)
        try:
            for n, site in enumerate(res_sites[1:-1]):
                res_tsr = np.tensordot(res_tsr, site, axes = ([2, 3],[0, 2])) #(i1 b1 i2 c1)x(i2,i3,c1,c2,b2)-> (i1 b1 i3 c2 b2)
                res_tsr = rearrange(res_tsr, 'i1 b1 i3 c2 b2 -> i1 (b1 b2) i3 c2', i1 = self.dim, i3 = self.dim, b1 = self.mdim**(n + 1), b2 = self.mdim, c2 = site.shape[3])
            
            res_tsr = np.tensordot(res_tsr, res_sites[-1], axes = ([2, 3],[0, 2])) #(i1 b i3 c2)x(i3,i4,c2,b3) -> (i1 b i4 b3)
            res_tsr = rearrange(res_tsr, 'i1 b1 i4 b3 -> i1 (b1 b3) i4', i1 = self.dim, i4 = self.dim, b3 = self.mdim)
        
        except ValueError:
            print("Error: inconsistent shapes between MPO sites and A tensors")
            return None

        sites_str = str()
        for n in range(l):
            sites_str += f"b{n} "
        expression = 'i (' + sites_str + ')j ' + '-> i ' + sites_str + 'j'
        slc = {f"b{n}": self.mdim for n in range(l)}
        slc['i'] = self.dim
        slc['j'] = self.dim
        return np.array(rearrange(res_tsr, expression, **slc))

class PhasedEMachine(EMachine):
    """
    Subclass of EMachine that includes a phase parameter for each output symbol.
    The phase modifies the A-matrices by multiplying each A[x] by exp(i * phase[x]).
    """
    def __init__(self, A: np.ndarray, phases: np.ndarray):
        """
        Initialize the PhasedEMachine with given A-matrices and phases.

        Parameters
        ----------
        A : numpy.ndarray, shape (m, d, d)
            The set of A-matrices defining the E-machine.
        phases : numpy.ndarray, shape (m,)
            The phase parameters for each output symbol.
        """
        super().__init__(A.astype(np.complex128))
        self.phases = np.array(phases, dtype=np.complex128)
        for x in range(self.mdim):
            self.A[x] *= np.exp(1j * self.phases[x])
        #self.E = purify(np.tensordot(self.A, np.conj(self.A), axes = ([0],[0])).transpose(0,2,1,3).reshape(self.dim**2, self.dim**2))
        