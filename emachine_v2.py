import numpy as np
from scipy.linalg import logm, expm
import itertools
from einops import rearrange
import Centraliser as cen
from sympy import Matrix

def norm_l(vec):
    return vec / (np.sum(vec) + 1e-16)

def norm_r(vec):
    return vec/(vec[0,0] + 1e-16)

def purify(mat, tol = 1e-10, decimals=4):
    arr = np.asarray(mat)
    # Replace NaN and Inf with 0
    arr = np.where(np.isnan(arr) | np.isinf(arr), 0, arr)
    if np.iscomplexobj(arr):
        re = np.real(arr)
        im = np.imag(arr)
        re = np.where(np.abs(re) < tol, 0, re)
        im = np.where(np.abs(im) < tol, 0, im)
        result = re + 1j * im
        return np.round(result, decimals=decimals)
    result = np.where(np.abs(arr) < tol, 0, arr)
    return np.round(result, decimals=decimals)

def pprint(array):
    print(purify(array))

def even_process(p):
    A = np.array([[[p, 0],
                   [0,0]],
                   [[0, 1-p],
                    [1, 0]]])**(1/2)
    return EMachine(A)

def upset_gambler(p, q):
    A = np.array([[[0, p],
                   [q, 0]],
                   [[1-p, 0],
                    [1-q, 0]]])**(1/2)
    return EMachine(A)

def golden_mean(p):
    A = np.array([[[0, p],
                   [0, 0]],
                   [[1-p, 0],
                    [1, 0]]])**(1/2)
    return EMachine(A)

def aklt():
    A = np.array([[[0, np.sqrt(2/3)],
                   [0, 0]],
                   [[-1/np.sqrt(3), 0],
                    [0, 1/np.sqrt(3)]],
                    [[0, 0],
                     [-np.sqrt(2/3), 0]]])
    return MatrixProductState(A)

def ghz():
    A = np.array([[[1, 0],
                   [0, 0]],
                   [[0, 0],
                    [0, 1]]]).astype(np.complex128)
    return MatrixProductState(A)

def bird(p, q):
    A = np.array([[[1-p, 0],
                  [0,  1-q]],
                  [[0, p],
                   [q, 0]]])**(1/2)
    return EMachine(A)

def aklt_stoch(p = 2/3):
    A = np.array([[[0, p],
                   [0, 0]],
                   [[1-p, 0],
                    [0, 1-p]],
                    [[0, 0],
                     [p, 0]]])**(1/2)
    return EMachine(A)

def perturbed_coin(p, q):
    A = np.array([[[1-p, 0],
                   [q, 0]],
                   [[0, p],
                    [0, 1-q]]])**(1/2)
    return EMachine(A)
class SiteMatrix(np.ndarray):
    def __init__(self, input_array):
        self.mdin, self.dim, _ = input_array.shape

    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        return obj
    
    def __and__(self, other):
        if self.mdim != other.mdim:
            raise ValueError("Mismatched mdim")
        else:
            sum = np.zeros((self.mdim, self.dim + other.dim, self.dim + other.dim), dtype=self.dtype)
            sum[:, :self.dim, :other.dim] = self
            sum[:, self.dim:, self.dim:] = other
            return SiteMatrix(sum)

    def __xor__(self, other):
        return np.kron(self, other)
    
    def __matmul__(self, other):
        if self.dim != other.dim:
            raise ValueError("Matrix dimensions do not match for multiplication")
        else:
            mat = rearrange(np.tensordot(self, other, axes=([2],[1])), "x i1 y i3 -> (x y) i1 i3")
            return SiteMatrix(mat)
    
    def __pow__(self, power):
        if isinstance(power, int) and power >= 1:
            result = self
            for _ in range(1, power):
                result = result @ self
            return result
        else:
            raise ValueError("Power must be a non-negative integer")

class MatrixProductState:
    """
    MatrixProductState: A class for representing and manipulating matrix product states (MPS) and associated representations.
    This class encapsulates the mathematical structure of matrix product states commonly used in tensor network methods,
    density matrix renormalization group (DMRG), and other numerical algorithms in quantum many-body physics.
    Attributes
    A : SiteMatrix
        The MPS site tensor with shape (mdim, dim, dim), where mdim is the physical dimension and dim is the bond dimension.
    dim : int
        Bond dimension (size of virtual indices in the MPS tensor).
    mdim : int
        Physical dimension (size of physical index in the MPS tensor).
    E : np.ndarray
        Transfer matrix obtained from contracting A with its conjugate, reshaped to (dim^2, dim^2).
    eig_r : tuple
        Eigendecomposition of E (eigenvalues, eigenvectors).
    eig_r_mat : np.ndarray
        Right dominant eigenmatrix of E, normalized via norm_r. Shape (dim, dim).
    eig_l : tuple
        Eigendecomposition of E.T (eigenvalues, eigenvectors).
    eig_l_mat : np.ndarray
        Left dominant eigenmatrix of E.T, normalized via norm_l. Shape (dim, dim).
    w_r : np.ndarray
        Cholesky decomposition of eig_r_mat (for canonical form computation).
    w_l : np.ndarray
        Cholesky decomposition of eig_l_mat (for canonical form computation).
    U : np.ndarray
        Left unitary from SVD of w_l @ w_r.
    lam_v : np.ndarray
        Singular values from SVD of w_l @ w_r.
    V : np.ndarray
        Right unitary from SVD of w_l @ w_r.
    gam : np.ndarray
        Gamma tensor (site tensor in canonical gauge). Shape (dim, dim).
    lam : np.ndarray
        Lambda tensor (diagonal matrix of weights). Shape (dim, dim).
    - The class assumes that the dominant eigenmatrices are positive-definite for canonical form construction.
      If they are not, gam and lam will be set to None.
    - Many methods rely on the transfer matrix formalism and eigendecomposition stored in initialization.
    - Tensor contraction patterns follow physic conventions for MPS/MPO operations.
    Methods are organized into groups:
      - Canonical form construction: can_r, can_l
      - Spectral analysis: correlation_length
      - Measurement and observables: measure, measure_can, observable, correlation
      - Ground-space and parent Hamiltonian: ground_space, orthogonal_projector, parent_hamiltonian
      - MPO decomposition and contraction: decompose_mpo, mpo_action
      - Symmetry analysis: apply_symmetry, apply_generator, sun_symmetry, son_symmetry,
                           virtual_symmetry, virtual_symmetry_gen, find_generators_coefficients
      - Utility methods: interaction_rank, to_ground_space, mps_inverse
        """

    def __init__(self, A : SiteMatrix):
        self.A = SiteMatrix(A)
        self.dim = A.shape[1]
        self.mdim = A.shape[0]
        self.E = np.tensordot(self.A, self.A.conj(), axes = ([0],[0])).transpose(0,2,1,3).reshape(self.dim**2, self.dim**2)

        self.eig_r = np.linalg.eig(self.E)
        leading_idx_r = np.argmax(np.abs(self.eig_r[0]))
        self.eig_r_mat = norm_r(rearrange(self.eig_r[1][:, leading_idx_r], '(a b) -> a b', a=self.dim, b=self.dim))
        self.eig_l = np.linalg.eig(self.E.T)
        leading_idx_l = np.argmax(np.abs(self.eig_l[0]))
        self.eig_l_mat = norm_l(rearrange(self.eig_l[1][:, leading_idx_l], '(a b) -> a b', a=self.dim, b=self.dim))
        try:
            self.w_r = np.linalg.cholesky(self.eig_r_mat + np.eye(self.dim)*1e-12)
            self.w_l = np.linalg.cholesky(self.eig_l_mat + np.eye(self.dim)*1e-12)

            self.U, self.lam_v, self.V = np.linalg.svd(self.w_l @ self.w_r)

            self.gam = self.V @ np.linalg.inv(self.w_r) @ np.array(self.A) @ np.linalg.inv(self.w_l) @ self.U
            self.lam = np.diag(self.lam_v)
        except:
            print("Warning: dominant eigenmatrices are not positive definite, cannot compute canonical form")
            self.gam = None
            self.lam = None

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
        return MatrixProductState(A_can)  
        
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
        return MatrixProductState(A_can)
    
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
    
    def measure(self, output: str):
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
    
    def interaction_rank(self, n, row_e = False):
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
        interaction_matrices = self.A**n
        stacked_matrices = np.array([mat.flatten() for mat in interaction_matrices])
        if row_e:
            print(Matrix(stacked_matrices).rref())
        return np.linalg.matrix_rank(stacked_matrices)

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
        
        return np.tensordot(self.A**l, M, axes=([1,2],[0,1]))

    def ground_space(self, l):
        mps = self.A**l
        gs = np.zeros((self.dim**2, self.mdim**l), dtype=self.A.dtype)
        basis = []
        for i in range(self.dim):
            for j in range(self.dim):
                M = np.zeros((self.dim, self.dim), dtype=self.A.dtype)
                M[i, j] = 1
                basis.append(M)
        for i, b in enumerate(basis):
            gs[i,:] = np.tensordot(mps, b, axes=([2,1],[0,1]))
        q, _ = np.linalg.qr(gs.T)
        return q.T
    
    def orthogonal_projector(self, l, reshape = True):
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
    
    def parent_hamiltonian(self, l, L):
        h = np.zeros((self.mdim**L, self.mdim**L), dtype=self.A.dtype)
        p = self.orthogonal_projector(l, reshape = False)
        for i in range(1, L - l + 2):
            if i == 1:
                s = np.kron(p, np.eye(self.mdim**(L - l)))
            elif i == L - l + 1:
                s = np.kron(np.eye(self.mdim**(L - l)), p)
            else:
                s = np.kron(np.eye(self.mdim**(i-1)), np.kron(p, np.eye(self.mdim**(L - i - l + 1))))
            h += s
        return h
    
    def mps_inverse(self, l):
        mps = np.array(rearrange(self.A**l, 'x i1 i2 -> x (i1 i2)', i1 = self.dim, i2 = self.dim))
        inv = np.linalg.inv(mps.T@mps)@mps.T
        return rearrange(inv, ' (i1 i2) x -> i1 i2 x', i1 = self.dim, i2 = self.dim)
    
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
        if l == 2:
            mpo_sites.append(mat)
            return mpo_sites
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
    
    def sun_symmetry(self, ham, return_coeff=True, full_output=True):
        tpg = cen.TensorProductGroup(N = self.mdim, D = int(np.log(ham.shape[0])/np.log(self.mdim)))
        return tpg.find_symmetric_generators(ham, return_coeff=return_coeff, verbose=full_output)

    def son_symmetry(self, ham, return_coeff=True, full_output=True):
        tpg = cen.TensorProductGroupSO(N = self.mdim, D = int(np.log(ham.shape[0])/np.log(self.mdim)))
        return tpg.find_symmetric_generators(ham, return_coeff=return_coeff, verbose=full_output, orthogonal=True)

    def apply_symmetry(self, symmetry):
        return MatrixProductState(np.tensordot(symmetry, self.A, axes=([1], [0])))

    def apply_generator(self, generator, theta=1.0):
        """
        Apply a symmetry transformation by exponentiating a generator and acting on the physical index.
        
        Parameters
        ----------
        generator : array_like, shape (mdim, mdim)
            Generator matrix in su(N) representation (traceless, Hermitian matrix).
        theta : float, optional
            Parameter for the exponential map U = exp(i*theta*generator). Default is 1.0.
        
        Returns
        -------
        MatrixProductState
            A new MatrixProductState with the unitary transformation applied to the physical index.
        
        Notes
        -----
        - The generator should be a traceless Hermitian matrix representing an element of the
          Lie algebra su(N), where N is the physical dimension (self.mdim).
        - The unitary operator U = exp(i*theta*generator) is computed using matrix exponential.
        - The transformation is applied to the physical index via tensor contraction:
          A'[p,i,j] = sum_q U[p,q] * A[q,i,j]
        
        Example
        -------
        # Apply a Pauli-X rotation
        mps_transformed = mps.apply_generator(sigma_x, theta=np.pi/4)
        """
        # Compute unitary transformation U = exp(i * theta * generator)
        U = expm(1j * theta * generator)
        
        # Apply U to physical index: A'[p,i,j] = sum_q U[p,q] * A[q,i,j]
        # This is equivalent to contracting U with the physical index (axis 0) of self.A
        A_new = np.tensordot(U, self.A, axes=([1], [0]))
        
        # Return new MatrixProductState
        return MatrixProductState(A_new)

    def virtual_symmetry(self, ham, verbose=False):
        """
        Find the virtual (bond) index generators K and phases φ for each physical symmetry generator.
        
        Given a physical space generator G (acting on all D sites as G^⊗D), finds the
        corresponding virtual space generator K (acting on bond indices) such that:
        
            G^⊗D = (G ⊗ I ⊗ ... ⊗ I) acts physically
        
        Maps this to: A[i] * K = G[i,j] * A[j] (in canonical gauge)
        
        Parameters:
            ham: Target Hamiltonian (mdim^D × mdim^D)
            verbose: If True, print diagnostic information
        
        Returns:
            List of dicts with keys:
            - 'generator': Virtual index generator K (dim × dim)
            - 'phase': Local phase φ (should be ~0)
            - 'generator_matrix': Original physical generator (mdim × mdim single-site)
            - 'error': Residual ||A*K - G*A|| (for validation)
            - 'rel_error': Relative error (normalized by generator norm)
        """
        em_can = self.can_l()
        can_A = np.array(em_can.A)
        vgen = []
        
        # Get physical symmetry generators (these are single-site, not full D-body)
        phys_generators = self.sun_symmetry(ham, full_output=False, return_coeff=False)
        
        for g in phys_generators:
            flat_A = can_A.reshape(self.mdim, -1)  # Shape (mdim, dim^2)
            M = [np.kron(a, np.eye(self.dim)) - np.kron(np.eye(self.dim), a.T) for a in can_A]
            b = np.tensordot(g, can_A, axes=([1], [0])).reshape(self.mdim, -1)
            M_adj = [np.zeros((self.dim**2, self.dim**2 + 1), dtype=g.dtype) for _ in range(len(M))]
            for i, m in enumerate(M):
                M_adj[i][:,:-1] = m
                M_adj[i][:,-1] = flat_A[i]

            # Stack all equations vertically
            M_stacked = np.vstack(M_adj)  # Shape: (mdim * dim^2, dim^2 + 1)
            b_stacked = np.concatenate([b[i] for i in range(len(b))], axis=0)  # Shape: (mdim * dim^2,)
            
            # Solve for [K_flat; phase] using least squares
            solution, residuals, rank, s = np.linalg.lstsq(M_stacked, b_stacked, rcond=None)
            # Extract K and phase
            K_flat = solution[:-1]  # First dim^2 elements
            phase = solution[-1]     # Last element
            K = K_flat.reshape(self.dim, self.dim)  # Reshape back to matrix
            
            # Compute error using original equation: ||b - M@K - phase*A||
            error = 0.0
            for i in range(self.mdim):
                residual = b[i] - M[i] @ K_flat - phase * flat_A[i]
                error += np.linalg.norm(residual)**2
            error = np.sqrt(error)
            
            # Relative error
            g_norm = np.linalg.norm(g)
            b_norm = np.linalg.norm(b_stacked)
            rel_error = error / (b_norm + 1e-16)
            
            if verbose:
                print(f"\nPhysical generator G:\n{purify(g)}")
                print(f"\nVirtual generator found:")
                print(f"  Phase: {phase:.6f}")
                print(f"  Error: {error:.6e}")
                print(f"  Relative error: {rel_error:.6e}")
                print(f"  Generator K:\n{purify(K)}")
            
            vgen.append({
                'generator': K,
                'phase': phase,
                'generator_matrix': g,
                'error': error,
                'rel_error': rel_error
            })
        
        return vgen
    
    def virtual_symmetry_gen(self, gen, verbose = False):
        em_can = self.can_l()
        can_A = np.array(em_can.A)
        vgen = []
        for g in gen:
            flat_A = can_A.reshape(self.mdim, -1)  # Shape (mdim, dim^2)
            M = [np.kron(a, np.eye(self.dim)) - np.kron(np.eye(self.dim), a.T) for a in can_A]
            b = np.tensordot(g, can_A, axes=([1], [0])).reshape(self.mdim, -1)
            M_adj = [np.zeros((self.dim**2, self.dim**2 + 1), dtype=g.dtype) for _ in range(len(M))]
            for i, m in enumerate(M):
                M_adj[i][:,:-1] = m
                M_adj[i][:,-1] = flat_A[i]

            # Stack all equations vertically
            M_stacked = np.vstack(M_adj)  # Shape: (mdim * dim^2, dim^2 + 1)
            b_stacked = np.concatenate([b[i] for i in range(len(b))], axis=0)  # Shape: (mdim * dim^2,)
            
            # Solve for [K_flat; phase] using least squares
            solution, residuals, rank, s = np.linalg.lstsq(M_stacked, b_stacked, rcond=None)
            # Extract K and phase
            K_flat = solution[:-1]  # First dim^2 elements
            phase = solution[-1]     # Last element
            K = K_flat.reshape(self.dim, self.dim)  # Reshape back to matrix
            
            # Compute error using original equation: ||b - M@K - phase*A||
            error = 0.0
            for i in range(self.mdim):
                residual = b[i] - M[i] @ K_flat - phase * flat_A[i]
                error += np.linalg.norm(residual)**2
            error = np.sqrt(error)
            
            # Relative error
            g_norm = np.linalg.norm(g)
            b_norm = np.linalg.norm(b_stacked)
            rel_error = error / (b_norm + 1e-16)
            
            if verbose:
                print(f"\nPhysical generator G:\n{purify(g)}")
                print(f"\nVirtual generator found:")
                print(f"  Phase: {phase:.6f}")
                print(f"  Error: {error:.6e}")
                print(f"  Relative error: {rel_error:.6e}")
                print(f"  Generator K:\n{purify(K)}")
            
            vgen.append({
                'generator': K,
                'phase': phase,
                'generator_matrix': g,
                'error': error,
                'rel_error': rel_error
            })
        
        return vgen

    def find_generators_coefficients(self, M, basis, verbose=True, dim = "physical", reference_coeffs=None):
        """
        Find site-local generator coefficients for a matrix M.
        
        Decomposes M into site-local generators:
        M = exp(i * sum_{s=1}^D sum_a θ_a^{(s)} * T^a^{(s)})
        
        Args:
            M: Matrix in SU(N)^⊗D (must be unitary with det ≈ 1)
            basis: List of basis generators for SU(N) (e.g., Gell-Mann matrices)
            verbose: If True, print decomposition details
            reference_coeffs: Optional reference coefficients to unwrap to the nearest branch
            
        Returns:
            coefficients: Array of shape (D, n_generators) where coefficients[s, a]
                         is θ_a for site s, or None if M is not in the group
        """
        if dim == "physical":
            tpg = cen.TensorProductGroup(self.mdim, 1)
        elif dim == "virtual":
            tpg = cen.TensorProductGroup(self.dim, 1)
        else:
            raise ValueError("dim must be 'physical' or 'virtual'")
        coefficients = tpg.find_generators_coefficients(M, verbose=verbose, basis_generators=basis, reference_coeffs=reference_coeffs)
        return coefficients
    
class EMachine(MatrixProductState):
    """
    EMachine class for analyzing classical stochastic processes through a tensor-based framework.
    This class extends MatrixProductState and provides methods for computing various information-theoretic
    and dynamical quantities from a set of transition matrices. It is designed to characterize the memory,
    correlations, and statistical properties of hidden Markov models and related classical systems.
    Attributes
    A : numpy.ndarray
        Tensor of shape (m, n, n) containing m matrices of size n×n, inherited from MatrixProductState.
    T : numpy.ndarray
        Tensor derived from A, computed as the element-wise square of A.
    Q : numpy.ndarray
        Shape (n, n), the sum of T over the first axis, representing the classical transition matrix.
    eig_l_classical : tuple
        Eigenvalue decomposition of Q.T (left eigendecomposition of Q).
        Contains (eigenvalues, eigenvectors).
    eig_r_classical : tuple
        Eigenvalue decomposition of Q (right eigendecomposition).
        Contains (eigenvalues, eigenvectors).
    B : numpy.ndarray
        Shape (n, n), the sum of T transposed along axes (2, 1, 0), representing emission probabilities.
    dim : int
        Dimension of the state space (inherited from MatrixProductState).
    Methods
    unitary()
        Return Hermitian similarity transforms of matrices in A using w_r.
    density()
    quantum_statistical_memory()
        Compute von Neumann entropy of the density matrix.
    topological_memory()
        Compute log2 of the rank of the density matrix.
    statistical_memory()
        Compute Shannon entropy of the diagonal of eig_l_mat.
    correlation_length()
        Calculate correlation length from eigenvalues of E-matrix.
    propagator(iterations, state)
        Stochastic sampler evolving a classical state and returning emitted symbols.
    state_distribution(n)
        Compute classical transition matrix raised to power n via eigendecomposition.
    emission_distribution(n)
        Compute emission distribution at time step n.
    mean(f)
        Compute expectation value of observable f under stationary distribution.
    asymptotic_variance(f)
        Calculate asymptotic variance accounting for temporal correlations.
    relation_probability(init_state, t, s)
        Compute relation probability matrix between states at two time steps.
    """
    
    def __init__(self, A):
        super().__init__(A)
        self.T = np.array(self.A)**2  # Assuming A is derived from T via elementwise square root
        self.Q = np.sum(self.T, axis=0)
        self.eig_l_classical = np.linalg.eig(self.Q.T)
        self.eig_r_classical = np.linalg.eig(self.Q)
        self.B = np.sum(self.T.transpose(2, 1, 0) , axis = 0)

    def __repr__(self):
        return f"EMachine(dim={self.dim}, A_shape={self.A.shape})"
    
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

    def asymptotic_variance(self, f):
        """
        Calculate the asymptotic variance of a function with respect to the stationary distribution.
        This method computes the asymptotic variance for a given function f by combining:
        1. The variance under the stationary distribution
        2. A covariance term derived from the spectral decomposition of the transition matrix
        The asymptotic variance accounts for correlations in the Markov chain through
        the eigenvalues and eigenvectors of the transition matrix.
        Parameters
        ----------
        f : array-like
            A function or array of values evaluated at each state for which to compute
            the asymptotic variance.
        Returns
        -------
        float
            The asymptotic variance of function f, computed as:
            var + 2 * cov
            where var is the variance under the stationary distribution and cov is
            the covariance correction term accounting for temporal correlations.
        Notes
        -----
        - Requires eigenvalue decomposition of the transition matrix (eig_r_classical, eig_l_classical)
        - Excludes eigenvalue 1 (stationary eigenvalue) from the computation
        - Uses stationary probability distribution (eig_l_mat.diagonal())
        """
        
        
        mean_f = self.mean(f)
        stat_prob = self.eig_l_mat.diagonal()
        var = np.dot(stat_prob @ self.B, (f - mean_f)**2)
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
    


class PhasedEMachine(EMachine):
    """
    Subclass of EMachine that includes a phase parameter for each output symbol.
    The phase modifies the A-matrices by multiplying each A[x] by exp(i * phase[x]).
    """
    def __init__(self, A: SiteMatrix, phases: np.ndarray):
        """
        Initialize the PhasedEMachine with given A-matrices and phases.

        Parameters
        ----------
        A : numpy.ndarray, shape (m, d, d)
            The set of A-matrices defining the E-machine.
        phases : numpy.ndarray, shape (m,)
            The phase parameters for each output symbol.
        """
        self.phases = np.array(phases, dtype=np.complex128)
        self.A = SiteMatrix(np.einsum('ij,ijk->ijk', np.exp(1j * self.phases), A))
        super().__init__(self.A.astype(np.complex128))
        #self.E = purify(np.tensordot(self.A, np.conj(self.A), axes = ([0],[0])).transpose(0,2,1,3).reshape(self.dim**2, self.dim**2))
