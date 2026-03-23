"""
Compute the centralizer of a matrix under SU(N)^⊗D
Generalized version supporting any SU(N) group and D-fold tensor products

IMPORTANT DISTINCTION:
======================

1. find_centralizer_special_basis(M):
   - Finds ONLY discrete basis elements that commute with M
   - Returns a FINITE set
   - Does NOT generate the full continuous centralizer subgroup
   
2. find_lie_algebra_generators(M):
   - Finds generators of the FULL continuous centralizer Lie algebra
   - These span all infinitesimal transformations that commute with M
   - Use expm() on linear combinations to generate ANY element of the centralizer
   - Dimension = number of independent generators (typically < (N²-1)*2^D)
   
3. generate_from_lie_algebra(generators):
   - Samples the continuous centralizer by exponentiating Lie algebra elements
   - Produces INFINITELY many distinct group elements
   - Guaranteed to centralize M (up to numerical precision)

RECOMMENDATION:
===============
For complete characterization of the centralizer, use find_lie_algebra_generators()
which gives you the generators needed to construct the entire centralizer subgroup.
"""

import numpy as np
from itertools import product
from scipy.linalg import expm, null_space, logm
from sympy import Matrix

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

class SUN:
    """Parametrization of SU(N) groups using Gell-Mann matrices and generalizations"""
    
    @staticmethod
    def gell_mann_basis(N):
        """
        Generate the Gell-Mann basis matrices for SU(N).
        
        Returns N²-1 traceless, Hermitian matrices that form a basis for su(N).
        These are the generalization of Pauli matrices to higher dimensions.
        
        Args:
            N: Dimension of the group (2 for SU(2), 3 for SU(3), etc)
            
        Returns:
            List of N²-1 basis matrices for su(N)
        """
        if N < 2:
            raise ValueError("N must be >= 2 for SU(N)")
        
        basis = []
        
        # Symmetric matrices (off-diagonal, symmetric part)
        for i in range(N):
            for j in range(i+1, N):
                M = np.zeros((N, N), dtype=complex)
                M[i, j] = 1
                M[j, i] = 1
                basis.append(M)
        
        # Antisymmetric matrices (off-diagonal, antisymmetric part)
        for i in range(N):
            for j in range(i+1, N):
                M = np.zeros((N, N), dtype=complex)
                M[i, j] = -1j
                M[j, i] = 1j
                basis.append(M)
        
        # Diagonal matrices (N-1 independent traceless diagonal)
        for k in range(N-1):
            M = np.zeros((N, N), dtype=complex)
            for i in range(k+1):
                M[i, i] = 1
            M[k+1, k+1] = -(k+1)
            # Normalize: proper normalization for diagonal elements
            # Tr(M^2) = (k+1) + (k+1)^2 = (k+1)(k+2)
            norm = np.sqrt((k+1) * (k+2))
            M = M / norm
            basis.append(M)
        
        return basis
    
    @staticmethod
    def identity(N):
        """Return the NxN identity matrix"""
        return np.eye(N, dtype=complex)
    
    @staticmethod
    def from_euler_angles(N, angles):
        """
        Generate SU(N) element from angles using successive 2D rotations.
        
        Args:
            N: Dimension of the group
            angles: List of angles for rotations in 2D subspaces
            
        Returns:
            An NxN unitary matrix in SU(N)
        """
        U = np.eye(N, dtype=complex)
        angle_idx = 0
        
        for i in range(N):
            for j in range(i+1, N):
                if angle_idx >= len(angles):
                    break
                theta = angles[angle_idx]
                # 2D rotation in the (i,j) plane
                U_ij = np.eye(N, dtype=complex)
                U_ij[i, i] = np.cos(theta/2)
                U_ij[i, j] = -1j * np.sin(theta/2)
                U_ij[j, i] = 1j * np.sin(theta/2)
                U_ij[j, j] = np.cos(theta/2)
                U = U @ U_ij
                angle_idx += 1
        
        return U


class SON:
    """Parametrization of SO(N) groups using real antisymmetric Lie algebra basis"""

    @staticmethod
    def so_basis(N):
        """
        Generate a basis for so(N): real antisymmetric NxN matrices.

        Basis elements are E_ij - E_ji for i < j.

        Args:
            N: Dimension of the group (N >= 2)

        Returns:
            List of N(N-1)/2 basis matrices for so(N)
        """
        if N < 2:
            raise ValueError("N must be >= 2 for SO(N)")

        basis = []
        for i in range(N):
            for j in range(i + 1, N):
                M = np.zeros((N, N), dtype=float)
                M[i, j] = 1.0
                M[j, i] = -1.0
                basis.append(M)
        return basis

    @staticmethod
    def identity(N):
        """Return the NxN identity matrix"""
        return np.eye(N, dtype=float)

    @staticmethod
    def from_plane_angles(N, angles):
        """
        Generate SO(N) element from plane-rotation angles.

        Args:
            N: Dimension of the group
            angles: List of rotation angles for planes (i,j), i<j, in lexicographic order

        Returns:
            An NxN orthogonal matrix in SO(N)
        """
        U = np.eye(N, dtype=float)
        angle_idx = 0

        for i in range(N):
            for j in range(i + 1, N):
                if angle_idx >= len(angles):
                    break
                theta = angles[angle_idx]
                U_ij = np.eye(N, dtype=float)
                U_ij[i, i] = np.cos(theta)
                U_ij[j, j] = np.cos(theta)
                U_ij[i, j] = -np.sin(theta)
                U_ij[j, i] = np.sin(theta)
                U = U @ U_ij
                angle_idx += 1

        if np.linalg.det(U) < 0:
            U[:, -1] *= -1
        return U

class PermutationGroup:
    """Work with S_D permutation group acting on D-fold tensor products"""
    
    @staticmethod
    def permutation_matrix(perm, D):
        """
        Generate the D x D permutation matrix for a given permutation of D elements.
        Args:
            perm: A list or array representing the permutation of [0, 1, ..., D-1]
            D: Size of the permutation (matrix will be D x D)
        Returns:
            A (D x D) permutation matrix that permutes the standard basis according to 'perm'
        """
        if sorted(perm) != list(range(D)):
            raise ValueError("Invalid permutation")
        P = np.zeros((D, D), dtype=complex)
        for i in range(D):
            P[perm[i], i] = 1
        return P

    @staticmethod
    def all_permutation_matrices(D):
        """
        Generate all permutation matrices for S_D acting on D-fold tensor products (N=2).
        Returns a list of permutation matrices (as numpy arrays).
        """
        import itertools
        perms = list(itertools.permutations(range(D)))
        matrices = [PermutationGroup.permutation_matrix(list(perm), D) for perm in perms]
        return matrices


class TensorProductGroup:
    """Work with U^⊗D tensor product group for SU(N)"""
    
    def __init__(self, N, D):
        """
        Initialize for D-fold tensor products of SU(N).
        
        Args:
            N: Dimension of each local group (2 for SU(2), 3 for SU(3), etc)
            D: Number of tensor product factors
        """
        self.N = N
        self.D = D
        self.local_dim = N
        self.hilbert_dim = N ** D
        self.basis = SUN.gell_mann_basis(N)
        self.n_generators = len(self.basis)
        self.identity_local = SUN.identity(N)
    
    def tensor_product(self, *matrices):
        """Compute tensor product of matrices"""
        if len(matrices) == 0:
            return np.eye(self.hilbert_dim, dtype=complex)
        result = matrices[0]
        for mat in matrices[1:]:
            result = np.kron(result, mat)
        return result
    
    @staticmethod
    def commutator(A, B, tolerance=1e-10):
        """Compute [A, B] = AB - BA and check if zero"""
        comm = A @ B - B @ A
        return comm
    
    @staticmethod
    def jacobi_identity(A, B, C):
        """Compute the Jacobi identity [A, [B, C]] + [B, [C, A]] + [C, [A, B]]"""
        com = TensorProductGroup.commutator
        return com(A, com(B, C)) + com(B, com(C, A)) + com(C, com(A, B))

    @staticmethod
    def anticommutator(A, B):
        """Compute {A, B} = AB + BA"""
        return A @ B + B @ A
    
    def find_symmetric_generators(self, M, tol=1e-10, return_coeff=True, row_echelon=False, verbose=True):
        """
        Find generators that commute with M.
        
        Args:
            M: Matrix to find commuting generators for
            tol: Tolerance for null space detection
            return_coeff: If True, return coefficients of generators.
                     If False, return numeric generators (sum of basis elements).
            row_echelon: If True, print row echelon form of commutator matrix
            verbose: If True, print detailed output. If False, minimal output.
        
        Returns:
            If return_coeff=True:
                List of coefficient vectors where generator = sum(coeff[i] * basis[i])
            If return_coeff=False:
                List of numeric generator matrices
        """
        # Build commutator matrix
        comm = np.zeros((M.shape[0]**2, self.n_generators), dtype=complex)
        for i, gen in enumerate(self.basis):
            genfull = np.zeros((self.hilbert_dim, self.hilbert_dim), dtype=complex)
            for pos in range(self.D):
                if pos == 0:
                    genterm = np.kron(gen, SUN.identity(self.N ** (self.D - 1)))
                elif pos == self.D - 1:
                    genterm = np.kron(SUN.identity(self.N ** (self.D - 1)), gen)
                else: 
                    genterm = np.kron(np.kron(SUN.identity(self.N**(pos)), gen), SUN.identity(self.N**(self.D - pos - 1)))
                genfull += genterm
            comm[:, i] = (genfull @ M - M @ genfull).flatten()
        
        # Find null space of commutators
        if row_echelon:
            print("Row echelon form of comm matrix:")
            pprint(np.array(Matrix(comm.T).rref()[0]))
        
        null_vecs = null_space(comm)
        
        # Print output display
        if verbose:
            print("\n" + "="*80)
            print("SYMMETRIC GENERATORS ANALYSIS")
            print("="*80)
        
        # Check if null space is empty
        if null_vecs.shape[1] == 0:
            if verbose:
                print("\n⚠️  No symmetry found: The null space is empty.")
                print("="*80 + "\n")
            return np.array([])
        
        # Re-orthonormalize null space using QR decomposition
        Q, R = np.linalg.qr(null_vecs)
        null_vecs_ortho = Q
        
        # Enforce real coefficients for Hermitian generators
        null_vecs_ortho_real = np.real(null_vecs_ortho)
        
        # Re-orthonormalize after enforcing reality
        Q_real, R_real = np.linalg.qr(null_vecs_ortho_real)
        null_vecs_ortho = Q_real
        
        # Print generators with index
        if verbose:
            print(f"\nFound {null_vecs_ortho.shape[1]} generator(s) in the null space:")
            print("-" * 80)
        
        # Filter out near-zero vectors
        result = []
        valid_indices = []
        
        for i in range(null_vecs_ortho.shape[1]):
            coeff_vec = null_vecs_ortho[:, i]
            
            # Coefficients are now guaranteed real
            coeff_vec = np.real(coeff_vec)
            
            if np.linalg.norm(coeff_vec) > tol:
                valid_indices.append(i)
                if verbose:
                    print(f"\nGenerator {len(valid_indices)}:")
                    print(f"  Null space basis vector index: {i}")
                    print(f"  Coefficients in local su({self.N}) basis (orthonormalized): {purify(coeff_vec)}")
                    
                    # Print the corresponding su(N) basis generators
                    print(f"\n  Basis generators in su({self.N}) (with non-zero coefficients):")
                    for j in range(self.n_generators):
                        coeff = coeff_vec[j]
                        if np.abs(coeff) > tol:
                            print(f"    [{j}] (coeff: {purify(coeff)}):")
                            print(purify(self.basis[j]))
                
                if return_coeff:
                    result.append(coeff_vec)
                else:
                    # Reconstruct local generator in su(N): sum of coeff[i] * basis[i]
                    gen_matrix_local = np.zeros((self.N, self.N), dtype=complex)
                    for j in range(self.n_generators):
                        gen_matrix_local += coeff_vec[j] * self.basis[j]
                    
                    # Enforce Hermiticity on the reconstructed generator
                    gen_matrix_local = (gen_matrix_local + gen_matrix_local.conj().T) / 2
                    
                    result.append(gen_matrix_local)
                    # Print generator matrix only when verbose=True
                    if verbose:
                        print(f"\n  Local su({self.N}) generator (Hermitian enforced):\n{purify(gen_matrix_local)}")
        
        # Print null space basis
        if verbose:
            print("\n" + "-" * 80)
            print(f"\nOrthonormalized null space basis ({null_vecs_ortho.shape[1]} vectors):")
            for i in range(null_vecs_ortho.shape[1]):
                vec_norm = np.linalg.norm(null_vecs_ortho[:, i])
                print(f"\nBasis vector {i+1} (norm = {vec_norm:.4f}):")
                print(purify(null_vecs_ortho[:, i]))
            
            print("\n" + "="*80 + "\n")
        
        # Convert result list to numpy array
        if len(result) == 0:
            return np.array([])
        else:
            return np.array(result)

    def find_generators_coefficients(self, M, verbose=True, basis_generators=None, reference_coeffs=None):
        """
        Given a matrix M in SU(N)^⊗D, find coefficients θ_a such that:
        M = exp(i * sum_a θ_a * G_a)

        where G_a are the basis generators.

        This uses the matrix logarithm: log(M) = i * sum_a θ_a * G_a

        Args:
            M: Matrix in SU(N)^⊗D (must be unitary with det ≈ 1)
            verbose: If True, print decomposition details
            basis_generators: Choice of basis generators:
                - None (default): Use full site-local basis (D × n_generators total)
                - "symmetry": Automatically find symmetry generators of M
                - List/array of matrices: Use provided generators as basis (local su(N) form)
            reference_coeffs: Optional reference coefficients to unwrap to the nearest branch.
                If provided, the function will adjust the recovered coefficients to be closest
                to this reference (useful for multi-valued matrix logarithm).
        
        Returns:
            coefficients: Array of coefficients in the chosen basis, or None if M is not in the group
                - If basis_generators=None: shape (D, n_generators) for site-local coefficients
                - If basis_generators="symmetry" or list: shape (n_basis_generators,) for direct basis
        """
        # Check if M is unitary
        if not np.allclose(M @ M.conj().T, np.eye(M.shape[0]), atol=1e-8):
            if verbose:
                print("⚠️  Warning: M is not unitary")
            return None

        # Check if det(M) ≈ 1 (up to a global phase)
        det_M = np.linalg.det(M)
        if not np.allclose(np.abs(det_M), 1.0, atol=1e-8):
            if verbose:
                print(f"⚠️  Warning: |det(M)| = {np.abs(det_M):.6f} ≠ 1")
            return None

        # Compute matrix logarithm
        log_M = logm(M)

        # Remove global phase: log(M) should be traceless for SU(N)
        trace_log = np.trace(log_M)
        log_M_traceless = log_M - (trace_log / M.shape[0]) * np.eye(M.shape[0], dtype=complex)

        # Extract Lie algebra element: log(M) = i * H where H is Hermitian
        H = -1j * log_M_traceless
            
        # Enforce Hermiticity
        H = (H + H.conj().T) / 2

        # Determine which basis to use
        use_custom_basis = False
        local_basis_generators = None
        
        if basis_generators is None:
            # Use full site-local basis (default)
            use_custom_basis = False
            basis_name = "site-local"
            
        elif isinstance(basis_generators, str) and basis_generators.lower() == "symmetry":
            # Automatically find symmetry generators
            if verbose:
                print("Finding symmetry generators...")
            local_basis_generators = self.find_symmetric_generators(M, return_coeff=False, verbose=False)
            
            if local_basis_generators is None or len(local_basis_generators) == 0:
                if verbose:
                    print("⚠️  No symmetry generators found. Using full generator basis.")
                use_custom_basis = False
                basis_name = "site-local"
            else:
                use_custom_basis = True
                basis_name = "symmetry"
                
        else:
            # User-provided basis generators
            try:
                local_basis_generators = np.array(basis_generators)
                if local_basis_generators.ndim == 2:
                    # Single generator provided, wrap in list
                    local_basis_generators = [local_basis_generators]
                elif local_basis_generators.ndim == 3:
                    # Multiple generators provided
                    local_basis_generators = list(local_basis_generators)
                else:
                    raise ValueError("Invalid shape for basis_generators")
                    
                use_custom_basis = True
                basis_name = "custom"
                if verbose:
                    print(f"Using {len(local_basis_generators)} user-provided generators")
            except Exception as e:
                if verbose:
                    print(f"⚠️  Error processing basis_generators: {e}")
                    print("    Using full generator basis.")
                use_custom_basis = False
                basis_name = "site-local"

        if use_custom_basis and local_basis_generators is not None:
            # Build full generators in tensor product form
            full_custom_gens = []
            for gen in local_basis_generators:
                genfull = np.zeros((self.hilbert_dim, self.hilbert_dim), dtype=complex)
                for pos in range(self.D):
                    if pos == 0:
                        genterm = np.kron(gen, SUN.identity(self.N ** (self.D - 1)))
                    elif pos == self.D - 1:
                        genterm = np.kron(SUN.identity(self.N ** (self.D - 1)), gen)
                    else:
                        genterm = np.kron(np.kron(SUN.identity(self.N**pos), gen), 
                                        SUN.identity(self.N**(self.D - pos - 1)))
                    genfull += genterm
                full_custom_gens.append(genfull)
            
            # Solve for coefficients in custom basis
            H_flat = H.flatten()
            n_custom_gens = len(full_custom_gens)
            gen_matrix = np.zeros((self.hilbert_dim**2, n_custom_gens), dtype=complex)
            
            for i, gen in enumerate(full_custom_gens):
                gen_matrix[:, i] = gen.flatten()
            
            # Solve least squares
            coefficients_flat, residuals, rank, s = np.linalg.lstsq(gen_matrix, H_flat, rcond=None)
            coefficients = np.real(coefficients_flat)
            
            # If reference coefficients provided, compute scaling factor
            if reference_coeffs is not None:
                reference_coeffs = np.array(reference_coeffs).flatten()
                if len(reference_coeffs) == len(coefficients):
                    # Check if there's a global scaling relationship
                    # Avoid division by zero
                    nonzero_mask = np.abs(reference_coeffs) > 1e-10
                    if np.any(nonzero_mask):
                        ratios = coefficients[nonzero_mask] / reference_coeffs[nonzero_mask]
                        mean_ratio = np.mean(ratios)
                        std_ratio = np.std(ratios)
                        
                        if verbose:
                            print(f"  Reference comparison:")
                            print(f"    Reference: {purify(reference_coeffs)}")
                            print(f"    Recovered: {purify(coefficients)}")
                            print(f"    Ratio (recovered/reference): {purify(ratios)}")
                            print(f"    Mean ratio: {mean_ratio:.6f}, Std: {std_ratio:.6f}")
                            
                            if std_ratio < 0.01:  # Consistent scaling
                                print(f"  → Coefficients are consistently scaled by {mean_ratio:.6f}")
            
            # Check reconstruction quality
            H_reconstructed = np.zeros_like(H)
            for i, coeff in enumerate(coefficients):
                H_reconstructed += coeff * full_custom_gens[i]
            
            reconstruction_error = np.linalg.norm(H - H_reconstructed)
            
            if verbose:
                print(f"\nLie algebra decomposition ({basis_name.capitalize()} Basis):")
                print(f"  Number of basis generators: {n_custom_gens}")
                print(f"  Reconstruction error: {reconstruction_error:.2e}")
                print(f"  Coefficients in {basis_name} basis: {purify(coefficients)}")
            
            # Verify: exp(i * sum θ_i * G_i) ≈ M
            H_from_coeffs = np.zeros_like(H)
            for i, coeff in enumerate(coefficients):
                H_from_coeffs += coeff * full_custom_gens[i]
            
            M_reconstructed = expm(1j * H_from_coeffs)
            
            # Account for global phase
            phase_diff = np.angle(np.trace(M_reconstructed.conj().T @ M) / M.shape[0])
            M_reconstructed *= np.exp(-1j * phase_diff)
            
            verification_error = np.linalg.norm(M - M_reconstructed)
            
            if verbose:
                print(f"  Verification: ||M - exp(i∑θ_i G_i)|| = {verification_error:.2e}")
                
                if verification_error > 1e-6:
                    print("⚠️  Warning: Large verification error")
            
            return coefficients

        else:
            # Build generators for each site separately (full basis)
            site_generators = []

            for site in range(self.D):
                site_gens = []
                for gen in self.basis:
                    if site == 0:
                        genterm = np.kron(gen, SUN.identity(self.N ** (self.D - 1)))
                    elif site == self.D - 1:
                        genterm = np.kron(SUN.identity(self.N ** (self.D - 1)), gen)
                    else:
                        genterm = np.kron(np.kron(SUN.identity(self.N**site), gen), 
                                        SUN.identity(self.N**(self.D - site - 1)))
                    site_gens.append(genterm)
                site_generators.append(site_gens)

            # Flatten all generators into columns
            H_flat = H.flatten()
            total_generators = self.D * self.n_generators
            gen_matrix = np.zeros((self.hilbert_dim**2, total_generators), dtype=complex)

            for site in range(self.D):
                for gen_idx in range(self.n_generators):
                    col_idx = site * self.n_generators + gen_idx
                    gen_matrix[:, col_idx] = site_generators[site][gen_idx].flatten()

            # Solve least squares
            coefficients_flat, residuals, rank, s = np.linalg.lstsq(gen_matrix, H_flat, rcond=None)

            # Reshape coefficients into (D, n_generators)
            coefficients = coefficients_flat.reshape(self.D, self.n_generators)
            coefficients = np.real(coefficients)
            
            # If reference coefficients provided, compute scaling factor
            if reference_coeffs is not None:
                reference_coeffs = np.array(reference_coeffs)
                if reference_coeffs.shape == coefficients.shape:
                    # Flatten for comparison
                    ref_flat = reference_coeffs.flatten()
                    coeff_flat = coefficients.flatten()
                    nonzero_mask = np.abs(ref_flat) > 1e-10
                    
                    if np.any(nonzero_mask):
                        ratios = coeff_flat[nonzero_mask] / ref_flat[nonzero_mask]
                        mean_ratio = np.mean(ratios)
                        std_ratio = np.std(ratios)
                        
                        if verbose:
                            print(f"  Reference comparison:")
                            print(f"    Mean ratio: {mean_ratio:.6f}, Std: {std_ratio:.6f}")
                            if std_ratio < 0.01:
                                print(f"  → Coefficients are consistently scaled by {mean_ratio:.6f}")

            # Check reconstruction quality
            H_reconstructed = np.zeros_like(H)
            for site in range(self.D):
                for gen_idx in range(self.n_generators):
                    H_reconstructed += coefficients[site, gen_idx] * site_generators[site][gen_idx]

            reconstruction_error = np.linalg.norm(H - H_reconstructed)

            if verbose:
                print(f"\nLie algebra decomposition (local site basis):")
                print(f"  Reconstruction error: {reconstruction_error:.2e}")
                print(f"\n  Coefficients θ_a^{{(s)}} by site:")
                for site in range(self.D):
                    print(f"    Site {site}: {purify(coefficients[site])}")

            # Verify
            H_from_coeffs = np.zeros_like(H)
            for site in range(self.D):
                for gen_idx in range(self.n_generators):
                    H_from_coeffs += coefficients[site, gen_idx] * site_generators[site][gen_idx]

            M_reconstructed = expm(1j * H_from_coeffs)

            # Account for global phase
            phase_diff = np.angle(np.trace(M_reconstructed.conj().T @ M) / M.shape[0])
            M_reconstructed *= np.exp(-1j * phase_diff)

            verification_error = np.linalg.norm(M - M_reconstructed)
            
            if verbose:
                print(f"  Verification: ||M - exp(i∑_{{s,a}} θ_a^{{(s)}} T_a^{{(s)}})|| = {verification_error:.2e}")
                
                if verification_error > 1e-6:
                    print("⚠️  Warning: Large verification error - M may not be in su(N)^⊗D")

            return coefficients


class TensorProductGroupSO(TensorProductGroup):
    """Work with O^⊗D tensor product group for SO(N)"""

    def __init__(self, N, D):
        """
        Initialize for D-fold tensor products of SO(N).

        Args:
            N: Dimension of each local group
            D: Number of tensor product factors
        """
        self.N = N
        self.D = D
        self.local_dim = N
        self.hilbert_dim = N ** D
        self.basis = SON.so_basis(N)
        self.n_generators = len(self.basis)
        self.identity_local = SON.identity(N)

    def find_symmetric_generators(self, M, tol=1e-10, return_coeff=True, row_echelon=False, verbose=True):
        """
        Find SO(N) generators that commute with M.

        Args:
            M: Matrix to find commuting generators for
            tol: Tolerance for null space detection
            return_coeff: If True, return coefficients of generators.
                If False, return numeric generators.
            row_echelon: If True, print row echelon form of commutator matrix
            verbose: If True, print detailed output

        Returns:
            If return_coeff=True: array of coefficient vectors
            If return_coeff=False: array of local so(N) generators
        """
        comm = np.zeros((M.shape[0] ** 2, self.n_generators), dtype=complex)
        for i, gen in enumerate(self.basis):
            genfull = np.zeros((self.hilbert_dim, self.hilbert_dim), dtype=complex)
            for pos in range(self.D):
                if pos == 0:
                    genterm = np.kron(gen, np.eye(self.N ** (self.D - 1)))
                elif pos == self.D - 1:
                    genterm = np.kron(np.eye(self.N ** (self.D - 1)), gen)
                else:
                    genterm = np.kron(np.kron(np.eye(self.N ** pos), gen), np.eye(self.N ** (self.D - pos - 1)))
                genfull += genterm
            comm[:, i] = (genfull @ M - M @ genfull).flatten()

        if row_echelon:
            print("Row echelon form of comm matrix:")
            pprint(np.array(Matrix(comm.T).rref()[0]))

        null_vecs = null_space(comm)

        if verbose:
            print("\n" + "=" * 80)
            print("SYMMETRIC GENERATORS ANALYSIS (SO)")
            print("=" * 80)

        if null_vecs.shape[1] == 0:
            if verbose:
                print("\n⚠️  No symmetry found: The null space is empty.")
                print("=" * 80 + "\n")
            return np.array([])

        Q, _ = np.linalg.qr(np.real(null_vecs))
        null_vecs_ortho = Q

        if verbose:
            print(f"\nFound {null_vecs_ortho.shape[1]} generator(s) in the null space:")
            print("-" * 80)

        result = []
        valid_indices = []

        for i in range(null_vecs_ortho.shape[1]):
            coeff_vec = np.real(null_vecs_ortho[:, i])
            if np.linalg.norm(coeff_vec) > tol:
                valid_indices.append(i)
                if verbose:
                    print(f"\nGenerator {len(valid_indices)}:")
                    print(f"  Null space basis vector index: {i}")
                    print(f"  Coefficients in local so({self.N}) basis (orthonormalized): {purify(coeff_vec)}")

                    print(f"\n  Basis generators in so({self.N}) (with non-zero coefficients):")
                    for j in range(self.n_generators):
                        coeff = coeff_vec[j]
                        if np.abs(coeff) > tol:
                            print(f"    [{j}] (coeff: {purify(coeff)}):")
                            print(purify(self.basis[j]))

                if return_coeff:
                    result.append(coeff_vec)
                else:
                    gen_matrix_local = np.zeros((self.N, self.N), dtype=float)
                    for j in range(self.n_generators):
                        gen_matrix_local += coeff_vec[j] * self.basis[j]

                    gen_matrix_local = np.real((gen_matrix_local - gen_matrix_local.T) / 2)
                    result.append(gen_matrix_local)
                    if verbose:
                        print(f"\n  Local so({self.N}) generator (antisymmetric enforced):\n{purify(gen_matrix_local)}")

        if verbose:
            print("\n" + "-" * 80)
            print(f"\nOrthonormalized null space basis ({null_vecs_ortho.shape[1]} vectors):")
            for i in range(null_vecs_ortho.shape[1]):
                vec_norm = np.linalg.norm(null_vecs_ortho[:, i])
                print(f"\nBasis vector {i + 1} (norm = {vec_norm:.4f}):")
                print(purify(null_vecs_ortho[:, i]))

            print("\n" + "=" * 80 + "\n")

        if len(result) == 0:
            return np.array([])
        return np.array(result)

    def find_generators_coefficients(self, M, verbose=True, basis_generators=None, reference_coeffs=None):
        """
        Given a matrix M in SO(N)^⊗D, find coefficients θ_a such that:
        M ≈ exp(sum_a θ_a G_a)

        where G_a are real antisymmetric basis generators.

        Args:
            M: Matrix in SO(N)^⊗D (must be orthogonal with det ≈ +1)
            verbose: If True, print decomposition details
            basis_generators: Choice of basis generators:
                - None: full site-local basis
                - "symmetry": automatically find symmetry generators
                - List/array: user-provided local generators
            reference_coeffs: Optional reference coefficients for comparison output

        Returns:
            coefficients in chosen basis, or None if input checks fail
        """
        if not np.allclose(M @ M.T, np.eye(M.shape[0]), atol=1e-8):
            if verbose:
                print("⚠️  Warning: M is not orthogonal")
            return None

        det_M = np.linalg.det(M)
        if not np.allclose(det_M, 1.0, atol=1e-8):
            if verbose:
                print(f"⚠️  Warning: det(M) = {det_M:.6f} ≠ 1")
            return None

        log_M = logm(M)
        A = np.real(log_M)
        A = (A - A.T) / 2

        use_custom_basis = False
        local_basis_generators = None

        if basis_generators is None:
            use_custom_basis = False
            basis_name = "site-local"
        elif isinstance(basis_generators, str) and basis_generators.lower() == "symmetry":
            if verbose:
                print("Finding symmetry generators...")
            local_basis_generators = self.find_symmetric_generators(M, return_coeff=False, verbose=False)

            if local_basis_generators is None or len(local_basis_generators) == 0:
                if verbose:
                    print("⚠️  No symmetry generators found. Using full generator basis.")
                use_custom_basis = False
                basis_name = "site-local"
            else:
                use_custom_basis = True
                basis_name = "symmetry"
        else:
            try:
                local_basis_generators = np.array(basis_generators)
                if local_basis_generators.ndim == 2:
                    local_basis_generators = [local_basis_generators]
                elif local_basis_generators.ndim == 3:
                    local_basis_generators = list(local_basis_generators)
                else:
                    raise ValueError("Invalid shape for basis_generators")

                use_custom_basis = True
                basis_name = "custom"
                if verbose:
                    print(f"Using {len(local_basis_generators)} user-provided generators")
            except Exception as e:
                if verbose:
                    print(f"⚠️  Error processing basis_generators: {e}")
                    print("    Using full generator basis.")
                use_custom_basis = False
                basis_name = "site-local"

        if use_custom_basis and local_basis_generators is not None:
            full_custom_gens = []
            for gen in local_basis_generators:
                gen = np.real((gen - np.asarray(gen).T) / 2)
                genfull = np.zeros((self.hilbert_dim, self.hilbert_dim), dtype=float)
                for pos in range(self.D):
                    if pos == 0:
                        genterm = np.kron(gen, np.eye(self.N ** (self.D - 1)))
                    elif pos == self.D - 1:
                        genterm = np.kron(np.eye(self.N ** (self.D - 1)), gen)
                    else:
                        genterm = np.kron(np.kron(np.eye(self.N ** pos), gen), np.eye(self.N ** (self.D - pos - 1)))
                    genfull += genterm
                full_custom_gens.append(genfull)

            A_flat = A.flatten()
            n_custom_gens = len(full_custom_gens)
            gen_matrix = np.zeros((self.hilbert_dim ** 2, n_custom_gens), dtype=float)
            for i, gen in enumerate(full_custom_gens):
                gen_matrix[:, i] = np.real(gen.flatten())

            coefficients, residuals, rank, s = np.linalg.lstsq(gen_matrix, np.real(A_flat), rcond=None)

            if reference_coeffs is not None and verbose:
                reference_coeffs = np.array(reference_coeffs).flatten()
                if len(reference_coeffs) == len(coefficients):
                    nonzero_mask = np.abs(reference_coeffs) > 1e-10
                    if np.any(nonzero_mask):
                        ratios = coefficients[nonzero_mask] / reference_coeffs[nonzero_mask]
                        print("  Reference comparison:")
                        print(f"    Reference: {purify(reference_coeffs)}")
                        print(f"    Recovered: {purify(coefficients)}")
                        print(f"    Ratio (recovered/reference): {purify(ratios)}")
                        print(f"    Mean ratio: {np.mean(ratios):.6f}, Std: {np.std(ratios):.6f}")

            A_reconstructed = np.zeros_like(A)
            for i, coeff in enumerate(coefficients):
                A_reconstructed += coeff * full_custom_gens[i]
            A_reconstructed = (A_reconstructed - A_reconstructed.T) / 2

            reconstruction_error = np.linalg.norm(A - A_reconstructed)

            if verbose:
                print(f"\nLie algebra decomposition ({basis_name.capitalize()} Basis, SO):")
                print(f"  Number of basis generators: {n_custom_gens}")
                print(f"  Reconstruction error: {reconstruction_error:.2e}")
                print(f"  Coefficients in {basis_name} basis: {purify(coefficients)}")

            M_reconstructed = expm(A_reconstructed)
            verification_error = np.linalg.norm(M - M_reconstructed)

            if verbose:
                print(f"  Verification: ||M - exp(∑θ_i G_i)|| = {verification_error:.2e}")
                if verification_error > 1e-6:
                    print("⚠️  Warning: Large verification error")

            return np.real(coefficients)

        site_generators = []
        for site in range(self.D):
            site_gens = []
            for gen in self.basis:
                if site == 0:
                    genterm = np.kron(gen, np.eye(self.N ** (self.D - 1)))
                elif site == self.D - 1:
                    genterm = np.kron(np.eye(self.N ** (self.D - 1)), gen)
                else:
                    genterm = np.kron(np.kron(np.eye(self.N ** site), gen), np.eye(self.N ** (self.D - site - 1)))
                site_gens.append(genterm)
            site_generators.append(site_gens)

        A_flat = np.real(A.flatten())
        total_generators = self.D * self.n_generators
        gen_matrix = np.zeros((self.hilbert_dim ** 2, total_generators), dtype=float)

        for site in range(self.D):
            for gen_idx in range(self.n_generators):
                col_idx = site * self.n_generators + gen_idx
                gen_matrix[:, col_idx] = np.real(site_generators[site][gen_idx].flatten())

        coefficients_flat, residuals, rank, s = np.linalg.lstsq(gen_matrix, A_flat, rcond=None)
        coefficients = coefficients_flat.reshape(self.D, self.n_generators)

        if reference_coeffs is not None and verbose:
            reference_coeffs = np.array(reference_coeffs)
            if reference_coeffs.shape == coefficients.shape:
                ref_flat = reference_coeffs.flatten()
                coeff_flat = coefficients.flatten()
                nonzero_mask = np.abs(ref_flat) > 1e-10
                if np.any(nonzero_mask):
                    ratios = coeff_flat[nonzero_mask] / ref_flat[nonzero_mask]
                    print("  Reference comparison:")
                    print(f"    Mean ratio: {np.mean(ratios):.6f}, Std: {np.std(ratios):.6f}")

        A_reconstructed = np.zeros_like(A)
        for site in range(self.D):
            for gen_idx in range(self.n_generators):
                A_reconstructed += coefficients[site, gen_idx] * site_generators[site][gen_idx]
        A_reconstructed = (A_reconstructed - A_reconstructed.T) / 2

        reconstruction_error = np.linalg.norm(A - A_reconstructed)

        if verbose:
            print("\nLie algebra decomposition (local site basis, SO):")
            print(f"  Reconstruction error: {reconstruction_error:.2e}")
            print("\n  Coefficients θ_a^{(s)} by site:")
            for site in range(self.D):
                print(f"    Site {site}: {purify(coefficients[site])}")

        M_reconstructed = expm(A_reconstructed)
        verification_error = np.linalg.norm(M - M_reconstructed)

        if verbose:
            print(f"  Verification: ||M - exp(∑_{{s,a}} θ_a^{{(s)}} T_a^{{(s)}})|| = {verification_error:.2e}")
            if verification_error > 1e-6:
                print("⚠️  Warning: Large verification error - M may not be in so(N)^⊗D")

        return np.real(coefficients)

class TensorProductGroupPm:
    """Work with D^⊗D tensor product group for Dihedral symmetries"""

    def __init__(self, N, D):
        """
        Initialize for D-fold tensor products of Dihedral group D_N.

        Args:
            N: Number of elements in the dihedral group (D_N has N! elements)
            D: Number of tensor product factors
        """
        self.N = N
        self.D = D
        self.permutations = PermutationGroup.all_permutation_matrices(N)
        self.hilbert_dim = N ** D
    
    def perm_tensors(self):
        """Generate the tensor product representation of a permutation"""
        result = []
        for mat in self.permutations:
            perm_tensor = mat
            for i in range(self.D - 1):
                perm_tensor = np.kron(perm_tensor, mat)
            result.append(perm_tensor)
        return result
    
    def find_symmetry(self, M, verbose=True, return_indices=False):
        """Find permutation symmetries of M by checking commutation with perm_tensors
        If verbose, print which permutations commute with M and their indices."""
        perms = self.perm_tensors()
        symmetries = []
        for i, perm in enumerate(perms):
            if np.allclose(perm @ M, M @ perm, atol=1e-8):
                symmetries.append(i)
                if verbose:
                    print(f"Permutation index {i} commutes with M.")
        if verbose:
            print(f"Total symmetries found: {len(symmetries)}")
            print(f"Symmetry indices: {symmetries}")
        if return_indices:
            return symmetries
        else:
            return [self.permutations[i] for i in symmetries]
    
