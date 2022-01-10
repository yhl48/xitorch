import warnings
import functools
import torch
from xitorch import LinearOperator
from typing import Union, Optional, Tuple, Sequence, Callable
from xitorch._utils.bcast import get_bcasted_dims
from xitorch._utils.tensor import tallqr, to_fortran_order
from xitorch.debug.modes import is_debug_enabled
from xitorch._utils.exceptions import MathWarning
from scipy.sparse import bmat, coo_matrix
import scipy
import numpy as np
import math
import pdb

def exacteig(A: LinearOperator, neig: int,
             mode: str, M: Optional[LinearOperator]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Eigendecomposition using explicit matrix construction.
    No additional option for this method.

    Warnings
    --------
    * As this method construct the linear operators explicitly, it might requires
      a large memory.
    """
    Amatrix = A.fullmatrix()  # (*BA, q, q)
    if M is None:
        # evals, evecs = torch.linalg.eigh(Amatrix, eigenvectors=True)  # (*BA, q), (*BA, q, q)
        evals, evecs = degen_symeig.apply(Amatrix)  # (*BA, q, q)
        return _take_eigpairs(evals, evecs, neig, mode)
    else:
        Mmatrix = M.fullmatrix()  # (*BM, q, q)

        # M decomposition to make A symmetric
        # it is done this way to make it numerically stable in avoiding
        # complex eigenvalues for (near-)degenerate case
        L = torch.linalg.cholesky(Mmatrix)  # (*BM, q, q)
        Linv = torch.inverse(L)  # (*BM, q, q)
        LinvT = Linv.transpose(-2, -1).conj()  # (*BM, q, q)
        A2 = torch.matmul(Linv, torch.matmul(Amatrix, LinvT))  # (*BAM, q, q)

        # calculate the eigenvalues and eigenvectors
        # (the eigvecs are normalized in M-space)
        # evals, evecs = torch.linalg.eigh(A2, eigenvectors=True)  # (*BAM, q, q)
        evals, evecs = degen_symeig.apply(A2)  # (*BAM, q, q)
        evals, evecs = _take_eigpairs(evals, evecs, neig, mode)  # (*BAM, neig) and (*BAM, q, neig)
        evecs = torch.matmul(LinvT, evecs)
        return evals, evecs

# temporary solution to https://github.com/pytorch/pytorch/issues/47599
class degen_symeig(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A):
        eival, eivec = torch.linalg.eigh(A)
        ctx.save_for_backward(eival, eivec)
        return eival, eivec

    @staticmethod
    def backward(ctx, grad_eival, grad_eivec):
        in_debug_mode = is_debug_enabled()

        eival, eivec = ctx.saved_tensors
        min_threshold = torch.finfo(eival.dtype).eps ** 0.6
        eivect = eivec.transpose(-2, -1).conj()

        # remove the degenerate part
        # see https://arxiv.org/pdf/2011.04366.pdf
        if grad_eivec is not None:
            # take the contribution from the eivec
            F = eival.unsqueeze(-2) - eival.unsqueeze(-1)
            idx = torch.abs(F) <= min_threshold
            F[idx] = float("inf")

            # if in debug mode, check the degeneracy requirements
            if in_debug_mode:
                degenerate = torch.any(idx)
                xtg = eivect @ grad_eivec
                diff_xtg = (xtg - xtg.transpose(-2, -1).conj())[idx]
                reqsat = torch.allclose(diff_xtg, torch.zeros_like(diff_xtg))
                # if the requirement is not satisfied, mathematically the derivative
                # should be `nan`, but here we just raise a warning
                if not reqsat:
                    msg = ("Degeneracy appears but the loss function seem to depend "
                           "strongly on the eigenvector. The gradient might be incorrect.\n")
                    msg += "Eigenvalues:\n%s\n" % str(eival)
                    msg += "Degenerate map:\n%s\n" % str(idx)
                    msg += "Requirements (should be all 0s):\n%s" % str(diff_xtg)
                    warnings.warn(MathWarning(msg))

            F = F.pow(-1)
            F = F * torch.matmul(eivect, grad_eivec)
            result = torch.matmul(eivec, torch.matmul(F, eivect))
        else:
            result = torch.zeros_like(eivec)

        # calculate the contribution from the eival
        if grad_eival is not None:
            result += torch.matmul(eivec, grad_eival.unsqueeze(-1) * eivect)

        # symmetrize to reduce numerical instability
        result = (result + result.transpose(-2, -1).conj()) * 0.5
        return result

def davidson(A: LinearOperator, neig: int,
             mode: str,
             M: Optional[LinearOperator] = None,
             max_niter: int = 1000,
             nguess: Optional[int] = None,
             v_init: str = "randn",
             max_addition: Optional[int] = None,
             min_eps: float = 1e-6,
             verbose: bool = False,
             **unused) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Using Davidson method for large sparse matrix eigendecomposition [2]_.

    Arguments
    ---------
    max_niter: int
        Maximum number of iterations
    v_init: str
        Mode of the initial guess (``"randn"``, ``"rand"``, ``"eye"``)
    max_addition: int or None
        Maximum number of new guesses to be added to the collected vectors.
        If None, set to ``neig``.
    min_eps: float
        Minimum residual error to be stopped
    verbose: bool
        Option to be verbose

    References
    ----------
    .. [2] P. Arbenz, "Lecture Notes on Solving Large Scale Eigenvalue Problems"
           http://people.inf.ethz.ch/arbenz/ewp/Lnotes/chapter12.pdf
    """
    # TODO: optimize for large linear operator and strict min_eps
    # Ideas:
    # (1) use better strategy to get the estimate on eigenvalues
    # (2) use restart strategy

    if nguess is None:
        nguess = neig
    if max_addition is None:
        max_addition = neig

    # get the shape of the transformation
    na = A.shape[-1]
    if M is None:
        bcast_dims = A.shape[:-2]
    else:
        bcast_dims = get_bcasted_dims(A.shape[:-2], M.shape[:-2])
    dtype = A.dtype
    device = A.device

    prev_eigvals = None
    prev_eigvalT = None
    stop_reason = "max_niter"
    shift_is_eigvalT = False
    idx = torch.arange(neig).unsqueeze(-1)  # (neig, 1)

    # set up the initial guess
    V = _set_initial_v(v_init.lower(), dtype, device,
                       bcast_dims, na, nguess,
                       M=M)  # (*BAM, na, nguess)

    best_resid: Union[float, torch.Tensor] = float("inf")
    AV = A.mm(V)
    for i in range(max_niter):
        VT = V.transpose(-2, -1)  # (*BAM,nguess,na)
        # Can be optimized by saving AV from the previous iteration and only
        # operate AV for the new V. This works because the old V has already
        # been orthogonalized, so it will stay the same
        # AV = A.mm(V) # (*BAM,na,nguess)
        T = torch.matmul(VT, AV)  # (*BAM,nguess,nguess)

        # eigvals are sorted from the lowest
        # eval: (*BAM, nguess), evec: (*BAM, nguess, nguess)
        eigvalT, eigvecT = torch.linalg.eigh(T)
        eigvalT, eigvecT = _take_eigpairs(eigvalT, eigvecT, neig, mode)  # (*BAM, neig) and (*BAM, nguess, neig)

        # calculate the eigenvectors of A
        eigvecA = torch.matmul(V, eigvecT)  # (*BAM, na, neig)

        # calculate the residual
        AVs = torch.matmul(AV, eigvecT)  # (*BAM, na, neig)
        LVs = eigvalT.unsqueeze(-2) * eigvecA  # (*BAM, na, neig)
        if M is not None:
            LVs = M.mm(LVs)
        resid = AVs - LVs  # (*BAM, na, neig)

        # print information and check convergence
        max_resid = resid.abs().max()
        if prev_eigvalT is not None:
            deigval = eigvalT - prev_eigvalT
            max_deigval = deigval.abs().max()
            if verbose:
                print("Iter %3d (guess size: %d): resid: %.3e, devals: %.3e" %
                      (i + 1, nguess, max_resid, max_deigval))  # type:ignore

        if max_resid < best_resid:
            best_resid = max_resid
            best_eigvals = eigvalT
            best_eigvecs = eigvecA
        if max_resid < min_eps:
            break
        if AV.shape[-1] == AV.shape[-2]:
            break
        prev_eigvalT = eigvalT

        # apply the preconditioner
        t = -resid  # (*BAM, na, neig)

        # orthogonalize t with the rest of the V
        t = to_fortran_order(t)
        Vnew = torch.cat((V, t), dim=-1)
        if Vnew.shape[-1] > Vnew.shape[-2]:
            Vnew = Vnew[..., :Vnew.shape[-2]]
        nadd = Vnew.shape[-1] - V.shape[-1]
        nguess = nguess + nadd
        if M is not None:
            MV_ = M.mm(Vnew)
            V, R = tallqr(Vnew, MV=MV_)
        else:
            V, R = tallqr(Vnew)
        AVnew = A.mm(V[..., -nadd:])  # (*BAM,na,nadd)
        AVnew = to_fortran_order(AVnew)
        AV = torch.cat((AV, AVnew), dim=-1)

    eigvals = best_eigvals  # (*BAM, neig)
    eigvecs = best_eigvecs  # (*BAM, na, neig)
    return eigvals, eigvecs

def lobpcg(A: LinearOperator,
           neig: int,
           mode: str,
           M: Optional[LinearOperator] = None,
           max_niter: int = 1000,
           v_init: str = "randn",
           min_eps: float = 1e-6,
           verbose: bool = False,
           B: Optional[LinearOperator] = None,
           **unused) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Locally Optimal Block Preconditioned Conjugate Gradient (LOBPCG)
    method to find largest or smallest eigenvalues and the corresponding
    eigenvectors of a symmetric generalized eigenvalue problem Ax=\u03BBBx [3]_.

    Arguments
    ---------
    max_niter: int
        Maximum number of iterations
    v_init: str
        Mode of the initial guess (``"randn"``, ``"rand"``, ``"eye"``)
    B: Linear Operator or None
       The right hand side operator in a generalized eigenproblem.
       Should be positive-definite and either complex Hermitian or real symmetric.
    min_eps: float
       Stopping criterion.
    verbose: bool
       Not implemented for lobpcg for now. Once implemented, could be set to true to be verbose.

    References
    ----------
    .. [3] https://en.wikipedia.org/wiki/LOBPCG
    """

    # get the shape of the transformation
    na = A.shape[-1]
    if M is None:
        bcast_dims = A.shape[:-2]
    else:
        bcast_dims = get_bcasted_dims(A.shape[:-2], M.shape[:-2])
    dtype = A.dtype
    device = A.device

    # set up the initial guess
    X = _set_initial_v(v_init.lower(), dtype, device,
                       bcast_dims, na, neig,
                       M=M)  # (*BAM, na, neig)

    blockVectorX = X
    blockVectorY = None

    if max_niter is None:
        max_niter = 20

    if blockVectorY is not None:
        sizeY = blockVectorY.shape[1]
    else:
        sizeY = 0

    if len(blockVectorX.shape) == 2:
        n, sizeX = blockVectorX.shape
    else:
        raise NotImplementedError("Currently only support A of a single batch size")

    if min_eps is None or min_eps < 0:
        min_eps = torch.sqrt(torch.Tensor([1e-15])) * n

    # B-orthonormalize X
    blockVectorX, blockVectorBX = _b_orthonormalize(B, blockVectorX)

    # Compute the initial Ritz vectors: solve the eigenproblem.
    blockVectorAX = A.mm(blockVectorX)
    gramXAX = torch.matmul(blockVectorX.T.conj(), blockVectorAX)

    _lambda, eigBlockVector = torch.linalg.eigh(gramXAX)
    ii = _get_indx(_lambda, sizeX, mode=mode)
    _lambda = _lambda[ii]

    eigBlockVector = torch.Tensor(eigBlockVector[:, ii])
    blockVectorX = torch.matmul(blockVectorX, eigBlockVector)
    blockVectorAX = torch.matmul(blockVectorAX, eigBlockVector)
    if B is not None:
        blockVectorBX = torch.matmul(blockVectorBX, eigBlockVector)

    # Active index set
    activeMask = torch.ones((sizeX,), dtype=bool)
    previousBlockSize = sizeX
    ident = torch.eye(sizeX, dtype=A.dtype)
    ident0 = torch.eye(sizeX, dtype=A.dtype)

    # Main iteration loop.
    blockVectorP = None  # set during iteration
    blockVectorAP = None
    blockVectorBP = None

    iterationNumber = -1
    restart = True
    explicitGramFlag = False

    # for iterationNumber in range(max_niter):
    while iterationNumber < max_niter:
        # print(iterationNumber)
        iterationNumber += 1

        # assert B is None
        if B is None:
            aux = blockVectorX * _lambda[None, :]
        else:
            aux = blockVectorBX * _lambda[None, :]

        blockVectorR = blockVectorAX - aux

        aux = torch.sum(blockVectorR.conj() * blockVectorR, dim=0)
        residualNorms = torch.sqrt(aux)

        ii = torch.where(residualNorms > min_eps, True, False)
        activeMask = activeMask & ii

        currentBlockSize = activeMask.sum()
        if currentBlockSize != previousBlockSize:
            previousBlockSize = currentBlockSize
            ident = torch.eye(currentBlockSize, dtype=A.dtype)

        if currentBlockSize == 0:
            break

        activeBlockVectorR = _as2d(blockVectorR[:, activeMask])

        if iterationNumber > 0:
            activeBlockVectorP = _as2d(blockVectorP[:, activeMask])
            activeBlockVectorAP = _as2d(blockVectorAP[:, activeMask])
            if B is not None:
                activeBlockVectorBP = _as2d(blockVectorBP[:, activeMask])
            # assert B is None

        # assert B is None
        if B is None:
            activeBlockVectorR = activeBlockVectorR - torch.matmul(blockVectorX,
                                                                   torch.matmul(blockVectorX.T.conj(),
                                                                                activeBlockVectorR))
        else:
            activeBlockVectorR = activeBlockVectorR - torch.matmul(blockVectorX,
                                                                   torch.matmul(blockVectorBX.T.conj(),
                                                                                activeBlockVectorR))

        # B-orthonormalize the preconditioned residuals
        aux = _b_orthonormalize(B, activeBlockVectorR)
        activeBlockVectorR, activeBlockVectorBR = aux

        activeBlockVectorAR = A.mm(activeBlockVectorR)

        if iterationNumber > 0:
            # assert B is None
            if B is None:
                aux = _b_orthonormalize(B, activeBlockVectorP, retInvR=True)
                activeBlockVectorP, _, invR, normal = aux
            else:
                aux = _b_orthonormalize(B, activeBlockVectorP, activeBlockVectorBP, retInvR=True)
                activeBlockVectorP, activeBlockVectorBP, invR, normal = aux

            # Function _b_orthonormalize returns None if Cholesky fails
            if activeBlockVectorP is not None:
                activeBlockVectorAP = activeBlockVectorAP / _safedenom(normal, 1e-12)
                activeBlockVectorAP = torch.matmul(activeBlockVectorAP, invR)
                restart = False
            else:
                restart = True

        # Perform the Rayleigh Ritz Procedure:
        # Compute symmetric Gram matrices:
        if activeBlockVectorAR.dtype == torch.float64:
            myeps = 1
        elif activeBlockVectorR.dtype == torch.float32:
            myeps = 1e-4
        else:
            myeps = 1e-8

        if residualNorms.max() > myeps and not explicitGramFlag:
            explicitGramFlag = False
        else:
            # Once explicitGramFlag, forever explicitGramFlag.
            explicitGramFlag = True

        # Shared memory assingments to simplify the code
        # assert B is None
        if B is None:
            blockVectorBX = blockVectorX
            activeBlockVectorBR = activeBlockVectorR
            if not restart:
                activeBlockVectorBP = activeBlockVectorP

        # Common submatrices:
        gramXAR = torch.matmul(blockVectorX.T.conj(), activeBlockVectorAR)
        gramRAR = torch.matmul(activeBlockVectorR.T.conj(), activeBlockVectorAR)

        if explicitGramFlag:
            gramRAR = (gramRAR + gramRAR.T.conj()) / 2
            gramXAX = torch.matmul(blockVectorX.T.conj(), blockVectorAX)
            gramXAX = (gramXAX + gramXAX.T.conj()) / 2
            gramXBX = torch.matmul(blockVectorX.T.conj(), blockVectorBX)
            gramRBR = torch.matmul(activeBlockVectorR.T.conj(), activeBlockVectorBR)
            gramXBR = torch.matmul(blockVectorX.T.conj(), activeBlockVectorBR)
        else:
            gramXAX = torch.diag(_lambda)
            gramXBX = ident0
            gramRBR = ident
            gramXBR = torch.zeros((sizeX, currentBlockSize), dtype=A.dtype)

        if not restart:
            gramXAP = torch.matmul(blockVectorX.T.conj(), activeBlockVectorAP)
            gramRAP = torch.matmul(activeBlockVectorR.T.conj(), activeBlockVectorAP)
            gramPAP = torch.matmul(activeBlockVectorP.T.conj(), activeBlockVectorAP)
            gramXBP = torch.matmul(blockVectorX.T.conj(), activeBlockVectorBP)
            gramRBP = torch.matmul(activeBlockVectorR.T.conj(), activeBlockVectorBP)
            if explicitGramFlag:
                gramPAP = (gramPAP + gramPAP.T.conj()) / 2
                gramPBP = torch.matmul(activeBlockVectorP.T.conj(), activeBlockVectorBP)
            else:
                gramPBP = ident

            gramA = torch.cat((torch.cat((gramXAX, gramXAR, gramXAP), dim=-1),
                               torch.cat((gramXAR.T.conj(), gramRAR, gramRAP), dim=-1),
                               torch.cat((gramXAP.T.conj(), gramRAP.T.conj(), gramPAP), dim=-1)), dim=-2)

            gramB = torch.cat((torch.cat((gramXBX, gramXBR, gramXBP), dim=-1),
                               torch.cat((gramXBR.T.conj(), gramRBR, gramRBP), dim=-1),
                               torch.cat((gramXBP.T.conj(), gramRBP.T.conj(), gramPBP), dim=-1)), dim=-2)

            try:
                _lambda, eigBlockVector = _eigh(gramA, gramB)
            except Exception as e:
                # try again after dropping the direction vectors P from RR
                restart = True

        if restart:
            gramA = torch.cat((torch.cat((gramXAX, gramXAR), dim=-1),
                               torch.cat((gramXAR.T.conj(), gramRAR), dim=-1)), dim=-2)

            gramB = torch.cat((torch.cat((gramXBX, gramXBR), dim=-1),
                               torch.cat((gramXBR.T.conj(), gramRBR), dim=-1)), dim=-2)
            try:
                _lambda, eigBlockVector = _eigh(gramA, gramB)
            except Exception as e:
                raise ValueError('eigh has failed in lobpcg iterations')

        ii = _get_indx(_lambda, sizeX, mode=mode)
        _lambda = _lambda[ii]
        eigBlockVector = eigBlockVector[:, ii]

        # Compute Ritz vectors
        # assert B is None
        if B is None:
            if not restart:
                eigBlockVectorX = eigBlockVector[:sizeX]
                eigBlockVectorR = eigBlockVector[sizeX:sizeX + currentBlockSize]
                eigBlockVectorP = eigBlockVector[sizeX + currentBlockSize:]

                pp = torch.matmul(activeBlockVectorR, eigBlockVectorR)
                pp += torch.matmul(activeBlockVectorP, eigBlockVectorP)

                app = torch.matmul(activeBlockVectorAR, eigBlockVectorR)
                app += torch.matmul(activeBlockVectorAP, eigBlockVectorP)
            else:
                eigBlockVectorX = eigBlockVector[:sizeX]
                eigBlockVectorR = eigBlockVector[sizeX:]

                pp = torch.matmul(activeBlockVectorR, eigBlockVectorR)
                app = torch.matmul(activeBlockVectorAR, eigBlockVectorR)

            blockVectorX = torch.matmul(blockVectorX, eigBlockVectorX) + pp
            blockVectorAX = torch.matmul(blockVectorAX, eigBlockVectorX) + app

            blockVectorP, blockVectorAP = pp, app

        else:
            if not restart:
                eigBlockVectorX = eigBlockVector[:sizeX]
                eigBlockVectorR = eigBlockVector[sizeX:sizeX + currentBlockSize]
                eigBlockVectorP = eigBlockVector[sizeX + currentBlockSize:]

                pp = torch.matmul(activeBlockVectorR, eigBlockVectorR)
                pp += torch.matmul(activeBlockVectorP, eigBlockVectorP)

                app = torch.matmul(activeBlockVectorAR, eigBlockVectorR)
                app += torch.matmul(activeBlockVectorAP, eigBlockVectorP)

                bpp = torch.matmul(activeBlockVectorBR, eigBlockVectorR)
                bpp += torch.matmul(activeBlockVectorBP, eigBlockVectorP)
            else:
                eigBlockVectorX = eigBlockVector[:sizeX]
                eigBlockVectorR = eigBlockVector[sizeX:]

                pp = torch.matmul(activeBlockVectorR, eigBlockVectorR)
                app = torch.matmul(activeBlockVectorAR, eigBlockVectorR)
                bpp = torch.matmul(activeBlockVectorBR, eigBlockVectorR)

            blockVectorX = torch.matmul(blockVectorX, eigBlockVectorX) + pp
            blockVectorAX = torch.matmul(blockVectorAX, eigBlockVectorX) + app
            blockVectorBX = torch.matmul(blockVectorBX, eigBlockVectorX) + bpp

            blockVectorP, blockVectorAP, blockVectorBP = pp, app, bpp

    return _lambda, blockVectorX

def _as2d(ar: torch.Tensor):
    """
    If the input array is 2D return it, if it is 1D, append a dimension,
    making it a column vector.
    """
    if ar.ndim == 2:
        return ar
    else:  # Assume 1!
        return ar[:, None]

def _get_indx(_lambda, num, mode):
    """Get `num` indices into `_lambda` depending on `largest` option."""
    ii = torch.argsort(_lambda)
    if mode == 'lowest':
        ii = ii[..., :num]
    else:
        ii = ii[..., -num:]

    return ii

def _b_orthonormalize(B: Optional[LinearOperator],
                      blockVectorV: torch.Tensor,
                      blockVectorBV: Optional[torch.Tensor] = None,
                      retInvR: bool = False):
    """B-orthonormalize the given block vector using Cholesky."""
    normalization = torch.max(blockVectorV, axis=-2, keepdim=True)[0]
    blockVectorV = blockVectorV / _safedenom(normalization, 1e-12)
    if blockVectorBV is None:
        if B is None:
            blockVectorBV = blockVectorV  # Shared data!!!
        else:
            blockVectorBV = B.mm(blockVectorV)
    else:
        blockVectorBV = blockVectorBV / _safedenom(normalization, 1e-12)
    VBV = torch.matmul(blockVectorV.T.conj(), blockVectorBV)
    try:
        # VBV is a Cholesky factor from now on...
        VBV = torch.cholesky(VBV, upper=True)
        VBV = torch.linalg.inv(VBV)
        blockVectorV = torch.matmul(blockVectorV, VBV)
        # blockVectorV = (cho_solve((VBV.T, True), blockVectorV.T)).T
        if B is not None:
            blockVectorBV = torch.matmul(blockVectorBV, VBV)
            # blockVectorBV = (cho_solve((VBV.T, True), blockVectorBV.T)).T
        else:
            blockVectorBV = None
    except TypeError:
        # raise ValueError('Cholesky has failed')
        blockVectorV = None
        blockVectorBV = None
        VBV = None

    if retInvR:
        return blockVectorV, blockVectorBV, VBV, normalization
    else:
        return blockVectorV, blockVectorBV

def _set_initial_v(vinit_type: str,
                   dtype: torch.dtype, device: torch.device,
                   batch_dims: Sequence,
                   na: int,
                   nguess: int,
                   M: Optional[LinearOperator] = None) -> torch.Tensor:

    torch.manual_seed(12421)
    if vinit_type == "eye":
        nbatch = functools.reduce(lambda x, y: x * y, batch_dims, 1)
        V = torch.eye(na, nguess, dtype=dtype, device=device).unsqueeze(
            0).repeat(nbatch, 1, 1).reshape(*batch_dims, na, nguess)
    elif vinit_type == "randn":
        V = torch.randn((*batch_dims, na, nguess), dtype=dtype, device=device)
    elif vinit_type == "random" or vinit_type == "rand":
        V = torch.rand((*batch_dims, na, nguess), dtype=dtype, device=device)
    else:
        raise ValueError("Unknown v_init type: %s" % vinit_type)

    # orthogonalize V
    if isinstance(M, LinearOperator):
        V, R = tallqr(V, MV=M.mm(V))
    else:
        V, R = tallqr(V)
    return V

def _take_eigpairs(eival, eivec, neig, mode):
    # eival: (*BV, na)
    # eivec: (*BV, na, na)
    if mode == "lowest":
        eival = eival[..., :neig]
        eivec = eivec[..., :neig]
    else:  # uppest
        eival = eival[..., -neig:]
        eivec = eivec[..., -neig:]
    return eival, eivec

def _safedenom(r: torch.Tensor, eps: float) -> torch.Tensor:
    r[r == 0] = eps
    return r

def _eigh(A: torch.Tensor, B: Optional[torch.Tensor] = None):
    """
    Helper function for converting a generalized eigenvalue problem
    A(X) = lambda(B(X)) to standard eigen value problem using cholesky
    transformation
    """
    if B is None:  # use cupy's eigh in standard case
        vals, vecs = torch.linalg.eigh(A)
        return vals, vecs
    else:
        R = _cholesky(B)
        RTi = torch.linalg.inv(R)
        Ri = torch.linalg.inv(R.T)
        F = torch.matmul(RTi, torch.matmul(A, Ri))
        vals, vecs = torch.linalg.eigh(F)
        eigVec = torch.matmul(Ri, vecs)
        return vals, eigVec

def _cholesky(B):
    """
    Wrapper around `cupy.linalg.cholesky` that raises LinAlgError if there are
    NaNs in the output
    """
    R = torch.linalg.cholesky(B)
    # if torch.any(torch.isnan(R)):
    #     raise RuntimeError()
    return R
