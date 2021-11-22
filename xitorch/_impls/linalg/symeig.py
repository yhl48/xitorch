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
        # evals, evecs = torch.symeig(Amatrix, eigenvectors=True)  # (*BA, q), (*BA, q, q)
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
        # evals, evecs = torch.symeig(A2, eigenvectors=True)  # (*BAM, q, q)
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
        eigvalT, eigvecT = torch.symeig(T, eigenvectors=True)
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

# torch.set_default_dtype(torch.float64)
# def lobpcg(A: LinearOperator,  # B: Optional[LinearOperator],
#            neig: int,
#            mode: str,
#            M: Optional[LinearOperator] = None,
#            max_niter: int = 1000,
#            nguess: Optional[int] = None,
#            v_init: str = "randn",
#            max_addition: Optional[int] = None,
#            min_eps: float = 1e-6,
#            verbose: bool = False,
#            **unused) -> Tuple[torch.Tensor, torch.Tensor]:

#     K = A.mm(torch.eye(A.shape[-1]))
#     # L, U = torch.lu(K)
#     # print(L)
#     # pdb.set_trace()
#     K = torch.diag(torch.diag(K))
#     K_inv = torch.inverse(K)

#     B = None

#     if nguess is None:
#         nguess = neig

#     # get the shape of the transformation
#     na = A.shape[-1]
#     if M is None:
#         bcast_dims = A.shape[:-2]
#     else:
#         bcast_dims = get_bcasted_dims(A.shape[:-2], M.shape[:-2])
#     dtype = A.dtype
#     device = A.device

#     if B is None:
#         # create a linop of identity tensor with shape == A.shape if B is None
#         b = torch.eye(n=A.shape[-2], m=A.shape[-1], dtype=dtype, device=device)
#         if len(A.shape) > 2:
#             BA = A.shape[:-2]
#             b = b.repeat(BA, 1, 1)
#         B = LinearOperator.m(b)
    
#     assert A.shape == B.shape

#     # set up the initial guess
#     V = _set_initial_v(v_init.lower(), dtype, device,
#                        bcast_dims, na, nguess,
#                        M=M)  # (*BAM, na, nguess)
#     V = V / _safedenom(torch.max(V, dim=-2, keepdim=True)[0], 1e-12)

#     C, theta = rayleigh_ritz(V, A, B)
#     theta = torch.diag(theta)

#     V = torch.matmul(V, C)
#     V = V / _safedenom(torch.max(V, dim=-2, keepdim=True)[0], 1e-12)

#     R = A.mm(V) - torch.matmul(B.mm(V), theta)
#     R = R - torch.matmul(V, torch.matmul(V.T, R))
#     R = R / _safedenom(torch.max(R, dim=-2, keepdim=True)[0], 1e-12)

#     P: Optional[torch.Tensor] = None
#     for i in range(max_niter):
#         # print(torch.linalg.cond(R))
#         # R = torch.matmul(K_inv, R)
#         # print(torch.linalg.cond(R))
#         S = torch.cat((V, R, P), dim=-1) if P is not None else torch.cat((V, R), dim=-1)
#         # S = torch.matmul(K_inv, S)
#         # print(torch.linalg.norm(S, dim=-1, keepdim=True).shape)
#         # S = S / _safedenom(torch.linalg.norm(S, dim=-2, keepdim=True), 1e-12)
#         # S = S / _safedenom(torch.max(S, dim=-2, keepdim=True)[0], 1e-12)
#         # print(S)
#         # print(torch.linalg.eigh(torch.matmul(S.T, S))[0])
#         # print(torch.matmul(S.T, S))
#         # pdb.set_trace()
#         C, theta = rayleigh_ritz(S, A, B)
#         # print(C)
#         theta = torch.diag(theta[:nguess])

#         V = torch.matmul(S, C[..., :nguess])  # correct
#         V = V / _safedenom(torch.max(V, dim=-2, keepdim=True)[0], 1e-12)

#         R = A.mm(V) - torch.matmul(B.mm(V), theta)
#         R = R - torch.matmul(V, torch.matmul(V.T, R))
#         R = R / _safedenom(torch.max(R, dim=-2, keepdim=True)[0], 1e-12)

#         P = torch.matmul(S[..., nguess:], C[..., nguess:, :nguess])  # correct
#         P = P - torch.matmul(V, torch.matmul(V.T, P))
#         P = P - torch.matmul(R, torch.matmul(R.T, P))
#         # P = P / _safedenom(torch.max(P, dim=-2, keepdim=True)[0], 1e-12)
#         # print(C)
#         # print(P)
#         # print(P)
#         # pdb.set_trace()
#         # print(theta)
#     return torch.diag(theta), V

# def rayleigh_ritz(S: torch.Tensor,
#                   A: LinearOperator,
#                   B: LinearOperator) -> Tuple[torch.Tensor, torch.Tensor]:
#     StBS = torch.matmul(S.T, B.mm(S))  # this can be ill-conditioned
#     # StBS = torch.matmul(torch.pinverse(StBS), StBS)
#     # assert torch.all(torch.diag(StBS) > 0)
#     # assert torch.allclose(torch.matmul(S.T, B.mm(S)), torch.matmul(S.T, S))  # delete later
#     D = torch.inverse(torch.diag(torch.diag(StBS))) ** 0.5  # source of error
#     # D = torch.eye(D.shape[-1])
#     # D = torch.diag(torch.diag(StBS)) ** -0.5
#     inp = torch.matmul(D, torch.matmul(StBS, D))
#     Binp = inp.shape[:-2]
#     jitter = torch.eye(inp.shape[-2], inp.shape[-1], dtype=inp.dtype, device=inp.device).repeat(*Binp, 1, 1) * 1e-6
#     R = torch.linalg.cholesky(inp + 1 * jitter)  # lower triangular R
#     # print(R)
#     R_inv = torch.linalg.inv(R)
#     # print(R_inv)
#     StAS = torch.matmul(S.T, A.mm(S))
#     # StAS = torch.matmul(torch.pinverse(StAS), StAS)
    
#     theta, Z = torch.linalg.eigh(R_inv * D * StAS * D * R_inv.T)
#     # print(theta)
#     # print(Z)
#     theta, Z = _take_eigpairs(theta, Z, len(theta), 'uppest')
#     # print(theta)
#     # print(Z)
#     # print(theta)
#     # C = torch.matmul(R_inv.T, Z)
#     C = torch.matmul(D, torch.matmul(R_inv.T, Z))  # why is C always zero?
#     # C = (C + C.T) / 2
#     # C = torch.diag(torch.diag(C))
#     print(torch.linalg.cond(R))
#     print(torch.linalg.cond(S))
#     print(torch.linalg.cond(D))
#     print(torch.linalg.cond(StBS))
#     print(torch.linalg.cond(StAS))
#     # print(C)
#     # print(torch.matmul(C.T, torch.matmul(StBS, C)))
#     # print(torch.matmul(C.T, torch.matmul(StAS, C)))
#     pdb.set_trace()
#     return C.T, theta

def lobpcg(A: LinearOperator,  # B: Optional[LinearOperator],
           neig: int,
           mode: str,
           M: Optional[LinearOperator] = None,
           max_niter: int = 1000,
           nguess: Optional[int] = None,
           v_init: str = "randn",
           max_addition: Optional[int] = None,
           min_eps: float = 1e-6,
           verbose: bool = False,
           **unused) -> Tuple[torch.Tensor, torch.Tensor]:
    
    
    B = None

    if nguess is None:
        nguess = neig

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
                       bcast_dims, na, nguess,
                       M=M)  # (*BAM, na, nguess)

    blockVectorX = X
    blockVectorY = None

    if max_niter is None:
        max_niter = 20

    if blockVectorY is not None:
        sizeY = blockVectorY.shape[1]
    else:
        sizeY = 0

    n, sizeX = blockVectorX.shape

    # B-orthonormalize X
    blockVectorX, blockVectorBX = _b_orthonormalize(B, blockVectorX)

    # Compute the initial Ritz vectors: solve the eigenproblem.
    blockVectorAX = A.mm(blockVectorX)
    gramXAX = torch.matmul(blockVectorX.T.conj(), blockVectorAX)

    _lambda, eigBlockVector = torch.linalg.eigh(gramXAX)
    ii = _get_indx(_lambda, sizeX, largest=True)
    _lambda = _lambda[ii]

    eigBlockVector = torch.Tensor(eigBlockVector[:, ii])
    blockVectorX = torch.matmul(blockVectorX, eigBlockVector)
    blockVectorAX = torch.matmul(blockVectorAX, eigBlockVector)

    # Active index set
    activeMask = torch.ones((sizeX), dtype=bool)
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
   
    residualTolerance = torch.sqrt(torch.Tensor([1e-15])) * n
    while iterationNumber < max_niter:
        iterationNumber += 1

        assert B is None
        aux = blockVectorX * _lambda[None, :]

        blockVectorR = blockVectorAX - aux

        aux = torch.sum(blockVectorR.conj() * blockVectorR, dim=0)
        residualNorms = torch.sqrt(aux)

        ii = torch.where(residualNorms > residualTolerance, True, False)
        activeMask = activeMask & ii

        currentBlockSize = activeMask.sum()
        if currentBlockSize != previousBlockSize:
            previousBlockSize = currentBlockSize
            ident = torch.eye(currentBlockSize, dtype=A.dtype)
        
        if currentBlockSize == 0:
            break
        
        # activeBlockVectorR = blockVectorR[:, activeMask]
        
        activeBlockVectorR = _as2d(blockVectorR[:, activeMask])

        if iterationNumber > 0:
            activeBlockVectorP = _as2d(blockVectorP[:, activeMask])
            activeBlockVectorAP = _as2d(blockVectorAP[:, activeMask])
            # activeBlockVectorP = blockVectorP[:, activeMask]
            # activeBlockVectorAP = blockVectorAP[:, activeMask]
        #     if B is not None:
        #         activeBlockVectorBP = _as2d(blockVectorBP[:, activeMask])

        assert B is None
        activeBlockVectorR = activeBlockVectorR - torch.matmul(blockVectorX,
                                torch.matmul(blockVectorX.T.conj(),
                                activeBlockVectorR))

        # B-orthonormalize the preconditioned residuals
        aux = _b_orthonormalize(B, activeBlockVectorR)
        activeBlockVectorR, activeBlockVectorBR = aux

        activeBlockVectorAR = A.mm(activeBlockVectorR)

        if iterationNumber > 0:
            assert B is None
            aux = _b_orthonormalize(B, activeBlockVectorP, retInvR=True)
            activeBlockVectorP, _, invR, normal = aux
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
            gramXAX = (gramXAX + gramXAX.T.conj())/2
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
                gramPAP = (gramPAP + gramPAP.T.conj())/2
                gramPBP = torch.matmul(activeBlockVectorP.T.conj(),
                                 activeBlockVectorBP)
            else:
                gramPBP = ident

            gramXAX = coo_matrix(gramXAX.numpy())
            gramXAR = coo_matrix(gramXAR.numpy())
            gramXAP = coo_matrix(gramXAP.numpy())
            gramXBX = coo_matrix(gramXBX.numpy())
            gramRAP = coo_matrix(gramRAP.numpy())
            gramRAR = coo_matrix(gramRAR.numpy())
            gramPAP = coo_matrix(gramPAP.numpy())
            gramXBR = coo_matrix(gramXBR.numpy())
            gramRBR = coo_matrix(gramRBR.numpy())
            gramXBP = coo_matrix(gramXBP.numpy())
            gramRBP = coo_matrix(gramRBP.numpy())
            gramPBP = coo_matrix(gramPBP.numpy())

            # gramA = torch.Tensor([[gramXAX, gramXAR, gramXAP],
            #                       [gramXAR.T.conj(), gramRAR, gramRAP],
            #                       [gramXAP.T.conj(), gramRAP.T.conj(), gramPAP]])
            # gramB = torch.Tensor([[gramXBX, gramXBR, gramXBP],
            #                       [gramXBR.T.conj(), gramRBR, gramRBP],
            #                       [gramXBP.T.conj(), gramRBP.T.conj(), gramPBP]])

            gramA = bmat([[gramXAX, gramXAR, gramXAP],
                          [gramXAR.T.conj(), gramRAR, gramRAP],
                          [gramXAP.T.conj(), gramRAP.T.conj(), gramPAP]]).toarray()
            gramB = bmat([[gramXBX, gramXBR, gramXBP],
                          [gramXBR.T.conj(), gramRBR, gramRBP],
                          [gramXBP.T.conj(), gramRBP.T.conj(), gramPBP]]).toarray()
            # gramA = torch.Tensor(gramA)
            # gramB = torch.Tensor(gramB)
            try:
                # _lambda, eigBlockVector = torch.linalg.eigh(gramA)
                _lambda, eigBlockVector = scipy.linalg.eigh(gramB)
                _lambda, eigBlockVector = torch.Tensor(_lambda), torch.Tensor(eigBlockVector)
            except:
                # try again after dropping the direction vectors P from RR
                restart = True

        if restart:
            gramXAX = coo_matrix(gramXAX.numpy())
            gramXAR = coo_matrix(gramXAR.numpy())
            gramRAR = coo_matrix(gramRAR.numpy())
            gramXBX = coo_matrix(gramXBX.numpy())
            gramXBR = coo_matrix(gramXBR.numpy())
            gramRBR = coo_matrix(gramRBR.numpy())


            gramA = bmat([[gramXAX, gramXAR],
                          [gramXAR.T.conj(), gramRAR]]).toarray()
            gramB = bmat([[gramXBX, gramXBR],
                          [gramXBR.T.conj(), gramRBR]]).toarray()
            # gramA = torch.Tensor(gramA)
            # gramB = torch.Tensor(gramB)
            # gramA = torch.Tensor([[gramXAX, gramXAR],
            #                       [gramXAR.T.conj(), gramRAR]])
            # gramB = torch.Tensor([[gramXBX, gramXBR],
            #                       [gramXBR.T.conj(), gramRBR]])

            # _lambda, eigBlockVector = torch.linalg.eigh(gramA)
            try:
                _lambda, eigBlockVector = scipy.linalg.eigh(gramA, gramB)
                _lambda, eigBlockVector = torch.Tensor(_lambda), torch.Tensor(eigBlockVector)
            except:
                raise ValueError('eigh has failed in lobpcg iterations')
        ii = _get_indx(_lambda, sizeX, largest=True)
        _lambda = _lambda[ii]
        eigBlockVector = eigBlockVector[:, ii]

        # Compute Ritz vectors
        assert B is None
        if not restart:
            eigBlockVectorX = eigBlockVector[:sizeX]
            eigBlockVectorR = eigBlockVector[sizeX:sizeX+currentBlockSize]
            eigBlockVectorP = eigBlockVector[sizeX+currentBlockSize:]

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

    assert B is None
    aux = blockVectorX * _lambda[None, :]
    blockVectorR = blockVectorAX - aux

    aux = torch.sum(blockVectorR.conj() * blockVectorR, dim=0)
    residualNorms = torch.sqrt(aux)
    return _lambda, blockVectorX

def _as2d(ar: torch.Tensor):
    """
    If the input array is 2D return it, if it is 1D, append a dimension,
    making it a column vector.
    """
    if ar.ndim == 2:
        return ar
    else:  # Assume 1!
        # aux = torch.Tensor(ar, copy=False)
        # aux.shape = (ar.shape[0], 1)
        # return aux
        return ar[:, None]

def _get_indx(_lambda, num, largest):
    """Get `num` indices into `_lambda` depending on `largest` option."""
    ii = torch.argsort(_lambda)
    print(ii)
    pdb.set_trace()
    if largest:
        ii = ii[:-num-1:-1]
        # ii = ii[..., -num:]
    else:
        ii = ii[..., :num]

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
    except:
        #raise ValueError('Cholesky has failed')
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

############# helper functions #############
def _safedenom(r: torch.Tensor, eps: float) -> torch.Tensor:
    r[r == 0] = eps
    return r

def _dot(r: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    # r: (*BR, nr, nc)
    # z: (*BR, nr, nc)
    # return: (*BR, 1, nc)
    return torch.einsum("...rc,...rc->...c", r.conj(), z).unsqueeze(-2)

def _setup_precond(precond: Optional[LinearOperator]) -> Callable[[torch.Tensor], torch.Tensor]:
    if isinstance(precond, LinearOperator):
        precond_fcn = lambda x: precond.mm(x)
    elif precond is None:
        precond_fcn = lambda x: x
    else:
        raise TypeError("precond can only be LinearOperator or None")
    return precond_fcn