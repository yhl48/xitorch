import torch
import lintorch as lt
from lintorch.tests.utils import compare_grad_with_fd, device_dtype_float_test, get_diagonally_dominant_class

@device_dtype_float_test()
def test_lsymeig(dtype, device):
    # generate the matrix
    def runtest(options):
        na = 10
        torch.manual_seed(123)
        A1 = (torch.rand((1,na,na))*0.1).to(dtype).to(device).requires_grad_(True)
        diag = (torch.arange(na, dtype=dtype)+1.0).to(device).unsqueeze(0).requires_grad_(True)
        Acls = get_diagonally_dominant_class(na)
        params = (A1, diag)

        A = Acls()
        neig = 4
        # evals: (nbatch, neig)
        # evecs: (nbatch, na, neig)
        evals, evecs = lt.lsymeig(A,
            neig=neig,
            params=params,
            fwd_options=options)

        # check with the eigendecomposition equation
        AU = A(evecs, *params)
        UE = evals.unsqueeze(1) * evecs
        assert torch.allclose(AU, UE, atol=1e-5, rtol=1e-5)

        # check orthogonality
        UTU = torch.bmm(evecs.transpose(-2,-1), evecs)
        eye = torch.eye(UTU.shape[-1]).unsqueeze(0).to(UTU.dtype).to(UTU.device)
        assert torch.allclose(UTU, eye, atol=1e-5, rtol=1e-5)

    all_options = [{
        "method": "davidson",
        "min_eps": 1e-9,
        },
        {
        "method": "exacteig",
        }]
    for options in all_options:
        runtest(options)

@device_dtype_float_test()
def test_lsymeig_with_M(dtype, device):
    # generate the matrix
    na = 10
    torch.manual_seed(123)
    A1 = (torch.rand((1,na,na))*0.1).to(dtype).to(device).requires_grad_(True)
    diag = (torch.arange(na, dtype=dtype)+1.0).to(device).unsqueeze(0).requires_grad_(True)
    Acls = get_diagonally_dominant_class(na)
    params = (A1, diag)
    M1 = (torch.rand((1,na,na))*0.01).to(dtype).to(device)
    mdiag = (torch.arange(na, dtype=dtype)+1.0).to(device).unsqueeze(0)
    Mcls = get_diagonally_dominant_class(na)
    mparams = (M1, mdiag)

    A = Acls()
    M = Mcls()
    neig = 4
    options = {
        "method": "exacteig",
        "min_eps": 1e-9,
    }
    # evals: (nbatch, neig)
    # evecs: (nbatch, na, neig)
    evals, evecs = lt.lsymeig(A,
        neig=neig,
        params=params,
        M=M,
        mparams=mparams,
        fwd_options=options)

    AU = A(evecs, *params)
    MUE = M(evals.unsqueeze(1) * evecs, *mparams)
    assert torch.allclose(AU, MUE, atol=1e-5, rtol=1e-5)

@device_dtype_float_test()
def test_solve(dtype, device):
    # generate the matrix
    na = 10
    ncols = 2
    torch.manual_seed(124)
    A1 = (torch.rand((1,na,na))*0.1).to(dtype).to(device)
    diag = (torch.arange(na, dtype=dtype)+1.0).to(device).unsqueeze(0)
    Acls = get_diagonally_dominant_class(na)
    M1 = (torch.rand((1,na,na))*0.1).to(dtype).to(device)
    mdiag = (torch.arange(na, dtype=dtype)+1.0).to(device).unsqueeze(0)
    Mcls = get_diagonally_dominant_class(na)
    xtrue = torch.rand(1,na,ncols).to(dtype).to(device)
    A = Acls()
    M = Mcls()
    biases = torch.rand(1,ncols).to(dtype).to(device)
    b = (A(xtrue, A1, diag) - biases.unsqueeze(1) * M(xtrue, M1, mdiag)).detach().requires_grad_()

    fwd_options = {
        "min_eps": 1e-9,
    }
    x = lt.solve(A, (A1, diag), b,
        biases = biases,
        M = M,
        mparams = (M1, mdiag),
        fwd_options = fwd_options)

    assert torch.allclose(x, xtrue, atol=1e-6, rtol=1e-6)
