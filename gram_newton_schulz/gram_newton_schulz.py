from types import SimpleNamespace
from typing import Dict, List, Optional
import torch
from torch import Tensor
from .coefficients import POLAR_EXPRESS_COEFFICIENTS

SYMMETRIC_KERNEL_TILE_SIZE = 256


_TORCH_BACKEND = SimpleNamespace(
    sym_mm=lambda A, B: A @ B,
    sym_baddbmm=lambda A, B, C, alpha=1., beta=1.: torch.baddbmm(C, A, B, alpha=alpha, beta=beta),
    mm=lambda A, B: A @ B,
    mm_add=lambda A, B, C, beta: torch.baddbmm(C, A, B, beta=beta),
)


def _make_kernel_backend():
    from quack.gemm_interface import gemm_symmetric, gemm, gemm_add
    return SimpleNamespace(
        sym_mm=gemm_symmetric,
        sym_baddbmm=lambda A, B, C, alpha=1., beta=1.: gemm_symmetric(A, B, C=C, alpha=alpha, beta=beta),
        mm=lambda A, B: gemm(A, B, tuned=False),
        mm_add=lambda A, B, C, beta: gemm_add(A, B, C=C, beta=beta, tuned=False),
    )


# ==========================================
# [ADDED]: Helper function to simulate INT8 quantization truncation error.
# Designed to be fully traceable by torch.compile without causing graph breaks.
# ==========================================
def _simulate_quantize(tensor: Tensor, simulate_int8: bool) -> Tensor:
    if not simulate_int8:
        return tensor
    scale = tensor.abs().max() / 127.0
    scale = torch.clamp(scale, min=1e-8)  # Prevent division by zero
    return torch.round(tensor / scale) * scale


# [MODIFIED]: Added 'simulate_int8=False' parameter to the function signature
def _make_compiled_gram(ops, ns_coefficients, gram_newton_schulz_reset_iterations, ns_epsilon, compile_kwargs, simulate_int8=False):
    """Build a compiled closure for gram Newton-Schulz with a fixed backend."""
    ns_coefficients = list(ns_coefficients)
    gram_newton_schulz_reset_iterations = set(gram_newton_schulz_reset_iterations)

    def _gram_newton_schulz(X: Tensor) -> Tensor:
        tall_skinny = X.size(-2) > X.size(-1)
        X = X.to(torch.float32)
        X = X / (X.norm(dim=(-2, -1), keepdim=True) + ns_epsilon)
        X = X.to(torch.float16)

        if tall_skinny:
            R = ops.sym_mm(X.mT, X)
        else:
            R = ops.sym_mm(X, X.mT)
            
        # [ADDED]: Simulate quantization truncation on the initial Gram matrix R
        R = _simulate_quantize(R, simulate_int8)

        batch_size = R.size(0)
        I = torch.eye(R.size(-1), device=X.device, dtype=X.dtype).unsqueeze(0).expand(batch_size, -1, -1).contiguous()
        Q = None

        for i, (a, b, c) in enumerate(ns_coefficients):
            if i in gram_newton_schulz_reset_iterations and i != 0:
                if tall_skinny:
                    X = ops.mm(X, Q)
                    R = ops.sym_mm(X.mT, X)
                else:
                    X = ops.mm(Q, X)
                    R = ops.sym_mm(X, X.mT)
                    
                # [ADDED]: Simulate quantization truncation on R after a Restart
                R = _simulate_quantize(R, simulate_int8)
                Q = None

            Z = ops.sym_baddbmm(R, R, C=R, alpha=c, beta=b)
            # [ADDED]: Simulate quantization truncation on the polynomial multiplier Z
            Z = _simulate_quantize(Z, simulate_int8)

            if i == 0 or i in gram_newton_schulz_reset_iterations:
                Q = Z + a * I
            else:
                Q = ops.sym_baddbmm(Q, Z, C=Q, beta=a)
            # [ADDED]: Simulate quantization truncation on the accumulated multiplier Q
            Q = _simulate_quantize(Q, simulate_int8)

            if i < len(ns_coefficients) - 1 and i + 1 not in gram_newton_schulz_reset_iterations:
                RZ = ops.sym_baddbmm(R, Z, C=R, beta=a)
                # [ADDED]: Simulate quantization truncation on the intermediate matrix RZ
                RZ = _simulate_quantize(RZ, simulate_int8)
                
                R = ops.sym_baddbmm(Z, RZ, C=RZ, beta=a)
                # [ADDED]: Simulate quantization truncation on the updated Gram matrix R
                R = _simulate_quantize(R, simulate_int8)

        if tall_skinny:
            X = ops.mm(X, Q)
        else:
            X = ops.mm(Q, X)
        return X

    if compile_kwargs is not None:
        _gram_newton_schulz = torch.compile(_gram_newton_schulz, **compile_kwargs)
    return _gram_newton_schulz


def _make_compiled_standard(ops, ns_coefficients, ns_epsilon, compile_kwargs):
    """Build a compiled closure for standard Newton-Schulz with a fixed backend."""
    ns_coefficients = list(ns_coefficients)

    def _standard_newton_schulz(X: Tensor) -> Tensor:
        tall_skinny = X.size(-2) > X.size(-1)
        X = X.to(torch.float32)
        X = X / (X.norm(dim=(-2, -1), keepdim=True) + ns_epsilon)
        X = X.to(torch.float16)

        for a, b, c in ns_coefficients:
            if tall_skinny:
                A = ops.sym_mm(X.mT, X)
            else:
                A = ops.sym_mm(X, X.mT)
            B = ops.sym_baddbmm(A, A, C=A, alpha=c, beta=b)
            if tall_skinny:
                X = ops.mm_add(X, B, C=X, beta=a)
            else:
                X = ops.mm_add(B, X, C=X, beta=a)
        return X

    if compile_kwargs is not None:
        _standard_newton_schulz = torch.compile(_standard_newton_schulz, **compile_kwargs)
    return _standard_newton_schulz


class GramNewtonSchulz:
    """
    Gram Newton-Schulz orthogonalization.
    """
    def __init__(
        self,
        ns_epsilon: float = 1e-7,
        ns_use_kernels: bool = True,
        ns_coefficients: Optional[List[List[float]]] = None,
        use_gram_newton_schulz: bool = True,
        gram_newton_schulz_reset_iterations: List[int] = None,
        compile_kwargs: Optional[Dict] = {"fullgraph": True, "mode": "reduce-overhead"},
        simulate_int8: bool = False,  # [ADDED]: Exposed INT8 toggle in the class constructor
    ):
        self.ns_epsilon = ns_epsilon
        self.ns_use_kernels = ns_use_kernels
        self.ns_coefficients = ns_coefficients if ns_coefficients is not None else POLAR_EXPRESS_COEFFICIENTS
        self.use_gram_newton_schulz = use_gram_newton_schulz
        
        # [ADDED]: Save the toggle as an instance attribute
        self.simulate_int8 = simulate_int8  

        if use_gram_newton_schulz:
            self.gram_newton_schulz_reset_iterations = gram_newton_schulz_reset_iterations if gram_newton_schulz_reset_iterations is not None else [2]

        kernel_backend = _make_kernel_backend() if self.ns_use_kernels else None

        if use_gram_newton_schulz:
            # [MODIFIED]: Passed self.simulate_int8 to _make_compiled_gram for the Torch backend
            self._gram_torch = _make_compiled_gram(
                _TORCH_BACKEND, self.ns_coefficients, self.gram_newton_schulz_reset_iterations, ns_epsilon, compile_kwargs, self.simulate_int8)
            
            if kernel_backend is not None:
                # [MODIFIED]: Passed self.simulate_int8 to _make_compiled_gram for the Custom Kernel backend
                self._gram_kernel = _make_compiled_gram(
                    kernel_backend, self.ns_coefficients, self.gram_newton_schulz_reset_iterations, ns_epsilon, compile_kwargs, self.simulate_int8)

        self._standard_torch = _make_compiled_standard(
            _TORCH_BACKEND, self.ns_coefficients, ns_epsilon, compile_kwargs)
        if kernel_backend is not None:
            self._standard_kernel = _make_compiled_standard(
                kernel_backend, self.ns_coefficients, ns_epsilon, compile_kwargs)

        self._kernel_backend = kernel_backend

    def _select_gram(self, X: Tensor) -> Tensor:
        if self._kernel_backend is not None and min(X.size(-2), X.size(-1)) >= SYMMETRIC_KERNEL_TILE_SIZE:
            return self._gram_kernel(X)
        return self._gram_torch(X)

    def _select_standard(self, X: Tensor) -> Tensor:
        if self._kernel_backend is not None and min(X.size(-2), X.size(-1)) >= SYMMETRIC_KERNEL_TILE_SIZE:
            return self._standard_kernel(X)
        return self._standard_torch(X)

    def __call__(self, X: Tensor) -> Tensor:
        original_shape = X.shape
        if X.ndim == 2:
            X = X.unsqueeze(0)
        elif X.ndim > 3:
            X = X.view(-1, *X.shape[-2:])

        original_dtype = X.dtype

        if self.use_gram_newton_schulz and max(X.shape[-2:]) > min(X.shape[-2:]):
            X = self._select_gram(X)
        else:
            X = self._select_standard(X)

        return X.to(original_dtype).view(original_shape)


class StandardNewtonSchulz(GramNewtonSchulz):
    """
    Standard Newton-Schulz orthogonalization.
    Equivalent to GramNewtonSchulz with use_gram_newton_schulz=False.
    """
    def __init__(
        self,
        ns_epsilon: float = 1e-7,
        ns_use_kernels: bool = True,
        ns_coefficients: Optional[List[List[float]]] = None,
        compile_kwargs: Optional[Dict] = {"fullgraph": True, "mode": "reduce-overhead"},
    ):
        super().__init__(
            ns_epsilon=ns_epsilon,
            ns_use_kernels=ns_use_kernels,
            ns_coefficients=ns_coefficients,
            use_gram_newton_schulz=False,
            compile_kwargs=compile_kwargs,
        )