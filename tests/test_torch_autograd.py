import os
import sys

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from maddpg.common.torch_autograd import conjugate_gradient, hvp


def flatten_tensors(tensors):
    return torch.cat([t.reshape(-1) for t in tensors], dim=0)


def set_params_from_flat(params, flat):
    offset = 0
    with torch.no_grad():
        for p in params:
            numel = p.numel()
            p.copy_(flat[offset : offset + numel].view_as(p))
            offset += numel


def build_loss(model, x, y):
    pred = model(x)
    return torch.mean((pred - y) ** 2)


def test_hvp_matches_finite_difference_and_cg():
    torch.manual_seed(7)

    model = torch.nn.Sequential(
        torch.nn.Linear(3, 4),
        torch.nn.Tanh(),
        torch.nn.Linear(4, 1),
    )
    params = [p for p in model.parameters() if p.requires_grad]
    x = torch.randn(8, 3)
    y = torch.randn(8, 1)

    loss = build_loss(model, x, y)
    num_params = sum(p.numel() for p in params)
    v = torch.randn(num_params)

    hv = hvp(loss, params, v).detach()

    theta0 = flatten_tensors([p.detach().clone() for p in params])
    eps = 1e-4

    set_params_from_flat(params, theta0 + eps * v)
    loss_pos = build_loss(model, x, y)
    grad_pos = flatten_tensors(torch.autograd.grad(loss_pos, params, create_graph=False, retain_graph=False)).detach()

    set_params_from_flat(params, theta0 - eps * v)
    loss_neg = build_loss(model, x, y)
    grad_neg = flatten_tensors(torch.autograd.grad(loss_neg, params, create_graph=False, retain_graph=False)).detach()

    set_params_from_flat(params, theta0)

    fd_hv = (grad_pos - grad_neg) / (2.0 * eps)
    max_err = torch.max(torch.abs(hv - fd_hv)).item()
    assert max_err < 2e-2, "HVP finite-difference mismatch too large: {}".format(max_err)

    # Solve Hx = b with CG on a guaranteed SPD quadratic form.
    torch.manual_seed(17)
    n = 16
    theta = torch.randn(n, requires_grad=True)
    m = torch.randn(n, n)
    A = m.t().mm(m) + 0.1 * torch.eye(n)
    loss_quad = 0.5 * theta @ A @ theta

    def hvp_fn(vec):
        return hvp(loss_quad, [theta], vec).detach()

    x_true = torch.randn(n)
    b = hvp_fn(x_true)
    x_cg = conjugate_gradient(hvp_fn, b, max_iter=25)
    residual = torch.norm(hvp_fn(x_cg) - b).item()
    assert residual < 1e-4, "CG residual too large: {}".format(residual)

    print("PASS: hvp finite-difference check and conjugate-gradient solver check.")


if __name__ == "__main__":
    test_hvp_matches_finite_difference_and_cg()
