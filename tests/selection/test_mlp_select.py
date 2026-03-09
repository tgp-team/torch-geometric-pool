import pytest
import torch

from tgp.select import MLPSelect, SelectOutput


def test_mlp_select_batched_2d_input_is_unsqueezed():
    torch.manual_seed(0)
    x = torch.randn(4, 3)
    selector = MLPSelect(in_channels=3, k=2, batched_representation=True)

    out = selector(x=x, mask=None)

    assert isinstance(out, SelectOutput)
    assert out.s.shape == (1, 4, 2)
    assert out.in_mask is None


def test_mlp_select_batched_without_mask():
    torch.manual_seed(0)
    x = torch.randn(2, 4, 3)
    selector = MLPSelect(in_channels=3, k=2, batched_representation=True)

    out = selector(x=x, mask=None)

    assert isinstance(out, SelectOutput)
    assert out.s.shape == (2, 4, 2)
    assert out.in_mask is None
    torch.testing.assert_close(
        out.s.sum(dim=-1), torch.ones(2, 4), atol=1e-6, rtol=1e-6
    )


def test_mlp_select_batched_with_mask():
    torch.manual_seed(1)
    x = torch.randn(2, 4, 3)
    mask = torch.tensor(
        [[True, True, False, False], [True, False, True, False]], dtype=torch.bool
    )
    selector = MLPSelect(in_channels=3, k=2, batched_representation=True)

    out = selector(x=x, mask=mask)

    assert isinstance(out, SelectOutput)
    assert out.in_mask is not None
    assert torch.equal(out.in_mask, mask)
    assert torch.all(out.s[~mask] == 0)
    torch.testing.assert_close(
        out.s[mask].sum(dim=-1),
        torch.ones(mask.sum().item(), device=out.s.device),
        atol=1e-6,
        rtol=1e-6,
    )


def test_mlp_select_unbatched_forward_repr_and_reset():
    torch.manual_seed(2)
    x = torch.randn(6, 3)
    batch = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
    selector = MLPSelect(
        in_channels=[3, 4],
        k=2,
        batched_representation=False,
        act="relu",
        dropout=0.0,
        s_inv_op="transpose",
    )

    out = selector(x=x, batch=batch)

    assert isinstance(out, SelectOutput)
    assert out.s.shape == (6, 2)
    assert out.batch is not None
    assert torch.equal(out.batch, batch)
    repr_str = repr(selector)
    assert "MLPSelect(" in repr_str
    assert "in_channels=[3, 4]" in repr_str
    assert "k=2" in repr_str
    selector.reset_parameters()


def test_mlp_select_unbatched_requires_2d_inputs():
    selector = MLPSelect(in_channels=3, k=2, batched_representation=False)
    x = torch.randn(1, 4, 3)

    with pytest.raises(AssertionError, match="x must be of shape \\[N, F\\]"):
        selector(x=x)
