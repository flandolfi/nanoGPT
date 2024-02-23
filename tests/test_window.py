import torch
import pytest

from model import CausalSelfAttention, GPTConfig


def test_window_fixed():
    block_size = 5
    exp_masks = [
        torch.tensor([[True, False, False, False, False],
                      [False, True, False, False, False],
                      [False, False, True, False, False],
                      [False, False, False, True, False],
                      [False, False, False, False, True]]),
        torch.tensor([[True, False, False, False, False],
                      [True, True, False, False, False],
                      [False, True, True, False, False],
                      [False, False, True, True, False],
                      [False, False, False, True, True]]),
        torch.tensor([[True, False, False, False, False],
                      [True, True, False, False, False],
                      [True, True, True, False, False],
                      [False, True, True, True, False],
                      [False, False, True, True, True]]),
        torch.tensor([[True, False, False, False, False],
                      [True, True, False, False, False],
                      [True, True, True, False, False],
                      [True, True, True, True, False],
                      [False, True, True, True, True]]),
        torch.tensor([[True, False, False, False, False],
                      [True, True, False, False, False],
                      [True, True, True, False, False],
                      [True, True, True, True, False],
                      [True, True, True, True, True]]),
    ]

    for window, exp_mask in enumerate(exp_masks, start=1):
        config = GPTConfig(block_size=block_size, window=window)
        csa = CausalSelfAttention(config)

        assert torch.equal(exp_mask[None, None], csa.attn_mask)

    # If window > block_size defaults to block_size
    config = GPTConfig(block_size=block_size, window=block_size + 1)
    csa = CausalSelfAttention(config)

    assert torch.equal(exp_masks[-1][None, None], csa.attn_mask)

    # If window <= 0 raises assertion error
    with pytest.raises(AssertionError):
        config = GPTConfig(window=0)
        CausalSelfAttention(config)

    with pytest.raises(AssertionError):
        config = GPTConfig(window=-1)
        CausalSelfAttention(config)


@pytest.mark.parametrize("block_size", [10, 20, 30])
@pytest.mark.parametrize("window", list(range(1, 11)))
def test_window_param(block_size, window):
    # Row sum is e.g. 1,2,3,3,3,3,...,3 for window=3
    exp_sum = torch.clip(torch.arange(block_size) + 1, max=window)

    config = GPTConfig(block_size=block_size, window=window)
    csa = CausalSelfAttention(config)

    assert torch.equal(exp_sum, csa.attn_mask[0, 0].sum(-1))

    # Column sum is row sum reversed, i.e. 3,...,3,3,3,2,1
    assert torch.equal(exp_sum.flip(0), csa.attn_mask[0, 0].sum(-2))

    # Expected total number of positive values in mask
    exp_total = block_size*window - window*(window - 1)//2
    assert csa.attn_mask.sum().item() == exp_total
