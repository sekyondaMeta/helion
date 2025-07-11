from __future__ import annotations

import unittest

import torch

import helion
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import code_and_output
import helion.language as hl


class TestErrors(TestCase):
    def test_tile_unpacking(self):
        @helion.kernel()
        def sum_kernel(x: torch.Tensor) -> torch.Tensor:
            batch, seq_len, hidden = x.size()
            out = x.new_empty(batch, hidden)
            for tile_batch, tile_hidden in hl.tile(batch, hidden):
                out[tile_batch, tile_hidden] = x[tile_batch, :, tile_hidden].sum(1)
            return out

        with self.assertRaises(helion.exc.FailedToUnpackTile):
            code_and_output(sum_kernel, (torch.randn(2, 3, 4, device=DEVICE),))

    def test_tile_overpacking(self):
        @helion.kernel()
        def fn(x: torch.Tensor) -> torch.Tensor:
            batch = x.size(0)
            out = x.new_empty(batch)
            for tile_wrapped_in_tuple in hl.tile([batch]):
                out[tile_wrapped_in_tuple] = x[tile_wrapped_in_tuple, :].sum(1)
            return out

        with self.assertRaises(helion.exc.OverpackedTile):
            code_and_output(fn, (torch.randn(100, 100, device=DEVICE),))

    def test_invalid_config_insufficient_block_sizes(self):
        """Test that InvalidConfig shows helpful message for missing block sizes."""

        @helion.kernel(config=helion.Config(block_sizes=[32, 64]))
        def fn(x: torch.Tensor) -> torch.Tensor:
            batch, seq_len, hidden = x.size()
            out = torch.empty_like(x)
            for tile_batch, tile_seq, tile_hidden in hl.tile([batch, seq_len, hidden]):
                out[tile_batch, tile_seq, tile_hidden] = x[
                    tile_batch, tile_seq, tile_hidden
                ]
            return out

        with self.assertRaisesRegex(
            helion.exc.InvalidConfig,
            r"Not enough values for config.*expected 3 block sizes.*got 2.*"
            r"Did you forget to specify block sizes for all your hl\.tile\(\) dimensions\?",
        ):
            code_and_output(
                fn,
                (torch.randn(4, 8, 16, device=DEVICE),),
            )

    def test_rank_mismatch_assignment(self):
        """Test that RankMismatch shows tensor shapes in assignment errors."""

        @helion.kernel()
        def fn(x: torch.Tensor) -> torch.Tensor:
            batch, seq_len = x.size()
            out = x.new_empty(batch, seq_len)
            for tile_batch, tile_seq in hl.tile([batch, seq_len]):
                scalar_val = x[tile_batch, 0].sum()  # Creates 0D tensor
                out[tile_batch, tile_seq] = scalar_val  # 0D -> 2D assignment
            return out

        with self.assertRaisesRegex(
            helion.exc.RankMismatch,
            r"Expected ndim=2, but got ndim=0.*You have too few indices",
        ):
            code_and_output(fn, (torch.randn(4, 8, device=DEVICE),))

    def test_rank_mismatch_indexing(self):
        """Test that RankMismatch shows tensor shapes in indexing errors."""

        @helion.kernel()
        def fn(x: torch.Tensor) -> torch.Tensor:
            batch = x.size(0)
            out = x.new_empty(batch)
            for tile_batch in hl.tile([batch]):
                scalar_val = x[tile_batch].sum()  # 1d index for 2d tensor
                out = scalar_val
            return out

        with self.assertRaisesRegex(
            helion.exc.RankMismatch,
            r"Expected ndim=2, but got ndim=1.*You have too few indices",
        ):
            code_and_output(fn, (torch.randn(4, 8, device=DEVICE),))

    def test_rank_mismatch_indexing_too_many(self):
        @helion.kernel()
        def fn(x: torch.Tensor) -> torch.Tensor:
            batch = x.size(0)
            fill = x.new_empty(batch, batch)
            for tile_batch in hl.tile(batch):
                fill = x[tile_batch, tile_batch]  # 2d index for 1d tensor
            return fill

        with self.assertRaisesRegex(
            helion.exc.RankMismatch,
            r"Expected ndim=1, but got ndim=2.*You have too many indices",
        ):
            code_and_output(fn, (torch.randn(8, device=DEVICE),))

    def test_invalid_device_for_loop(self):
        """Test that InvalidDeviceForLoop is raised for invalid for loops on device."""

        @helion.kernel()
        def fn(x: torch.Tensor) -> torch.Tensor:
            batch = x.size(0)
            out = x.new_empty(batch)
            for tile_batch in hl.tile(batch):
                for i in {1: None, 2: None, 3: None}:
                    out[tile_batch] = x[tile_batch] + i
            return out

        with self.assertRaises(helion.exc.InvalidDeviceForLoop):
            code_and_output(fn, (torch.randn(8, device=DEVICE),))


if __name__ == "__main__":
    unittest.main()
