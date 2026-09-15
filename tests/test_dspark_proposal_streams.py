# SPDX-License-Identifier: Apache-2.0

import numpy as np

from vllm_metal.v1.dspark.sampling import RequestRandomStreams
from vllm_metal.v1.dspark_proposer import _pad_uniforms


class TestProposalStreamsAreBatchIndependent:
    """``RequestRandomStreams`` promises the same seed reproduces the same draws
    regardless of batch composition. A request must therefore consume exactly its
    own cap of proposal uniforms, however wide the widest co-scheduled request is.
    """

    @staticmethod
    def _draw(
        cap: int, batch_width: int, seed: int = 1234
    ) -> tuple[np.ndarray, np.random.Generator]:
        streams = RequestRandomStreams(np.random.SeedSequence(seed))
        row = _pad_uniforms(streams.proposal.random(cap), batch_width)
        return row, streams.proposal

    def test_row_is_padded_to_the_batch_width(self):
        row, _ = self._draw(cap=4, batch_width=7)
        assert row.shape == (7,)

    def test_the_request_own_positions_are_unaffected_by_a_wider_neighbour(self):
        alone, _ = self._draw(cap=4, batch_width=4)
        batched, _ = self._draw(cap=4, batch_width=7)
        assert np.array_equal(alone[:4], batched[:4])

    def test_a_wider_neighbour_does_not_advance_this_stream(self):
        # The regression: drawing the batch-wide cap consumed 7 uniforms from a
        # request that only drafts 4, so its NEXT step started from a different
        # point purely because of who it was scheduled with.
        _, alone = self._draw(cap=4, batch_width=4)
        _, batched = self._draw(cap=4, batch_width=7)
        assert np.array_equal(alone.random(3), batched.random(3))

    def test_padding_is_never_mistaken_for_a_draw(self):
        row, _ = self._draw(cap=2, batch_width=5)
        assert np.array_equal(row[2:], np.zeros(3))

    def test_a_cap_equal_to_the_batch_width_is_not_padded(self):
        row, _ = self._draw(cap=7, batch_width=7)
        assert row.shape == (7,)
        assert not np.array_equal(row[-1:], np.zeros(1))

    def test_a_zero_cap_consumes_nothing(self):
        _, stopped = self._draw(cap=0, batch_width=7)
        reference = RequestRandomStreams(np.random.SeedSequence(1234)).proposal
        assert np.array_equal(stopped.random(3), reference.random(3))
