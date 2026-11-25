"""Regression: encoder-decoder generate output is decoder-only.

The hf adapter must not slice a seq2seq generation by the encoder
input length — HF returns decoder ids only for encoder-decoder
models, so slicing dropped every generation shorter than the prompt
and fabricated empty outputs (trivially-passing probe records).
Causal LMs DO return prompt+continuation and keep the slice.
"""

from clause16e.adapters.hf_local import strip_prompt


def test_encoder_decoder_output_is_never_sliced():
    # seq2seq row: decoder ids only, shorter than the "prompt" length.
    row = [7, 8, 9]
    assert strip_prompt(row, 50, is_encoder_decoder=True) == row


def test_causal_output_slices_off_the_prompt():
    row = [1, 2, 3, 4, 5]
    assert strip_prompt(row, 3, is_encoder_decoder=False) == [4, 5]
