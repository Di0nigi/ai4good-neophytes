import unittest

import torch
import torch.nn as nn

from models.semseg_plm import (
    forward_patches_in_chunks,
    merge_output_patches_to_image,
    patch_image_to_batch,
)


class PatchMicroBatchTest(unittest.TestCase):
    def test_micro_batches_preserve_patch_order_and_stitched_output(self):
        torch.manual_seed(7)
        image = torch.randn(1, 3, 128, 128)
        head = nn.Conv2d(3, 7, kernel_size=1).eval()

        patches, count, stride, pad = patch_image_to_batch(image, patch_size=64)
        forward_fn = lambda batch, _: head(batch)

        full = forward_patches_in_chunks(patches, None, None, forward_fn)
        chunked = forward_patches_in_chunks(patches, None, 2, forward_fn)

        self.assertTrue(torch.equal(full, chunked))

        full_image = merge_output_patches_to_image(
            full, count, 64, stride, 1, 128, pad
        )
        chunked_image = merge_output_patches_to_image(
            chunked, count, 64, stride, 1, 128, pad
        )
        self.assertTrue(torch.equal(full_image, chunked_image))


if __name__ == "__main__":
    unittest.main()
