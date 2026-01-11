#!/usr/bin/env python3
"""Verify the hypothesis: tensor data is packed WITHOUT padding, but offsets include padding."""

import sys
import struct
import mmap


def verify_hypothesis(path):
    with open(path, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

        data_offset = 8351360

        # According to GGUF spec, offsets should be exact positions in data section
        # Let's compute "packed" offsets (without alignment padding)

        print("=== VERIFYING OFFSET HYPOTHESIS ===")
        print("If tensor data is packed without gaps, offsets should be cumulative sums of tensor sizes")
        print()

        # From our tensor info parsing:
        # blk.0.attn_norm: relative=656670720, size=2560*4=10240
        # blk.0.ffn_down: relative=656680960, size=6912*2560/4=4423680 (I2_S)
        # ...

        # Let's manually trace through the tensors to find cumulative offset
        # We need to know ALL tensors in order

        # Actually, let me just check: if blk.1.ffn_sub_norm data actually starts at
        # blk.1.ffn_down end (678530272), what would the F32 values look like?

        blk1_ffn_down_end = 674106592 + 4423680  # = 678530272
        actual_data_offset = data_offset + blk1_ffn_down_end

        print(f"If blk.1.ffn_sub_norm data is at end of ffn_down (no padding):")
        print(f"  Packed offset: {blk1_ffn_down_end}")
        print(f"  Absolute: {actual_data_offset}")
        print()

        # Read first 64 bytes at this "packed" location
        first_bytes = mm[actual_data_offset:actual_data_offset + 64]
        print("First 64 bytes (if no padding):")
        print(f"  {' '.join(f'{b:02x}' for b in first_bytes[:32])}")
        print(f"  {' '.join(f'{b:02x}' for b in first_bytes[32:])}")
        print()
        print("As F32 values:")
        for i in range(16):
            val = struct.unpack_from('<f', first_bytes, i * 4)[0]
            print(f"  [{i:2d}] {val:.6f}")

        print()
        print("=== COMPARISON ===")
        print()

        # Now compare with recorded offset
        recorded_offset = 678530304
        print(f"At RECORDED offset ({recorded_offset}, abs {data_offset + recorded_offset}):")
        rec_bytes = mm[data_offset + recorded_offset:data_offset + recorded_offset + 64]
        print(f"  {' '.join(f'{b:02x}' for b in rec_bytes[:32])}")
        for i in range(4):
            val = struct.unpack_from('<f', rec_bytes, i * 4)[0]
            print(f"  [{i}] {val:.6e}")

        print()
        print(f"The difference between recorded and packed offset: {recorded_offset - blk1_ffn_down_end} bytes")
        print()

        # Let's also check if ALL layers have this 32-byte offset error
        # by looking at tensor data continuity

        print("=== CHECKING PACKED VS RECORDED FOR MULTIPLE TENSORS ===")

        # Build list of tensors with their recorded offsets and sizes
        # (I'll use the values from our debug output)
        tensors = [
            ("blk.0.attn_norm", 656670720, 2560 * 4, "F32"),
            ("blk.0.ffn_down", 656680960, 6912 * 2560 // 4, "I2_S"),
            ("blk.0.ffn_sub_norm", 661104672, 6912 * 4, "F32"),
            ("blk.0.ffn_gate", 661132320, 2560 * 6912 // 4, "I2_S"),
            ("blk.0.ffn_up", 665556032, 2560 * 6912 // 4, "I2_S"),
            ("blk.0.ffn_norm", 669979744, 2560 * 4, "F32"),
            ("blk.0.attn_sub_norm", 669989984, 2560 * 4, "F32"),
            ("blk.0.attn_k", 670000224, 2560 * 640 // 4, "I2_S"),
            ("blk.0.attn_output", 670409856, 2560 * 2560 // 4, "I2_S"),
            ("blk.0.attn_q", 672048288, 2560 * 2560 // 4, "I2_S"),
            ("blk.0.attn_v", 673686720, 2560 * 640 // 4, "I2_S"),
            ("blk.1.attn_norm", 674096352, 2560 * 4, "F32"),
            ("blk.1.ffn_down", 674106592, 6912 * 2560 // 4, "I2_S"),
            ("blk.1.ffn_sub_norm", 678530304, 6912 * 4, "F32"),
        ]

        # Now trace cumulative offsets
        packed_offset = tensors[0][1]  # Start from first tensor
        print(f"Starting from first tensor at {packed_offset}")
        print()

        for i, (name, recorded, size, dtype) in enumerate(tensors):
            # Check if recorded matches our expectation
            diff = recorded - packed_offset
            if diff == 0:
                status = "✓ MATCH"
            elif diff == 32:
                status = "PADDING +32"
            else:
                status = f"DIFF {diff}"

            print(f"{name}: recorded={recorded}, packed={packed_offset}, {status}")

            # Update packed offset for next tensor
            packed_offset += size
            # For I2_S, check if we need to align to 32 after this tensor
            # (Spoiler: if Microsoft's converter added alignment to offsets but not data, this will accumulate)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: verify_offset_hypothesis.py <path-to-gguf>")
        sys.exit(1)
    verify_hypothesis(sys.argv[1])
