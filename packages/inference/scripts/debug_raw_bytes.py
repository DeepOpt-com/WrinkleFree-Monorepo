#!/usr/bin/env python3
"""Debug raw bytes around problematic tensor offsets."""

import sys
import struct
import mmap


def debug_bytes(path):
    with open(path, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

        # From our debug output:
        # blk.1.ffn_sub_norm: relative=678530304, absolute=686881664

        data_offset = 8351360
        blk1_ffn_sub_norm_rel = 678530304
        blk1_ffn_sub_norm_abs = data_offset + blk1_ffn_sub_norm_rel

        # Also look at blk.1.ffn_down which should end right before
        blk1_ffn_down_rel = 674106592
        blk1_ffn_down_size = 6912 * 2560 // 4  # I2_S: 2 bits per element
        blk1_ffn_down_end = blk1_ffn_down_rel + blk1_ffn_down_size

        print(f"=== DEBUG RAW BYTES ===")
        print(f"Data section starts at: {data_offset}")
        print()
        print(f"blk.1.ffn_down (I2_S):")
        print(f"  relative start: {blk1_ffn_down_rel}")
        print(f"  size: {blk1_ffn_down_size}")
        print(f"  relative end: {blk1_ffn_down_end}")
        print(f"  aligned end: {(blk1_ffn_down_end + 31) // 32 * 32}")
        print()
        print(f"blk.1.ffn_sub_norm (F32):")
        print(f"  relative offset: {blk1_ffn_sub_norm_rel}")
        print(f"  absolute offset: {blk1_ffn_sub_norm_abs}")
        print()

        # Read bytes from end of blk.1.ffn_down
        end_abs = data_offset + blk1_ffn_down_end
        print(f"Last 32 bytes of blk.1.ffn_down (at abs {end_abs - 32}):")
        last_bytes = mm[end_abs - 32:end_abs]
        print(f"  {' '.join(f'{b:02x}' for b in last_bytes)}")
        print()

        # Read alignment padding (if any)
        padding_size = blk1_ffn_sub_norm_rel - blk1_ffn_down_end
        if padding_size > 0:
            print(f"Padding bytes ({padding_size} bytes):")
            padding = mm[data_offset + blk1_ffn_down_end:data_offset + blk1_ffn_sub_norm_rel]
            print(f"  {' '.join(f'{b:02x}' for b in padding)}")
            print()

        # Read first 64 bytes of blk.1.ffn_sub_norm
        print(f"First 64 bytes of blk.1.ffn_sub_norm (at abs {blk1_ffn_sub_norm_abs}):")
        first_bytes = mm[blk1_ffn_sub_norm_abs:blk1_ffn_sub_norm_abs + 64]
        print(f"  {' '.join(f'{b:02x}' for b in first_bytes[:32])}")
        print(f"  {' '.join(f'{b:02x}' for b in first_bytes[32:])}")
        print()

        # Interpret as F32
        print("Interpreted as F32 values:")
        for i in range(16):
            val = struct.unpack_from('<f', first_bytes, i * 4)[0]
            print(f"  [{i}] {val}")
        print()

        # Now compare with blk.0.ffn_sub_norm which works
        blk0_ffn_sub_norm_abs = data_offset + 661104672
        print(f"First 64 bytes of blk.0.ffn_sub_norm (at abs {blk0_ffn_sub_norm_abs}):")
        first_bytes0 = mm[blk0_ffn_sub_norm_abs:blk0_ffn_sub_norm_abs + 64]
        print(f"  {' '.join(f'{b:02x}' for b in first_bytes0[:32])}")
        print(f"  {' '.join(f'{b:02x}' for b in first_bytes0[32:])}")
        print()
        print("Interpreted as F32 values:")
        for i in range(16):
            val = struct.unpack_from('<f', first_bytes0, i * 4)[0]
            print(f"  [{i}] {val}")

        print()

        # Check if tensor data is actually stored sequentially by file position
        # Maybe Microsoft stores tensors in a different order?
        print("=== CHECKING TENSOR DATA SEQUENCE ===")

        # Let's check bytes at regular intervals in the data section
        # to see where F32 norm data actually is
        print("Scanning for F32 ~1.0 values (RMSNorm gamma patterns)...")

        found_count = 0
        for rel_offset in range(0, 100_000_000, 1_000_000):  # Every 1MB
            abs_off = data_offset + rel_offset
            if abs_off + 16 > len(mm):
                break
            chunk = mm[abs_off:abs_off + 16]
            floats = struct.unpack_from('<4f', chunk)

            # Check if first 4 floats look like gamma values (0.5 < x < 2.0)
            if all(0.5 < f < 2.0 for f in floats):
                print(f"  Found F32 ~1.0 pattern at rel_offset {rel_offset}: {floats}")
                found_count += 1
                if found_count > 10:
                    print("  ... (stopping after 10)")
                    break


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: debug_raw_bytes.py <path-to-gguf>")
        sys.exit(1)
    debug_bytes(sys.argv[1])
