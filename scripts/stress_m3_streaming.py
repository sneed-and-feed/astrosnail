"""
Adversarial Stress Test Suite for Milestone 3 (Memory & Streaming)
Challenger: Challenger M3-1
"""

import gc
import io
import math
import os
import sys
import tempfile
import tracemalloc
from pathlib import Path
import numpy as np

# Ensure astrosnail is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from astrosnail.io import (
    FITS_BLOCK_SIZE,
    read_fits_header,
    stream_fits_cube,
    write_fits_cube,
    stream_lc_records,
    stream_lc_chunks,
    parse_lc_table,
    write_lc_file,
    stream_cat_records,
    stream_cat_chunks,
    parse_cat_file,
    write_cat_file,
    generate_standard_catalogs,
)


def run_all_challenges():
    print("=" * 70)
    print("CHALLENGER M3-1: MEMORY & STREAMING EMPIRICAL ADVERSARIAL STRESS SUITE")
    print("=" * 70)

    results = {}

    # -------------------------------------------------------------------------
    # CHALLENGE 1: ACTIVE HEAP BOUNDS ON FITS STREAMING (64 MB & 128 MB)
    # -------------------------------------------------------------------------
    print("\n--- Challenge 1: Active Heap Bounds on FITS Streaming ---")

    # 1.1: 64 MB FITS cube streaming
    print("[1.1] Testing 64 MB FITS cube streaming...")
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as tf:
        fits_64mb_path = tf.name

    try:
        # 1024 x 128 x 128 float32 = 64.0 MB
        shape_64 = (1024, 128, 128)
        data_64 = np.zeros(shape_64, dtype=np.float32)
        write_fits_cube(fits_64mb_path, data_64, chunk_z=16)
        del data_64
        gc.collect()

        file_size_64 = os.path.getsize(fits_64mb_path)
        print(f"  64 MB file created: {file_size_64} bytes ({file_size_64 / (1024*1024):.2f} MB)")

        # Profile zero-copy stream (copy=False)
        tracemalloc.start()
        tracemalloc.reset_peak()
        n_chunks_64 = 0
        sum_vals_64 = 0.0
        for slab in stream_fits_cube(fits_64mb_path, chunk_z=16, copy=False):
            n_chunks_64 += 1
            sum_vals_64 += float(np.sum(slab[:1, :1, :1]))
        _, peak_bytes_64_zero = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mb_64_zero = peak_bytes_64_zero / (1024 * 1024)
        print(f"  [Zero-Copy] Chunks: {n_chunks_64}, Peak Heap: {peak_mb_64_zero:.4f} MB ({peak_bytes_64_zero} bytes)")

        # Profile reuse_buffer=True
        tracemalloc.start()
        tracemalloc.reset_peak()
        n_chunks_64_reuse = 0
        for slab in stream_fits_cube(fits_64mb_path, chunk_z=16, reuse_buffer=True):
            n_chunks_64_reuse += 1
        _, peak_bytes_64_reuse = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mb_64_reuse = peak_bytes_64_reuse / (1024 * 1024)
        print(f"  [Reuse-Buffer] Chunks: {n_chunks_64_reuse}, Peak Heap: {peak_mb_64_reuse:.4f} MB ({peak_bytes_64_reuse} bytes)")

        results["64mb_zero_copy_peak_mb"] = peak_mb_64_zero
        results["64mb_reuse_buffer_peak_mb"] = peak_mb_64_reuse
        assert peak_mb_64_zero < 2.0, f"64MB Zero-copy peak heap exceeded 2.0 MB: {peak_mb_64_zero} MB"
        assert peak_mb_64_reuse < 2.0, f"64MB Reuse-buffer peak heap exceeded 2.0 MB: {peak_mb_64_reuse} MB"
        print("  => PASS: 64 MB Heap Bounds satisfied (< 2.0 MB)")
    finally:
        if os.path.exists(fits_64mb_path):
            try:
                os.remove(fits_64mb_path)
            except Exception as e:
                print(f"  Warning: failed to delete temp file {fits_64mb_path}: {e}")

    # 1.2: 128 MB FITS cube streaming
    print("\n[1.2] Testing 128 MB FITS cube streaming...")
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as tf:
        fits_128mb_path = tf.name

    try:
        # 2048 x 128 x 128 float32 = 128.0 MB
        shape_128 = (2048, 128, 128)
        data_128 = np.zeros(shape_128, dtype=np.float32)
        write_fits_cube(fits_128mb_path, data_128, chunk_z=16)
        del data_128
        gc.collect()

        file_size_128 = os.path.getsize(fits_128mb_path)
        print(f"  128 MB file created: {file_size_128} bytes ({file_size_128 / (1024*1024):.2f} MB)")

        # Profile zero-copy stream (copy=False)
        tracemalloc.start()
        tracemalloc.reset_peak()
        n_chunks_128 = 0
        for slab in stream_fits_cube(fits_128mb_path, chunk_z=16, copy=False):
            n_chunks_128 += 1
            _ = float(np.min(slab))
        _, peak_bytes_128_zero = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mb_128_zero = peak_bytes_128_zero / (1024 * 1024)
        print(f"  [Zero-Copy] Chunks: {n_chunks_128}, Peak Heap: {peak_mb_128_zero:.4f} MB ({peak_bytes_128_zero} bytes)")

        # Profile reuse_buffer=True
        tracemalloc.start()
        tracemalloc.reset_peak()
        n_chunks_128_reuse = 0
        for slab in stream_fits_cube(fits_128mb_path, chunk_z=16, reuse_buffer=True):
            n_chunks_128_reuse += 1
        _, peak_bytes_128_reuse = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mb_128_reuse = peak_bytes_128_reuse / (1024 * 1024)
        print(f"  [Reuse-Buffer] Chunks: {n_chunks_128_reuse}, Peak Heap: {peak_mb_128_reuse:.4f} MB ({peak_bytes_128_reuse} bytes)")

        results["128mb_zero_copy_peak_mb"] = peak_mb_128_zero
        results["128mb_reuse_buffer_peak_mb"] = peak_mb_128_reuse
        assert peak_mb_128_zero < 2.0, f"128MB Zero-copy peak heap exceeded 2.0 MB: {peak_mb_128_zero} MB"
        assert peak_mb_128_reuse < 2.0, f"128MB Reuse-buffer peak heap exceeded 2.0 MB: {peak_mb_128_reuse} MB"
        print("  => PASS: 128 MB Heap Bounds satisfied (< 2.0 MB)")
    finally:
        if os.path.exists(fits_128mb_path):
            try:
                os.remove(fits_128mb_path)
            except Exception as e:
                print(f"  Warning: failed to delete temp file {fits_128mb_path}: {e}")

    # -------------------------------------------------------------------------
    # CHALLENGE 2: GENERATOR LIFETIME OVERLAP & REFERENCE HOLDING
    # -------------------------------------------------------------------------
    print("\n--- Challenge 2: Generator Lifetime Overlap & Reference Holding ---")
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as tf:
        fits_overlap_path = tf.name

    try:
        # Create 64 MB cube
        shape = (1024, 128, 128)
        data = np.arange(1024 * 128 * 128, dtype=np.float32).reshape(shape)
        write_fits_cube(fits_overlap_path, data, chunk_z=16)
        del data
        gc.collect()

        # 2.1: Accumulate references to all 64 slabs in a list
        print("[2.1] Accumulating references to all 64 zero-copy slabs in a list...")
        tracemalloc.start()
        tracemalloc.reset_peak()
        held_slabs = []
        for slab in stream_fits_cube(fits_overlap_path, chunk_z=16, copy=False):
            held_slabs.append(slab)

        _, peak_bytes_held = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mb_held = peak_bytes_held / (1024 * 1024)
        print(f"  Held {len(held_slabs)} slabs. Peak Heap: {peak_mb_held:.4f} MB ({peak_bytes_held} bytes)")

        # Verify that all slabs are still readable and bit-accurate
        assert len(held_slabs) == 64
        assert held_slabs[0][0, 0, 0] == 0.0
        assert held_slabs[-1][-1, -1, -1] == float(1024 * 128 * 128 - 1)

        # Check read-only protection of zero-copy memmap slice views
        try:
            held_slabs[0][0, 0, 0] = 999.0
            print("  WARNING: Memmap slice is writable! (Expected read-only)")
            is_readonly = False
        except ValueError as ve:
            print(f"  Read-only enforcement confirmed: {ve}")
            is_readonly = True

        del held_slabs
        gc.collect()

        # 2.2: Partial consumption and early generator exit
        print("[2.2] Testing partial consumption (break after 5 slabs)...")
        gen = stream_fits_cube(fits_overlap_path, chunk_z=16, copy=False)
        partial_slabs = []
        for i, slab in enumerate(gen):
            partial_slabs.append(slab)
            if i == 4:
                break
        assert len(partial_slabs) == 5
        del gen
        del partial_slabs
        gc.collect()
        print("  Early generator termination cleanly cleaned up.")

        # 2.3: Holding references with reuse_buffer=True
        print("[2.3] Testing reference holding with reuse_buffer=True...")
        tracemalloc.start()
        tracemalloc.reset_peak()
        reuse_refs = []
        for slab in stream_fits_cube(fits_overlap_path, chunk_z=16, reuse_buffer=True):
            reuse_refs.append(slab)
        _, peak_bytes_reuse_refs = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mb_reuse_refs = peak_bytes_reuse_refs / (1024 * 1024)
        print(f"  Held {len(reuse_refs)} reuse_buffer references. Peak Heap: {peak_mb_reuse_refs:.4f} MB")
        # In reuse_buffer mode, all references point to the same buffer
        assert reuse_refs[0].base is reuse_refs[1].base or reuse_refs[0] is reuse_refs[1] or np.shares_memory(reuse_refs[0], reuse_refs[1])
        del reuse_refs
        gc.collect()

        results["held_slabs_peak_mb"] = peak_mb_held
        results["is_readonly"] = is_readonly
        print("  => PASS: Generator Lifetime Overlap verified.")
    finally:
        if os.path.exists(fits_overlap_path):
            try:
                os.remove(fits_overlap_path)
            except Exception as e:
                print(f"  Warning: failed to delete temp file {fits_overlap_path}: {e}")

    # -------------------------------------------------------------------------
    # CHALLENGE 3: EDGE & CORRUPTED FILE ROBUSTNESS
    # -------------------------------------------------------------------------
    print("\n--- Challenge 3: Edge & Corrupted File Robustness ---")

    # 3.1: Zero-byte files
    print("\n[3.1] Testing zero-byte files...")
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as tf:
        zero_fits_path = tf.name
    with tempfile.NamedTemporaryFile(suffix=".lc", delete=False) as tf:
        zero_lc_path = tf.name
    with tempfile.NamedTemporaryFile(suffix=".cat", delete=False) as tf:
        zero_cat_path = tf.name

    try:
        # Zero-byte FITS
        try:
            hdr = read_fits_header(zero_fits_path)
            print(f"  read_fits_header on 0-byte file: shape={hdr.shape}, cards={len(hdr.cards)}")
            zero_fits_header_ok = True
        except Exception as e:
            print(f"  read_fits_header on 0-byte file raised: {type(e).__name__}: {e}")
            zero_fits_header_ok = False

        try:
            for _ in stream_fits_cube(zero_fits_path):
                pass
            print("  stream_fits_cube on 0-byte file yielded without error (unexpected)")
            zero_fits_cube_caught = False
        except ValueError as ve:
            print(f"  stream_fits_cube on 0-byte file properly rejected: {ve}")
            zero_fits_cube_caught = True
        except Exception as e:
            print(f"  stream_fits_cube on 0-byte file raised: {type(e).__name__}: {e}")
            zero_fits_cube_caught = True

        # Zero-byte LC
        lc_recs = list(stream_lc_records(zero_lc_path))
        lc_table = parse_lc_table(zero_lc_path)
        print(f"  stream_lc_records on 0-byte file: {len(lc_recs)} records")
        print(f"  parse_lc_table on 0-byte file: shape={lc_table.shape}")
        assert len(lc_recs) == 0
        assert lc_table.shape == (0, 3)

        # Zero-byte CAT
        cat_recs = list(stream_cat_records(zero_cat_path))
        cat_table = parse_cat_file(zero_cat_path)
        print(f"  stream_cat_records on 0-byte file: {len(cat_recs)} records")
        print(f"  parse_cat_file on 0-byte file: {len(cat_table)} records")
        assert len(cat_recs) == 0
        assert len(cat_table) == 0

        print("  => PASS: Zero-byte files handled gracefully.")
    finally:
        for p in [zero_fits_path, zero_lc_path, zero_cat_path]:
            if os.path.exists(p):
                os.remove(p)

    # 3.2: Truncated FITS files
    print("\n[3.2] Testing truncated FITS files...")
    # 3.2.1: Header truncated (< 2880 bytes, e.g. 500 bytes)
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as tf:
        trunc_hdr_path = tf.name
        tf.write(b"SIMPLE  =                    T / Standard FITS\nBITPIX  =                  -32\n")

    try:
        try:
            hdr = read_fits_header(trunc_hdr_path)
            print(f"  Truncated header (<2880 bytes): shape={hdr.shape}, cards={hdr.cards}")
        except Exception as e:
            print(f"  Truncated header raised: {type(e).__name__}: {e}")

        try:
            for _ in stream_fits_cube(trunc_hdr_path):
                pass
            print("  stream_fits_cube on truncated header did not raise (unexpected)")
        except ValueError as ve:
            print(f"  stream_fits_cube on truncated header caught ValueError: {ve}")
        except Exception as e:
            print(f"  stream_fits_cube on truncated header caught: {type(e).__name__}: {e}")
    finally:
        if os.path.exists(trunc_hdr_path):
            os.remove(trunc_hdr_path)

    # 3.2.2: Valid header claiming 32x24x24 cube, but data truncated (0 bytes data or only 100 bytes)
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as tf:
        trunc_data_path = tf.name
        # Create full header
        full_cube = np.zeros((32, 24, 24), dtype=np.float32)
        write_fits_cube(trunc_data_path, full_cube)
        # Now truncate the file to just header (2880 bytes) plus 100 bytes of data
        tf.close()
        with open(trunc_data_path, "r+b") as fp:
            fp.truncate(2880 + 100)

    try:
        print("  Testing valid header with truncated data (2880 + 100 bytes)...")
        hdr = read_fits_header(trunc_data_path)
        print(f"  Header parsed shape: {hdr.shape}")
        try:
            for slab in stream_fits_cube(trunc_data_path, chunk_z=8, copy=False):
                print(f"    Yielded slab: {slab.shape}")
            print("  stream_fits_cube completed without raising on truncated data! (Unexpected)")
        except (ValueError, OSError) as e:
            print(f"  stream_fits_cube properly raised on truncated data: {type(e).__name__}: {e}")
    finally:
        if os.path.exists(trunc_data_path):
            os.remove(trunc_data_path)

    # 3.3: Headers lacking END card
    print("\n[3.3] Testing headers lacking END card...")
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as tf:
        no_end_path = tf.name
        # Create 2880 bytes of cards without END
        cards = [
            b"SIMPLE  =                    T / Standard FITS                                  ",
            b"BITPIX  =                  -32 / 32-bit float                                   ",
            b"NAXIS   =                    2 / 2D image                                       ",
            b"NAXIS1  =                   10 / Width                                          ",
            b"NAXIS2  =                   10 / Height                                         ",
        ]
        # Pad with COMMENT cards to fill exactly 2880 bytes
        while len(cards) < 36:
            cards.append(b"COMMENT No END card here                                                        ")
        raw = b"".join(cards)
        assert len(raw) == 2880
        tf.write(raw)
        # Add 100 floats of data
        tf.write(np.zeros((10, 10), dtype=">f4").tobytes())

    try:
        hdr = read_fits_header(no_end_path)
        print(f"  Header without END parsed: shape={hdr.shape}, NAXIS={hdr.cards.get('NAXIS')}, data_offset={hdr.data_offset}")
        # What does stream_fits_cube do?
        try:
            slabs = list(stream_fits_cube(no_end_path, chunk_z=5, copy=False))
            print(f"  stream_fits_cube yielded {len(slabs)} slabs of shape {slabs[0].shape}")
        except Exception as e:
            print(f"  stream_fits_cube raised: {type(e).__name__}: {e}")
        del slabs
        del hdr
        gc.collect()
    finally:
        if os.path.exists(no_end_path):
            try:
                os.remove(no_end_path)
            except Exception as e:
                print(f"  Warning: failed to delete {no_end_path}: {e}")

    # 3.4: Non-2880 byte alignments
    print("\n[3.4] Testing non-2880 byte alignments...")
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as tf:
        unaligned_path = tf.name
        # Write valid cube then append 13 unaligned garbage bytes
        cube = np.ones((8, 16, 16), dtype=np.float32)
        write_fits_cube(unaligned_path, cube)
        tf.close()
        with open(unaligned_path, "ab") as fp:
            fp.write(b"EXTRA_GARBAGE")

    try:
        size = os.path.getsize(unaligned_path)
        print(f"  Unaligned file size: {size} (size % 2880 = {size % 2880})")
        hdr = read_fits_header(unaligned_path)
        print(f"  Header shape: {hdr.shape}")
        # Verify streaming still reads valid data despite trailing unaligned bytes
        slabs = list(stream_fits_cube(unaligned_path, chunk_z=4, copy=False))
        assert len(slabs) == 2
        assert slabs[0].shape == (4, 16, 16)
        print(f"  stream_fits_cube read all {len(slabs)} slabs correctly despite unaligned trailing bytes.")
        del slabs
        del hdr
        gc.collect()
    finally:
        if os.path.exists(unaligned_path):
            try:
                os.remove(unaligned_path)
            except Exception as e:
                print(f"  Warning: failed to delete {unaligned_path}: {e}")

    # 3.5: NaN / Inf pixel values in FITS cubes
    print("\n[3.5] Testing NaN and Inf pixel values in FITS cubes...")
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as tf:
        nan_cube_path = tf.name

    try:
        cube_special = np.zeros((16, 16, 16), dtype=np.float32)
        cube_special[0, 0, 0] = np.nan
        cube_special[1, 1, 1] = np.inf
        cube_special[2, 2, 2] = -np.inf
        cube_special[3, 3, 3] = 1e30

        write_fits_cube(nan_cube_path, cube_special)

        # Read back with copy=False
        slabs_zero = list(stream_fits_cube(nan_cube_path, chunk_z=4, copy=False))
        reconstructed = np.concatenate(slabs_zero, axis=0)
        assert np.isnan(reconstructed[0, 0, 0])
        assert np.isinf(reconstructed[1, 1, 1]) and reconstructed[1, 1, 1] > 0
        assert np.isinf(reconstructed[2, 2, 2]) and reconstructed[2, 2, 2] < 0
        assert reconstructed[3, 3, 3] == np.float32(1e30)
        print("  Zero-copy stream preserves NaN, +Inf, -Inf, and large floats.")
        del slabs_zero
        del reconstructed
        gc.collect()

        # Read back with copy=True and scaling
        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as tf2:
            scaled_nan_path = tf2.name
        write_fits_cube(scaled_nan_path, cube_special, bscale=2.5, bzero=10.0)
        slabs_scaled = list(stream_fits_cube(scaled_nan_path, chunk_z=4, copy=True))
        rec_scaled = np.concatenate(slabs_scaled, axis=0)
        assert np.isnan(rec_scaled[0, 0, 0])
        assert np.isinf(rec_scaled[1, 1, 1])
        print("  Scaled stream preserves NaN and Inf correctly.")
        del slabs_scaled
        del rec_scaled
        gc.collect()
        if os.path.exists(scaled_nan_path):
            try:
                os.remove(scaled_nan_path)
            except Exception as e:
                print(f"  Warning: failed to delete {scaled_nan_path}: {e}")
    finally:
        if os.path.exists(nan_cube_path):
            try:
                os.remove(nan_cube_path)
            except Exception as e:
                print(f"  Warning: failed to delete {nan_cube_path}: {e}")

    # 3.6: Streaming .lc with corrupted lines, missing columns, non-numeric strings
    print("\n[3.6] Testing .lc with corrupted lines, missing columns, non-numeric strings...")
    with tempfile.NamedTemporaryFile(suffix=".lc", mode="w", delete=False, encoding="utf-8") as tf:
        corrupt_lc_path = tf.name
        tf.write("# Test corrupted light curve\n")
        tf.write("# time flux flux_err\n")
        tf.write("1.0 100.5 0.5\n")               # valid 3-col
        tf.write("2.0 101.2\n")                   # valid 2-col (default err 0.0)
        tf.write("   \n")                         # blank line
        tf.write("# another comment\n")           # comment line
        tf.write("3.0  102.5   0.4   extra_col\n") # 4-col (extra columns ignored)
        tf.write("4.0, 103.1, 0.3\n")             # comma-separated
        tf.write("single_column_only\n")          # 1-col (missing columns)
        tf.write("5.0\n")                         # 1-col numeric
        tf.write("bad_time 104.0 0.5\n")          # non-numeric time
        tf.write("6.0 bad_flux 0.5\n")            # non-numeric flux
        tf.write("7.0 105.0 bad_err\n")           # non-numeric err
        tf.write("8.0 NaN 0.5\n")                 # NaN float string
        tf.write("9.0 Inf 0.5\n")                 # Inf float string
        tf.write("10.0 106.0 0.2\n")              # valid 3-col

    try:
        # Let's inspect behavior of stream_lc_records
        recs = []
        errors = []
        with open(corrupt_lc_path, "r", encoding="utf-8") as fp:
            for line_no, line in enumerate(fp, 1):
                try:
                    # Test line by line
                    r = list(stream_lc_records(io.StringIO(line)))
                    if r:
                        recs.extend(r)
                except Exception as e:
                    errors.append((line_no, line.strip(), type(e).__name__, str(e)))

        print(f"  Total lines tested. Valid/parsed records: {len(recs)}, Exceptions: {len(errors)}")
        for line_no, l_text, err_type, err_msg in errors:
            print(f"    Line {line_no} ('{l_text}') -> {err_type}: {err_msg}")

        # Check NaN and Inf parsing
        nan_rec = [r for r in recs if r[0] == 8.0]
        if nan_rec:
            print(f"  Parsed line 8 (NaN): {nan_rec[0]} -> math.isnan: {math.isnan(nan_rec[0][1])}")
        inf_rec = [r for r in recs if r[0] == 9.0]
        if inf_rec:
            print(f"  Parsed line 9 (Inf): {inf_rec[0]} -> math.isinf: {math.isinf(inf_rec[0][1])}")
    finally:
        if os.path.exists(corrupt_lc_path):
            os.remove(corrupt_lc_path)

    # 3.7: Streaming .cat with irregular whitespace, missing columns, empty files
    print("\n[3.7] Testing .cat with irregular whitespace, missing columns, empty files...")
    with tempfile.NamedTemporaryFile(suffix=".cat", mode="w", delete=False, encoding="utf-8") as tf:
        corrupt_cat_path = tf.name
        tf.write("# Astrometric catalog with anomalies\n")
        tf.write("# source_id ra dec mag mag_err\n")
        tf.write("1001   180.2541   15.3421   18.52   0.03\n")     # standard spaces
        tf.write("1002\t180.2614\t15.3409\t19.14\t0.05\n")         # tabs
        tf.write("SRC_ALPHA, 180.2589, 15.3487, 20.02, 0.08\n")     # commas and string ID
        tf.write("   \n")                                           # blank
        tf.write("1003  180.2641  15.3354\n")                       # 3 columns (missing mag, mag_err)
        tf.write("1004\n")                                          # 1 column
        tf.write("1005  bad_ra  15.3354  20.0  0.1\n")              # bad ra
        tf.write("1006  180.2641  bad_dec  20.0  0.1\n")            # bad dec
        tf.write("1007  180.2641  15.3354  bad_mag  0.1\n")         # bad mag
        tf.write("1008  180.2641  15.3354  20.0  bad_err\n")        # bad err
        tf.write("1009   180.5000   15.5000   17.50   0.02   EXTRA\n") # 6 columns (extra ignored)

    try:
        cat_recs = []
        cat_errors = []
        with open(corrupt_cat_path, "r", encoding="utf-8") as fp:
            for line_no, line in enumerate(fp, 1):
                try:
                    r = list(stream_cat_records(io.StringIO(line)))
                    if r:
                        cat_recs.extend(r)
                except Exception as e:
                    cat_errors.append((line_no, line.strip(), type(e).__name__, str(e)))

        print(f"  Total lines tested. Valid/parsed records: {len(cat_recs)}, Exceptions: {len(cat_errors)}")
        for line_no, l_text, err_type, err_msg in cat_errors:
            print(f"    Line {line_no} ('{l_text}') -> {err_type}: {err_msg}")

        # Verify parsed records
        ids = [r[0] for r in cat_recs]
        print(f"  Successfully parsed IDs: {ids}")
        assert 1001 in ids
        assert 1002 in ids
        assert "SRC_ALPHA" in ids
        assert 1009 in ids
        # 1003 and 1004 had < 5 columns and were silently skipped
        assert 1003 not in ids
        assert 1004 not in ids
        print("  => Irregular whitespace, string IDs, extra columns, and missing columns handled.")
    finally:
        if os.path.exists(corrupt_cat_path):
            os.remove(corrupt_cat_path)

    print("\n" + "=" * 70)
    print("ALL EMPIRICAL CHALLENGES COMPLETED.")
    print("=" * 70)
    return results


if __name__ == "__main__":
    res = run_all_challenges()
    print("Summary results dictionary:", res)
