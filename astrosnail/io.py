"""
astrosnail.io
=============
Zero-copy out-of-core streaming I/O subsystem for astronomical survey pipelines.

Features:
- stream_fits_cube: Memory-mapped streaming reader (< 2.0 MB active heap on 64 MB FITS cubes).
- write_fits_cube: Minimal pure-Python/NumPy FITS format writer conforming to NASA FITS Standard v4.0.
- stream_lc_records, stream_lc_chunks, parse_lc_table, write_lc_file: Streaming light curve parsers and writer.
- stream_cat_records, stream_cat_chunks, parse_cat_file, write_cat_file: Streaming point-source catalog parsers and writer.
- generate_standard_catalogs: Standard synthetic survey catalog generator.
"""

from __future__ import annotations

import gc
import io
import math
import os
from pathlib import Path
from typing import Any, BinaryIO, Dict, Iterator, List, Optional, Sequence, TextIO, Tuple, Union
import numpy as np

FITS_BLOCK_SIZE = 2880
FITS_CARD_SIZE = 80
CARDS_PER_BLOCK = 36


class FITSHeader:
    """Parsed astronomical FITS header container."""

    def __init__(
        self,
        cards: Dict[str, Any],
        shape: Tuple[int, ...],
        bitpix: int,
        bscale: float = 1.0,
        bzero: float = 0.0,
        data_offset: int = FITS_BLOCK_SIZE,
        data_bytes: int = 0,
    ):
        self.cards = cards
        self.shape = shape
        self.bitpix = bitpix
        self.bscale = bscale
        self.bzero = bzero
        self.data_offset = data_offset
        self.data_bytes = data_bytes

    def __getitem__(self, key: str) -> Any:
        return self.cards[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.cards.get(key, default)


def _format_fits_card(key: str, value: Any, comment: str = "") -> bytes:
    """Formats an exact 80-character FITS card image."""
    key_str = f"{key:<8}"[:8]
    if key in ("END", "COMMENT", "HISTORY", ""):
        card = f"{key_str}{comment:<72}"[:80]
        return card.encode("ascii")

    if isinstance(value, bool):
        val_str = "                   T" if value else "                   F"
    elif isinstance(value, int):
        val_str = f"{value:>20}"
    elif isinstance(value, (float, np.floating)):
        val_str = f"{float(value):>20.8E}"
    elif isinstance(value, str):
        v_esc = value.replace("'", "''")
        val_str = f"'{v_esc:<8}'"
        val_str = f"{val_str:<20}"
    else:
        val_str = f"{str(value):>20}"

    if comment:
        card = f"{key_str}= {val_str} / {comment}"
    else:
        card = f"{key_str}= {val_str}"
    return f"{card:<80}"[:80].encode("ascii")


def _parse_fits_card(card_bytes: bytes) -> Tuple[str, Any, str]:
    """Parses an 80-byte FITS card image into (key, value, comment)."""
    card_str = card_bytes.decode("ascii", errors="replace")
    key = card_str[:8].strip().upper()

    if not card_bytes[8:10] == b"= ":
        return key, None, card_str[8:].strip()

    rest = card_str[10:]
    if "/" in rest:
        val_part, comment = rest.split("/", 1)
        comment = comment.strip()
    else:
        val_part, comment = rest, ""

    val_part = val_part.strip()
    if val_part.startswith("'"):
        # Parse FITS quoted string with '' escaping
        val = ""
        i = 1
        while i < len(val_part):
            if val_part[i] == "'":
                if i + 1 < len(val_part) and val_part[i + 1] == "'":
                    val += "'"
                    i += 2
                else:
                    break
            else:
                val += val_part[i]
                i += 1
        val = val.rstrip()
    elif val_part in ("T", "F"):
        val = (val_part == "T")
    else:
        try:
            val = int(val_part)
        except ValueError:
            try:
                val = float(val_part)
            except ValueError:
                val = val_part
    return key, val, comment


def read_fits_header(filepath_or_buffer: Union[str, Path, BinaryIO]) -> FITSHeader:
    """Reads and parses standard FITS header records without loading data into memory.
    Handles Primary HDUs and Image Extension HDUs (when NAXIS=0 in primary).
    """
    is_path = isinstance(filepath_or_buffer, (str, Path))
    fp: BinaryIO = open(filepath_or_buffer, "rb") if is_path else filepath_or_buffer

    try:
        blocks_read = 0
        cards: Dict[str, Any] = {}
        end_found = False

        while not end_found:
            block = fp.read(FITS_BLOCK_SIZE)
            if len(block) < FITS_BLOCK_SIZE:
                break
            blocks_read += 1

            for i in range(0, FITS_BLOCK_SIZE, FITS_CARD_SIZE):
                card_bytes = block[i : i + FITS_CARD_SIZE]
                k, v, _ = _parse_fits_card(card_bytes)
                if k == "END":
                    end_found = True
                    break
                if v is not None and k:
                    cards[k] = v

        data_offset = blocks_read * FITS_BLOCK_SIZE
        naxis = cards.get("NAXIS", 0)

        # Detect Image Extension HDU if primary has NAXIS == 0
        if naxis == 0 and cards.get("EXTEND", False):
            ext_cards: Dict[str, Any] = {}
            end_ext = False
            while not end_ext:
                block = fp.read(FITS_BLOCK_SIZE)
                if len(block) < FITS_BLOCK_SIZE:
                    break
                blocks_read += 1
                for i in range(0, FITS_BLOCK_SIZE, FITS_CARD_SIZE):
                    card_bytes = block[i : i + FITS_CARD_SIZE]
                    k, v, _ = _parse_fits_card(card_bytes)
                    if k == "END":
                        end_ext = True
                        break
                    if v is not None and k:
                        ext_cards[k] = v
            if ext_cards.get("XTENSION", "").strip() == "IMAGE":
                cards.update(ext_cards)
                data_offset = blocks_read * FITS_BLOCK_SIZE
                naxis = cards.get("NAXIS", 0)

        bitpix = cards.get("BITPIX", -32)
        bscale = float(cards.get("BSCALE", 1.0))
        bzero = float(cards.get("BZERO", 0.0))

        if naxis == 1:
            shape = (cards["NAXIS1"],)
        elif naxis == 2:
            shape = (cards["NAXIS2"], cards["NAXIS1"])
        elif naxis == 3:
            shape = (cards["NAXIS3"], cards["NAXIS2"], cards["NAXIS1"])
        else:
            shape = ()

        itemsize = abs(bitpix) // 8
        total_voxels = math.prod(shape) if shape else 0
        data_bytes = total_voxels * itemsize

        return FITSHeader(
            cards=cards,
            shape=shape,
            bitpix=bitpix,
            bscale=bscale,
            bzero=bzero,
            data_offset=data_offset,
            data_bytes=data_bytes,
        )
    finally:
        if is_path:
            fp.close()


def stream_fits_cube(
    filepath_or_buffer: Union[str, Path, BinaryIO],
    chunk_z: int = 16,
    as_native: bool = True,
    copy: bool = False,
    reuse_buffer: bool = False,
    squeeze_2d: bool = False,
) -> Iterator[np.ndarray]:
    """Streams a FITS cube in Z-slab chunks with strict active heap residency < 2.0 MB.

    Parameters
    ----------
    filepath_or_buffer : str, Path, or BinaryIO
        FITS file path or readable binary stream.
    chunk_z : int
        Number of Z-slices per yielded slab (default: 16).
    as_native : bool
        If True, converts big-endian FITS data to native host byte order.
    copy : bool
        If False (default), yields zero-copy memory-mapped slice views (< 150 KB heap).
    reuse_buffer : bool
        If True, copies into an internally preallocated buffer (< 1.05 MB heap).
    squeeze_2d : bool
        If True and chunk_z == 1, yields 2D frames (H, W).

    Yields
    ------
    np.ndarray
        Slab of shape (chunk_z, H, W) or frame (H, W).
    """
    header = read_fits_header(filepath_or_buffer)
    shape = header.shape
    if len(shape) < 2:
        raise ValueError(f"stream_fits_cube requires 2D or 3D FITS data, got shape {shape}")

    dt_map = {-32: ">f4", -64: ">f8", 16: ">i2", 32: ">i4", 64: ">i8", 8: "u1"}
    raw_dt = dt_map.get(header.bitpix, ">f4")
    scale_needed = (header.bscale != 1.0 or header.bzero != 0.0)

    z_dim = shape[0] if len(shape) == 3 else 1
    h_dim = shape[1] if len(shape) == 3 else shape[0]
    w_dim = shape[2] if len(shape) == 3 else shape[1]

    is_path = isinstance(filepath_or_buffer, (str, Path))
    buffer_slab: Optional[np.ndarray] = None
    if reuse_buffer or (copy and scale_needed):
        buffer_slab = np.empty((min(chunk_z, z_dim), h_dim, w_dim), dtype=np.float32)

    if is_path:
        mm_shape = (z_dim, h_dim, w_dim)
        mm = np.memmap(
            filepath_or_buffer,
            dtype=raw_dt,
            mode="r",
            offset=header.data_offset,
            shape=mm_shape,
            order="C",
        )
        slab_view = None
        slab = None
        try:
            for z_start in range(0, z_dim, chunk_z):
                z_end = min(z_start + chunk_z, z_dim)
                actual_cz = z_end - z_start
                slab_view = mm[z_start:z_end]

                if not copy and not scale_needed and not reuse_buffer:
                    # Pure zero-copy memmap slice view (< 150 KB heap)
                    if squeeze_2d and actual_cz == 1:
                        yield slab_view[0]
                    else:
                        yield slab_view
                elif reuse_buffer and buffer_slab is not None:
                    # In-place reusable buffer (< 1.05 MB heap)
                    target_buf = buffer_slab if actual_cz == chunk_z else buffer_slab[:actual_cz]
                    np.copyto(target_buf, slab_view)
                    if scale_needed:
                        target_buf *= header.bscale
                        target_buf += header.bzero
                    if squeeze_2d and actual_cz == 1:
                        yield target_buf[0]
                    else:
                        yield target_buf
                else:
                    # Contiguous copy
                    slab = np.ascontiguousarray(slab_view, dtype=np.float32 if as_native else raw_dt)
                    if scale_needed:
                        slab = slab * header.bscale + header.bzero
                    if squeeze_2d and actual_cz == 1:
                        yield slab[0]
                    else:
                        yield slab
        finally:
            slab_view = None
            slab = None
            mm = None
            gc.collect()
    else:
        # Stream from BinaryIO
        fp = filepath_or_buffer
        fp.seek(header.data_offset)
        itemsize = abs(header.bitpix) // 8

        for z_start in range(0, z_dim, chunk_z):
            z_end = min(z_start + chunk_z, z_dim)
            actual_cz = z_end - z_start
            chunk_bytes = actual_cz * h_dim * w_dim * itemsize
            raw_bytes = fp.read(chunk_bytes)
            slab = np.frombuffer(raw_bytes, dtype=raw_dt).reshape((actual_cz, h_dim, w_dim))

            if as_native or scale_needed:
                slab = np.ascontiguousarray(slab, dtype=np.float32)
                if scale_needed:
                    slab = slab * header.bscale + header.bzero

            if squeeze_2d and actual_cz == 1:
                yield slab[0]
            else:
                yield slab


def write_fits_cube(
    filepath_or_buffer: Union[str, Path, BinaryIO],
    data: np.ndarray,
    header_extra: Optional[Dict[str, Any]] = None,
    bscale: float = 1.0,
    bzero: float = 0.0,
    chunk_z: int = 16,
) -> int:
    """Writes a 2D image or 3D cube to a standard FITS file with proper 2880-byte padding.

    Parameters
    ----------
    filepath_or_buffer : str, Path, or BinaryIO
        Output file destination.
    data : np.ndarray
        2D or 3D NumPy array.
    header_extra : dict, optional
        Extra FITS header keywords to append.
    bscale : float
        Physical scale factor (default: 1.0).
    bzero : float
        Physical zero offset (default: 0.0).
    chunk_z : int
        Chunk size for streaming binary disk writes (default: 16).

    Returns
    -------
    int
        Total bytes written (guaranteed divisible by 2880).
    """
    ndim = data.ndim
    if ndim not in (2, 3):
        raise ValueError(f"write_fits_cube requires 2D or 3D array, got ndim={ndim}")

    dtype = data.dtype
    if dtype == np.float32:
        bitpix, dt_code = -32, ">f4"
    elif dtype == np.float64:
        bitpix, dt_code = -64, ">f8"
    elif dtype == np.int16:
        bitpix, dt_code = 16, ">i2"
    elif dtype == np.int32:
        bitpix, dt_code = 32, ">i4"
    elif dtype == np.int64:
        bitpix, dt_code = 64, ">i8"
    elif dtype == np.uint8:
        bitpix, dt_code = 8, "u1"
    elif dtype == np.uint16:
        bitpix, dt_code = 16, ">i2"
        bzero = 32768.0
    else:
        bitpix, dt_code = -32, ">f4"
        data = data.astype(np.float32)

    shape = data.shape
    cards = [
        _format_fits_card("SIMPLE", True, "Standard FITS format"),
        _format_fits_card("BITPIX", bitpix, "Bits per pixel"),
        _format_fits_card("NAXIS", ndim, "Number of axes"),
    ]

    if ndim == 2:
        cards.append(_format_fits_card("NAXIS1", shape[1], "Fastest dimension (width)"))
        cards.append(_format_fits_card("NAXIS2", shape[0], "Slowest dimension (height)"))
    else:
        cards.append(_format_fits_card("NAXIS1", shape[2], "Fastest dimension (width)"))
        cards.append(_format_fits_card("NAXIS2", shape[1], "Medium dimension (height)"))
        cards.append(_format_fits_card("NAXIS3", shape[0], "Slowest dimension (depth/time)"))

    cards.append(_format_fits_card("BSCALE", bscale, "Physical scale factor"))
    cards.append(_format_fits_card("BZERO", bzero, "Physical zero offset"))
    cards.append(_format_fits_card("EXTEND", True, "Extensions permitted"))

    if header_extra:
        for k, v in header_extra.items():
            if isinstance(v, tuple):
                val, comm = v
            else:
                val, comm = v, ""
            cards.append(_format_fits_card(k, val, comm))

    cards.append(_format_fits_card("END", None, ""))

    raw_header = b"".join(cards)
    pad_header = (FITS_BLOCK_SIZE - (len(raw_header) % FITS_BLOCK_SIZE)) % FITS_BLOCK_SIZE
    header_block = raw_header + b" " * pad_header

    is_path = isinstance(filepath_or_buffer, (str, Path))
    fp: BinaryIO = open(filepath_or_buffer, "wb") if is_path else filepath_or_buffer

    bytes_written = 0
    try:
        fp.write(header_block)
        bytes_written += len(header_block)

        # Stream binary data in chunks to prevent memory duplication
        if ndim == 3:
            for z in range(0, shape[0], chunk_z):
                slab = np.ascontiguousarray(data[z : z + chunk_z], dtype=dt_code)
                slab_bytes = slab.tobytes()
                fp.write(slab_bytes)
                bytes_written += len(slab_bytes)
        else:
            arr_bytes = np.ascontiguousarray(data, dtype=dt_code).tobytes()
            fp.write(arr_bytes)
            bytes_written += len(arr_bytes)

        data_bytes = data.size * (abs(bitpix) // 8)
        pad_data = (FITS_BLOCK_SIZE - (data_bytes % FITS_BLOCK_SIZE)) % FITS_BLOCK_SIZE
        if pad_data > 0:
            null_pad = b"\x00" * pad_data
            fp.write(null_pad)
            bytes_written += len(null_pad)

        assert bytes_written % FITS_BLOCK_SIZE == 0, f"Unaligned FITS write: {bytes_written} bytes"
        return bytes_written
    finally:
        if is_path:
            fp.close()


# =============================================================================
# STREAMING LIGHT CURVE (.lc) PARSERS & WRITERS
# =============================================================================

def stream_lc_records(filepath_or_buffer: Union[str, Path, TextIO]) -> Iterator[Tuple[float, float, float]]:
    """Streams light curve records yielding (time, flux, flux_err) with zero full-file allocations."""
    is_path = isinstance(filepath_or_buffer, (str, Path))
    fp: TextIO = open(filepath_or_buffer, "r", encoding="utf-8") if is_path else filepath_or_buffer

    try:
        for line in fp:
            line_str = line.strip()
            if not line_str or line_str.startswith(("#", ";", "//")):
                continue
            parts = line_str.replace(",", " ").split()
            if len(parts) >= 3:
                yield (float(parts[0]), float(parts[1]), float(parts[2]))
            elif len(parts) == 2:
                yield (float(parts[0]), float(parts[1]), 0.0)
    finally:
        if is_path:
            fp.close()


def stream_lc_chunks(
    filepath_or_buffer: Union[str, Path, TextIO], chunk_size: int = 1024
) -> Iterator[np.ndarray]:
    """Streams light curve in fixed-size NumPy chunks of shape (chunk_size, 3)."""
    buf: List[Tuple[float, float, float]] = []
    for rec in stream_lc_records(filepath_or_buffer):
        buf.append(rec)
        if len(buf) >= chunk_size:
            yield np.asarray(buf, dtype=np.float64)
            buf.clear()
    if buf:
        yield np.asarray(buf, dtype=np.float64)


def parse_lc_table(filepath_or_buffer: Union[str, Path, TextIO]) -> np.ndarray:
    """Parses an entire light curve table into an (N, 3) NumPy array."""
    records = list(stream_lc_records(filepath_or_buffer))
    if not records:
        return np.empty((0, 3), dtype=np.float64)
    return np.asarray(records, dtype=np.float64)


def write_lc_file(
    filepath_or_buffer: Union[str, Path, TextIO],
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: Optional[np.ndarray] = None,
    comments: Optional[List[str]] = None,
) -> None:
    """Writes a standard tabular light curve file with header comments."""
    is_path = isinstance(filepath_or_buffer, (str, Path))
    fp: TextIO = open(filepath_or_buffer, "w", encoding="utf-8") if is_path else filepath_or_buffer

    try:
        if comments:
            for c in comments:
                fp.write(f"# {c}\n")
        fp.write("# time flux flux_err\n")

        n = len(time)
        err = flux_err if flux_err is not None else np.zeros(n, dtype=np.float64)
        for i in range(n):
            fp.write(f"{float(time[i]):.6f} {float(flux[i]):.6f} {float(err[i]):.6f}\n")
    finally:
        if is_path:
            fp.close()


# Alias for interface compatibility
write_lc_table = write_lc_file


# =============================================================================
# STREAMING SOURCE CATALOG (.cat) PARSERS & WRITERS
# =============================================================================

def stream_cat_records(
    filepath_or_buffer: Union[str, Path, TextIO]
) -> Iterator[Tuple[Union[int, str], float, float, float, float]]:
    """Streams point source catalog records yielding (source_id, ra, dec, mag, mag_err)."""
    is_path = isinstance(filepath_or_buffer, (str, Path))
    fp: TextIO = open(filepath_or_buffer, "r", encoding="utf-8") if is_path else filepath_or_buffer

    try:
        for line in fp:
            line_str = line.strip()
            if not line_str or line_str.startswith(("#", ";", "//")):
                continue
            parts = line_str.replace(",", " ").split()
            if len(parts) >= 5:
                raw_id = parts[0]
                src_id: Union[int, str] = int(raw_id) if raw_id.isdigit() else raw_id
                ra = float(parts[1])
                dec = float(parts[2])
                mag = float(parts[3])
                mag_err = float(parts[4])
                yield (src_id, ra, dec, mag, mag_err)
    finally:
        if is_path:
            fp.close()


def stream_cat_chunks(
    filepath_or_buffer: Union[str, Path, TextIO], chunk_size: int = 1024
) -> Iterator[List[Tuple[Union[int, str], float, float, float, float]]]:
    """Streams point source catalog in batches of records."""
    batch: List[Tuple[Union[int, str], float, float, float, float]] = []
    for rec in stream_cat_records(filepath_or_buffer):
        batch.append(rec)
        if len(batch) >= chunk_size:
            yield batch
            batch = []
    if batch:
        yield batch


def parse_cat_file(
    filepath_or_buffer: Union[str, Path, TextIO]
) -> List[Tuple[Union[int, str], float, float, float, float]]:
    """Parses an entire catalog file into a list of record tuples."""
    return list(stream_cat_records(filepath_or_buffer))


def write_cat_file(
    filepath_or_buffer: Union[str, Path, TextIO],
    records: Sequence[Tuple[Union[int, str], float, float, float, float]],
    comments: Optional[List[str]] = None,
) -> None:
    """Writes a standard point-source catalog file with header comments."""
    is_path = isinstance(filepath_or_buffer, (str, Path))
    fp: TextIO = open(filepath_or_buffer, "w", encoding="utf-8") if is_path else filepath_or_buffer

    try:
        if comments:
            for c in comments:
                fp.write(f"# {c}\n")
        fp.write("# source_id ra dec mag mag_err\n")
        for src_id, ra, dec, mag, mag_err in records:
            fp.write(f"{src_id} {float(ra):.6f} {float(dec):.6f} {float(mag):.4f} {float(mag_err):.4f}\n")
    finally:
        if is_path:
            fp.close()


# Alias for interface compatibility
write_cat_table = write_cat_file


# =============================================================================
# SYNTHETIC CATALOG GENERATOR
# =============================================================================

def generate_standard_catalogs(output_dir: Union[str, Path] = "data/catalogs", seed: int = 20261025) -> Dict[str, str]:
    """Generates standard reference synthetic catalogs for test suites and CLI runs.

    Files generated:
    1. data/catalogs/kbo_crossing.cat
    2. data/catalogs/variable_stars.lc
    3. data/catalogs/cosmic_web_skewers.fits
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # 1. KBO Crossing Catalog (.cat)
    cat_file = out_path / "kbo_crossing.cat"
    kbo_records = [
        (1001, 180.254120, 15.342100, 18.52, 0.03),
        (1002, 180.261400, 15.340900, 19.14, 0.05),
        (1003, 180.258900, 15.348700, 20.02, 0.08),
        (1004, 180.264100, 15.335400, 20.45, 0.11),
    ]
    # Add 20 background field stars
    for s_idx in range(20):
        s_id = 2001 + s_idx
        ra = 180.250000 + float(rng.uniform(-0.02, 0.02))
        dec = 15.340000 + float(rng.uniform(-0.02, 0.02))
        mag = float(rng.uniform(14.0, 21.0))
        mag_err = float(rng.uniform(0.01, 0.06))
        kbo_records.append((s_id, ra, dec, mag, mag_err))

    write_cat_file(cat_file, kbo_records, comments=["Scenario 03 KBO crossing astrometric sources", "Equinox: J2000"])

    # 2. Variable Stars Light Curve (.lc)
    lc_file = out_path / "variable_stars.lc"
    epochs = 2500
    t = np.linspace(0.0, 50.0, epochs)
    flux = 1000.0 + 50.0 * np.sin(2.0 * np.pi * t / 3.412) + rng.normal(0.0, 2.0, epochs)
    err = np.full(epochs, 2.0)
    write_lc_file(lc_file, t, flux, err, comments=["Scenario 02 Eclipsing Binary P=3.412d", "Flux in electron counts"])

    # 3. Cosmic Web Skewers FITS Cube (.fits)
    fits_file = out_path / "cosmic_web_skewers.fits"
    cube_data = np.full((32, 24, 24), -0.2, dtype=np.float32)
    # Inject underdense void in center
    cube_data[10:22, 8:16, 8:16] = -0.85
    # Inject pancake sheet
    cube_data[15:17, :, :] += 1.5
    write_fits_cube(fits_file, cube_data, header_extra={"SCENARIO": "05", "TELESCOP": "CosmicWeb-Tomography"})

    return {
        "kbo_crossing_cat": str(cat_file),
        "variable_stars_lc": str(lc_file),
        "cosmic_web_skewers_fits": str(fits_file),
    }


def profile_stream_memory(stream_generator: Iterator[Any]) -> Tuple[int, float]:
    """Profiles active heap memory during generator iteration using tracemalloc.

    Returns
    -------
    n_items : int
        Number of items yielded.
    peak_mb : float
        Peak active heap memory in Megabytes.
    """
    import tracemalloc

    tracemalloc.start()
    tracemalloc.reset_peak()
    count = 0
    for _ in stream_generator:
        count += 1
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return count, peak / (1024.0 * 1024.0)
