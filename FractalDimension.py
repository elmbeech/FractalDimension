"""
title: FractalDimension.py

language: python3
date: 2019-09-16
license: GPLv3
author:
    original 3D implementation: Daniel Dondrop
    multidimensional modification: Elmar Bucher, codex
    multiprocessing modification: Elmar Bucher, codex

description:
    Capable implementation for box-counting in n-dimensional numpy arrays.
"""

# libraries
import numpy as np
import matplotlib.pyplot as plt
import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import shared_memory

# global variables
_WORKER_SHARED_MEMORY = None
_WORKER_VOXELS = None
_WORKER_ARRAY_SHAPE = None
_WORKER_N_OFFSETS = None


# functions
def _initialize_scale_worker(shared_name, voxels_shape, voxels_dtype, array_shape, n_offsets):
    """Attach worker globals to shared voxel coordinates."""
    global _WORKER_SHARED_MEMORY
    global _WORKER_VOXELS
    global _WORKER_ARRAY_SHAPE
    global _WORKER_N_OFFSETS

    _WORKER_SHARED_MEMORY = shared_memory.SharedMemory(name=shared_name)
    _WORKER_VOXELS = np.ndarray(
        voxels_shape,
        dtype=np.dtype(voxels_dtype),
        buffer=_WORKER_SHARED_MEMORY.buf,
    )
    _WORKER_ARRAY_SHAPE = array_shape
    _WORKER_N_OFFSETS = n_offsets


def _count_boxes_for_scale(scale, voxels, array_shape, n_offsets, verbose=False):
    """Return the minimum occupied-box count over offsets at one scale."""
    # handle input
    offsets = [0] if n_offsets == 0 else np.linspace(0, scale, n_offsets)
    if verbose:
        print('offsets:', offsets)
    # processing
    minimum_count = None
    for offset in offsets:
        bin_edges = [
            np.arange(-offset, size + scale + 1, scale)
            for size in array_shape
        ]
        histogram, _ = np.histogramdd(voxels, bins=bin_edges)
        count = int(np.count_nonzero(histogram))
        if minimum_count is None or count < minimum_count:
            minimum_count = count
    # output
    return minimum_count


def _count_boxes_in_worker(scale):
    minimum_count = _count_boxes_for_scale(
        scale=scale,
        voxels=_WORKER_VOXELS,
        array_shape=_WORKER_ARRAY_SHAPE,
        n_offsets=_WORKER_N_OFFSETS,
        verbose=False,
    )
    return minimum_count


def fractal_dimension(
        array,
        max_box_size=None,
        min_box_size=0,
        n_scales=20,
        n_offsets=0,
        n_workers=None,
        verbose=False,
    ):
    """Calculate the fractal dimension of an N-dimensional numpy array.

    Args:
        array (np.ndarray): An N-dimensional array with N >= 1. Each axis is
            treated as a geometric dimension, so axes should use compatible
            units and scales.
        max_box_size (int): The largest box size, given as the power of 2
            so that 2**max_box_size gives the sidelength of the largest box.
        min_box_size (int): The smallest box size, given as the power of 2
            so that 2**min_box_size gives the sidelength of the smallest box.
            Default value 1.
        n_scales (int): number of scales to measure over.
        n_offsets (int): number of offsets to search over to find the smallest
            set N(s) to cover  all voxels > 0.
        n_workers (int or None): Number of worker processes for calculating
            scales. Defaults to 1 (sequential). None uses up to one process per
            scale, limited by the available CPUs.
        verbose (bool): set to true to have the scale and offset values listed
            and see the analytical plot of a calculation.
    """
    # check array dimension
    array = np.asarray(array)
    if array.ndim < 1:
        raise ValueError("array must have at least one dimension")

    # determine the scales to measure on
    if max_box_size == None:
        # default max size is the largest power of 2 that fits in the smallest dimension of the array:
        max_box_size = int(np.floor(np.log2(np.min(array.shape))))
    scales = np.floor(np.logspace(max_box_size, min_box_size, num=n_scales, base=2))
    scales = np.unique(scales) # remove duplicates that could occur as a result of the floor
    if verbose:
        print('scales:', sorted(scales))

    # determ the number of cpus to use for calculation
    if isinstance(n_workers, (bool, np.bool_)) or (n_workers is not None and (not isinstance(n_workers, (int, np.integer)) or n_workers < 1)):
        raise ValueError("n_workers must be a positive integer or None")
    cpu_count = os.cpu_count() or 1
    requested_workers = cpu_count if n_workers is None else n_workers
    worker_count = min(requested_workers, cpu_count, len(scales))

    # get the locations of all non-zero pixels
    voxels = np.ascontiguousarray(np.column_stack(np.where(array > 0)))
    if voxels.shape[0] == 0:
        raise ValueError("array must contain at least one value (hoxel, voxel, pixel) greater than zero")

    # count the minimum amount of boxes touched
    if worker_count == 1:
        Ns = [
            _count_boxes_for_scale(
                scale=scale,
                voxels=voxels,
                array_shape=array.shape,
                n_offsets=n_offsets,
                verbose=verbose,
            )
            for scale in scales
        ]
    else:
        shared_memory_block = shared_memory.SharedMemory(
            create=True,
            size=voxels.nbytes,
        )
        try:
            shared_voxels = np.ndarray(
                voxels.shape,
                dtype=voxels.dtype,
                buffer=shared_memory_block.buf,
            )
            shared_voxels[:] = voxels
            with ProcessPoolExecutor(
                max_workers=worker_count,
                initializer=_initialize_scale_worker,
                initargs=(
                    shared_memory_block.name,
                    voxels.shape,
                    voxels.dtype.str,
                    array.shape,
                    n_offsets,
                ),
            ) as executor:
                Ns = list(executor.map(_count_boxes_in_worker, scales))
        finally:
            shared_memory_block.close()
            shared_memory_block.unlink()

    Ns = np.asarray(Ns)

    # only keep scales at which Ns changed
    scales  = np.array([np.min(scales[Ns == x]) for x in np.unique(Ns)])
    Ns = np.unique(Ns)
    Ns = Ns[Ns > 0]
    scales = scales[:len(Ns)]

    # perform fit
    coeffs = np.polyfit(np.log(1/scales), np.log(Ns), 1)
    slope = coeffs[0]

    # make plot
    if verbose:
        fig, ax = plt.subplots(figsize = (8,6))
        ax.scatter(np.log(1/scales), np.log(np.unique(Ns)), c="teal", label="Measured ratios")
        ax.set_ylabel(r"$\log N(\epsilon)$")
        ax.set_xlabel(r"$\log 1/ \epsilon$")
        fitted_y_vals = np.polyval(coeffs, np.log(1/scales))
        ax.plot(np.log(1/scales), fitted_y_vals, "k--", label=f"Fit: {np.round(coeffs[0],3)}X+{coeffs[1]}")
        ax.legend()
        fig.tight_layout()

    # output
    return slope
