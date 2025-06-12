"""
Combine block-similarities to construct a sparse adjacency matrix.
Construct a small set and mapping from the adjacency matrix.
"""

import argparse
from pathlib import Path
import numpy as np
from typing import Union, Tuple, List
from tqdm import tqdm
from scipy.sparse import coo_matrix
import torch


def sparse_maximum(
    tensor1: torch.sparse_coo_tensor,
    tensor2: torch.sparse_coo_tensor,
) -> torch.sparse_coo_tensor:
    """
    Compute the element-wise maximum of two sparse tensors of the same
    shape and type.

    Args:
        tensor1 (torch.sparse_coo_tensor): First sparse tensor.
        tensor2 (torch.sparse_coo_tensor): Second sparse tensor.

    Returns:
        torch.sparse_coo_tensor: Sparse tensor containing the element-
        wise maximum.
    """
    if tensor1.size() != tensor2.size():
        raise ValueError("The tensors must have the same shape.")
    if tensor1.dtype != tensor2.dtype or tensor1.device != tensor2.device:
        raise ValueError("The tensors must have the same dtype and device.")

    # Coalesce both tensors to ensure valid sparse format
    tensor1 = tensor1.coalesce()
    tensor2 = tensor2.coalesce()

    # Combine all unique indices
    combined_indices = torch.cat([tensor1.indices(), tensor2.indices()], dim=1)
    unique_indices, inverse = torch.unique(
        combined_indices, dim=1, return_inverse=True
    )

    # Initialize values for both tensors at the unique indices
    values_tensor1 = torch.zeros(
        unique_indices.size(1), dtype=tensor1.dtype, device=tensor1.device
    )
    values_tensor2 = torch.zeros(
        unique_indices.size(1), dtype=tensor2.dtype, device=tensor2.device
    )

    # Fill values for tensor1
    inverse_tensor1 = inverse[: tensor1.indices().size(1)]
    values_tensor1.scatter_(
        dim=0,
        index=inverse_tensor1,
        src=tensor1.values(),
    )

    # Fill values for tensor2
    inverse_tensor2 = inverse[tensor1.indices().size(1) :]
    values_tensor2.scatter_(
        dim=0,
        index=inverse_tensor2,
        src=tensor2.values(),
    )

    # Compute the maximum at each index
    max_values = torch.maximum(values_tensor1, values_tensor2)

    # Create the final sparse tensor
    result_tensor = torch.sparse_coo_tensor(
        unique_indices, max_values, size=tensor1.size()
    ).coalesce()

    return result_tensor


def collect_sub_bank_overlaps(
    target_folder: Union[str, Path],
    threshold: float = 0.0,
    shape: Union[None, Tuple[int, int]] = None,
    to_bool: bool = False,
    symmetrize: bool = True,
    device: str = "cuda",
) -> torch.Tensor:
    target_folder = Path(target_folder)

    # List all files matching the pattern
    files = sorted(
        target_folder.glob("overlap_matrix_sub_bank_*.npz"),
        key=lambda x: int(x.stem.split("_")[-1]),
    )

    # Initialize lists to collect data
    all_rows, all_cols, all_data = [], [], []
    max_idx = -1

    for file in tqdm(files):
        block = np.load(file)

        # Extract indices and values
        i_inds = block["i_inds"]
        j_inds = block["j_inds"]
        values = block["overlaps_ij"]

        # Update maximum index across files
        max_idx = max(max_idx, i_inds.max(), j_inds.max())

        # Ensure dimensions match
        if len(i_inds) != values.shape[0] or len(j_inds) != values.shape[1]:
            raise ValueError(f"Mismatch in dimensions for file: {file}")

        # Generate row and column indices for all non-zero elements
        rows = np.repeat(i_inds, len(j_inds))
        cols = np.tile(j_inds, len(i_inds))
        values = values.ravel()

        # Filter by threshold
        above_threshold = values >= threshold
        rows = rows[above_threshold]
        cols = cols[above_threshold]
        values = values[above_threshold]

        # Append filtered data to global lists
        all_rows.append(rows)
        all_cols.append(cols)
        all_data.append(values)

    # Concatenate all the data across files
    final_rows = torch.tensor(np.concatenate(all_rows), device=device)
    final_cols = torch.tensor(np.concatenate(all_cols), device=device)
    final_data = torch.tensor(np.concatenate(all_data), device=device)

    # Define the shape of the sparse matrix
    shape = shape if shape else (max_idx + 1, max_idx + 1)

    # Create a sparse tensor
    sparse_matrix = torch.sparse_coo_tensor(
        indices=torch.stack([final_rows, final_cols]),
        values=final_data,
        size=shape,
        device=device,
    ).coalesce()

    # Apply filters or transformations
    if to_bool:
        sparse_matrix = sparse_matrix.to(dtype=torch.bool)
    if symmetrize:
        # Transpose the sparse matrix
        sparse_transpose = torch.sparse_coo_tensor(
            indices=sparse_matrix.indices()[[1, 0]],
            values=sparse_matrix.values(),
            size=sparse_matrix.size(),
            device=sparse_matrix.device,
        ).coalesce()

        sparse_matrix = sparse_maximum(sparse_matrix, sparse_transpose)

    return sparse_matrix


def find_representatives(
    adjacency_matrix: torch.sparse_coo_tensor,
) -> Tuple[List[int], List[List[int]]]:
    """
    Optimized function to find representative elements in a sparse adjacency matrix.

    Args:
        adjacency_matrix (torch.sparse_coo_tensor): A sparse adjacency matrix.

    Returns:
        Tuple[List[int], List[List[int]]]:
            - List of representative indices.
            - List of lists, where each sublist contains indices of elements covered by the representative.
    """
    # Ensure the adjacency matrix is coalesced
    adjacency_matrix = adjacency_matrix.coalesce()

    # Get the total number of nodes
    n = adjacency_matrix.size(0)
    reps = []  # List of representative indices
    mapping = []  # List of covered elements for each representative

    # Keep track of covered nodes
    covered = torch.zeros(n, dtype=torch.bool, device=adjacency_matrix.device)

    # Extract adjacency matrix indices and values
    row_indices, col_indices = adjacency_matrix.indices()
    adjacency_values = adjacency_matrix.values()

    # Precompute the row-wise sum for uncovered nodes
    row_sums = torch.zeros(
        n, dtype=torch.int32, device=adjacency_matrix.device
    )
    row_sums.index_add_(0, row_indices, adjacency_values.to(torch.int32))

    with tqdm(total=n, desc="Coverage Progress", unit="elements") as pbar:
        while not torch.all(covered):
            # Set coverage counts for covered nodes to -1
            row_sums[covered] = -1

            # Find the node with the highest coverage count
            best_element = torch.argmax(row_sums).item()

            # Add the best element to the representative set
            reps.append(best_element)

            # Find all elements similar to the best element
            similar_elements = col_indices[row_indices == best_element]
            mapping.append(similar_elements.cpu().tolist())

            # Mark all similar elements as covered
            previously_covered = covered.sum().item()
            covered[similar_elements] = True
            newly_covered = covered.sum().item() - previously_covered

            # Update the progress bar
            pbar.update(newly_covered)
            pbar.set_postfix({"Selected Set Size": len(reps)})

    return reps, mapping


def _find_representatives_scipy(
    adjacency_matrix: coo_matrix,
) -> Tuple[List[int], List[List[int]]]:
    """
    Legacy code. Use find_representatives instead.
    adjacency_matrix (coo_matrix): Sparse adjacency matrix.

    reps (list of int): List of selected elements (indices).
    mapping (list of lists of int): Mapping from elements in the small
    set to elements in the full set.
    """
    n = adjacency_matrix.shape[0]
    reps = []
    mapping = []
    covered = np.zeros(n, dtype=bool)

    with tqdm(total=n, desc="Coverage Progress", unit="elements") as pbar:
        while not np.all(covered):
            # Sum the adjacency matrix row-wise for uncovered nodes
            coverage_count = adjacency_matrix.dot(~covered)
            best_element = np.argmax(coverage_count)

            # Add best_element to the set
            reps.append(best_element)

            # Find all elements similar to best_element
            similar_elements = adjacency_matrix[best_element].nonzero()[1]
            mapping.append(list(similar_elements))

            # Mark all similar elements as covered
            previously_covered = np.sum(covered)
            covered[similar_elements] = True
            newly_covered = np.sum(covered) - previously_covered

            # Update progress bar
            pbar.update(newly_covered)
            pbar.set_postfix({"Selected Set Size": len(reps)})

    return reps, mapping


def parse_arguments():
    """Parser for arguments"""
    parser = argparse.ArgumentParser(
        description="Calculate overlap matrix for waveforms."
    )
    parser.add_argument(
        "--overlaps_folder",
        type=Path,
        required=True,
        help="Waveform overlap folder",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        required=False,
        default=None,
        help="Folder / filename to save overlap data."
        + "Defaults to overlaps_folder.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        required=False,
        default=0.85,
        help="Threshold for similarity between waveforms.",
    )

    kwargs = vars(parser.parse_args())
    if kwargs["output_path"] is None:
        kwargs["output_path"] = kwargs["overlaps_folder"]
    kwargs["output_path"] = Path(kwargs["output_path"])
    if kwargs["output_path"].is_dir():
        kwargs["output_path"] = kwargs["output_path"] / "small_set.json"
    return kwargs


def cluster(overlaps_folder, output_path, threshold):
    adj_mat = collect_sub_bank_overlaps(overlaps_folder, threshold)
    selected_set, mapping = find_representatives(adj_mat)
    with open(output_path, "w", encoding="utf-8") as fp:
        fp.write(f'{{"selected_set": {selected_set}, "mapping": {mapping}}}')


if __name__ == "__main__":
    kwargs = parse_arguments()
    cluster(**kwargs)
    print(f"Small set saved to {kwargs['output_path']}")
