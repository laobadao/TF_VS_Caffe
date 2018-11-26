import numpy as np

from tensorflow.contrib.image.python.ops import image_ops
from ..np_utils import matcher


class GreedyBipartiteMatcher(matcher.Matcher):
  """Wraps a Tensorflow greedy bipartite matcher."""

  def __init__(self, use_matmul_gather=False):
    """Constructs a Matcher.

    Args:
      use_matmul_gather: Force constructed match objects to use matrix
        multiplication based gather instead of standard tf.gather.
        (Default: False).
    """
    super(GreedyBipartiteMatcher, self).__init__(
        use_matmul_gather=use_matmul_gather)

  def _match(self, similarity_matrix, num_valid_rows=-1):
    """Bipartite matches a collection rows and columns. A greedy bi-partite.

    TODO(rathodv): Add num_valid_columns options to match only that many columns
    with all the rows.

    Args:
      similarity_matrix: Float tensor of shape [N, M] with pairwise similarity
        where higher values mean more similar.
      num_valid_rows: A scalar or a 1-D tensor with one element describing the
        number of valid rows of similarity_matrix to consider for the bipartite
        matching. If set to be negative, then all rows from similarity_matrix
        are used.

    Returns:
      match_results: int32 tensor of shape [M] with match_results[i]=-1
        meaning that column i is not matched and otherwise that it is matched to
        row match_results[i].
    """
    # Convert similarity matrix to distance matrix as tf.image.bipartite tries
    # to find minimum distance matches.
    distance_matrix = -1 * similarity_matrix
    _, match_results = image_ops.bipartite_match(
        distance_matrix, num_valid_rows)
    match_results = np.reshape(match_results, [-1])
    match_results = np.cast(match_results, np.int32)
    return match_results
