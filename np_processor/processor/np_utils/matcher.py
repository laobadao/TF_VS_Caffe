from abc import ABCMeta
from abc import abstractmethod

from ..np_utils import ops
import numpy as np


class Match(object):
    """Class to store results from the matcher.

    This class is used to store the results from the matcher. It provides
    convenient methods to query the matching results.
    """

    def __init__(self, match_results, use_matmul_gather=False):

        if match_results.shape.ndims != 1:
            raise ValueError('match_results should have rank 1')
        if match_results.dtype != np.int32:
            raise ValueError('match_results should be an int32 or int64 scalar '
                             'tensor')
        self._match_results = match_results
        # TODO
        # if use_matmul_gather:
        #     self._gather_op = ops.matmul_gather_on_zeroth_axis

    @property
    def match_results(self):

        return self._match_results

    def matched_column_indices(self):

        return self._reshape_and_cast(np.where(np.greater(self._match_results, -1)))

    def matched_column_indicator(self):

        return np.greater_equal(self._match_results, 0)

    def num_matched_columns(self):
        return np.size(self.matched_column_indices())

    def unmatched_column_indices(self):

        return self._reshape_and_cast(np.where(np.equal(self._match_results, -1)))

    def unmatched_column_indicator(self):

        return np.equal(self._match_results, -1)

    def num_unmatched_columns(self):
        return np.size(self.unmatched_column_indices())

    def ignored_column_indices(self):

        return self._reshape_and_cast(np.where(self.ignored_column_indicator()))

    def ignored_column_indicator(self):

        return np.equal(self._match_results, -2)

    def num_ignored_columns(self):
        """Returns number (int32 scalar tensor) of matched columns."""
        return np.size(self.ignored_column_indices())

    def unmatched_or_ignored_column_indices(self):

        return self._reshape_and_cast(np.where(np.greater(0, self._match_results)))

    def matched_row_indices(self):

        return self._reshape_and_cast(
            np.take(self._match_results, self.matched_column_indices(),  axis=0))

    def _reshape_and_cast(self, t):
        return np.cast(np.reshape(t, [-1]), np.int32)

    def gather_based_on_match(self, input_tensor, unmatched_value,
                              ignored_value):

        input_tensor = np.concatenate([np.stack([ignored_value, unmatched_value]),
                                       input_tensor], axis=0)
        gather_indices = np.maximum(self.match_results + 2, 0)

        gathered_tensor = np.take(input_tensor, gather_indices, axis=0)

        return gathered_tensor


class Matcher(object):
    """Abstract base class for matcher.
    """
    __metaclass__ = ABCMeta

    def __init__(self, use_matmul_gather=False):
        """Constructs a Matcher.

        Args:
          use_matmul_gather: Force constructed match objects to use matrix
            multiplication based gather instead of standard np.gather.
            (Default: False).
        """
        self._use_matmul_gather = use_matmul_gather

    def match(self, similarity_matrix, scope=None, **params):

        return Match(self._match(similarity_matrix, **params),
                     self._use_matmul_gather)

    @abstractmethod
    def _match(self, similarity_matrix, **params):
        """Method to be overridden by implementations.

        Args:
          similarity_matrix: Float tensor of shape [N, M] with pairwise similarity
            where higher value means more similar.
          **params: Additional keyword arguments for specific implementations of
            the Matcher.

        Returns:
          match_results: Integer tensor of shape [M]: match_results[i]>=0 means
            that column i is matched to row match_results[i], match_results[i]=-1
            means that the column is not matched. match_results[i]=-2 means that
            the column is ignored (usually this happens when there is a very weak
            match which one neither wants as positive nor negative example).
        """
        pass
