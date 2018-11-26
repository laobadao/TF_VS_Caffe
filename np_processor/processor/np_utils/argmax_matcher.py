
import numpy as np

from ..np_utils import matcher
from ..np_utils import shape_utils


class ArgMaxMatcher(matcher.Matcher):
  """Matcher based on highest value.

  This class computes matches from a similarity matrix. Each column is matched
  to a single row.

  To support object detection target assignment this class enables setting both
  matched_threshold (upper threshold) and unmatched_threshold (lower thresholds)
  defining three categories of similarity which define whether examples are
  positive, negative, or ignored:
  (1) similarity >= matched_threshold: Highest similarity. Matched/Positive!
  (2) matched_threshold > similarity >= unmatched_threshold: Medium similarity.
          Depending on negatives_lower_than_unmatched, this is either
          Unmatched/Negative OR Ignore.
  (3) unmatched_threshold > similarity: Lowest similarity. Depending on flag
          negatives_lower_than_unmatched, either Unmatched/Negative OR Ignore.
  For ignored matches this class sets the values in the Match object to -2.
  """

  def __init__(self,
               matched_threshold,
               unmatched_threshold=None,
               negatives_lower_than_unmatched=True,
               force_match_for_each_row=False,
               use_matmul_gather=False):

    super(ArgMaxMatcher, self).__init__(use_matmul_gather=use_matmul_gather)
    if (matched_threshold is None) and (unmatched_threshold is not None):
      raise ValueError('Need to also define matched_threshold when'
                       'unmatched_threshold is defined')
    self._matched_threshold = matched_threshold
    if unmatched_threshold is None:
      self._unmatched_threshold = matched_threshold
    else:
      if unmatched_threshold > matched_threshold:
        raise ValueError('unmatched_threshold needs to be smaller or equal'
                         'to matched_threshold')
      self._unmatched_threshold = unmatched_threshold
    if not negatives_lower_than_unmatched:
      if self._unmatched_threshold == self._matched_threshold:
        raise ValueError('When negatives are in between matched and '
                         'unmatched thresholds, these cannot be of equal '
                         'value. matched: %s, unmatched: %s',
                         self._matched_threshold, self._unmatched_threshold)
    self._force_match_for_each_row = force_match_for_each_row
    self._negatives_lower_than_unmatched = negatives_lower_than_unmatched

  def _match(self, similarity_matrix):
    """Tries to match each column of the similarity matrix to a row.

    Args:
      similarity_matrix: tensor of shape [N, M] representing any similarity
        metric.

    Returns:
      Match object with corresponding matches for each of M columns.
    """

    def _match_when_rows_are_empty():
      """Performs matching when the rows of similarity matrix are empty.

      When the rows are empty, all detections are false positives. So we return
      a tensor of -1's to indicate that the columns do not match to any rows.

      Returns:
        matches:  int32 tensor indicating the row each column matches to.
      """
      similarity_matrix_shape = shape_utils.combined_static_and_dynamic_shape(
          similarity_matrix)
      return -1 * np.ones([similarity_matrix_shape[1]], dtype=np.int32)

    def _match_when_rows_are_non_empty():
      """Performs matching when the rows of similarity matrix are non empty.

      Returns:
        matches:  int32 tensor indicating the row each column matches to.
      """
      # Matches for each column
      matches = int(np.argmax(similarity_matrix, 0))

      # Deal with matched and unmatched threshold
      if self._matched_threshold is not None:
        # Get logical indices of ignored and unmatched columns as np.int64
        matched_vals = np.max(similarity_matrix, 0)
        below_unmatched_threshold = np.greater(self._unmatched_threshold,
                                               matched_vals)
        between_thresholds = np.logical_and(
            np.greater_equal(matched_vals, self._unmatched_threshold),
            np.greater(self._matched_threshold, matched_vals))

        if self._negatives_lower_than_unmatched:
          matches = self._set_values_using_indicator(matches,
                                                     below_unmatched_threshold,
                                                     -1)
          matches = self._set_values_using_indicator(matches,
                                                     between_thresholds,
                                                     -2)
        else:
          matches = self._set_values_using_indicator(matches,
                                                     below_unmatched_threshold,
                                                     -2)
          matches = self._set_values_using_indicator(matches,
                                                     between_thresholds,
                                                     -1)

      if self._force_match_for_each_row:
        similarity_matrix_shape = shape_utils.combined_static_and_dynamic_shape(
            similarity_matrix)
        force_match_column_ids = np.argmax(similarity_matrix, 1)

        force_match_column_indicators = np.one_hot(
            force_match_column_ids, depth=similarity_matrix_shape[1])

        force_match_row_ids = np.argmax(force_match_column_indicators, 0)

        force_match_column_mask = np.cast(
            np.max(force_match_column_indicators, 0), np.bool)
        final_matches = np.where(force_match_column_mask,
                                 force_match_row_ids, matches)
        return final_matches
      else:
        return matches

    if similarity_matrix.shape.is_fully_defined():

      if similarity_matrix.shape[0].value == 0:
        return _match_when_rows_are_empty()
      else:
        return _match_when_rows_are_non_empty()
    else:

      return np.cond(
          np.greater(np.shape(similarity_matrix)[0], 0),
          _match_when_rows_are_non_empty, _match_when_rows_are_empty)

  def _set_values_using_indicator(self, x, indicator, val):
    """Set the indicated fields of x to val.

    Args:
      x: tensor.
      indicator: boolean with same shape as x.
      val: scalar with value to set.

    Returns:
      modified tensor.
    """
    indicator = np.cast(indicator, x.dtype)
    return np.add(np.multiply(x, 1 - indicator), val * indicator)
