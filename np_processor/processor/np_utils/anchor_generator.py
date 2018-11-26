"""Base anchor generator.

The job of the anchor generator is to create (or load) a collection
of bounding boxes to be used as anchors.

Generated anchors are assumed to match some convolutional grid or list of grid
shapes.  For example, we might want to generate anchors matching an 8x8
feature map and a 4x4 feature map.  If we place 3 anchors per grid location
on the first feature map and 6 anchors per grid location on the second feature
map, then 3*8*8 + 6*4*4 = 288 anchors are generated in total.

To support fully convolutional settings, feature map shapes are passed
dynamically at generation time.  The number of anchors to place at each location
is static --- implementations of AnchorGenerator must always be able return
the number of anchors that it uses per location for each feature map.
"""
from abc import ABCMeta
from abc import abstractmethod


class AnchorGenerator(object):
    """Abstract base class for anchor generators."""
    __metaclass__ = ABCMeta

    @abstractmethod
    def name_scope(self):
        """Name scope.

        Must be defined by implementations.

        Returns:
          a string representing the name scope of the anchor generation operation.
        """
        pass

    @property
    def check_num_anchors(self):
        """Whether to dynamically check the number of anchors generated.

        Can be overridden by implementations that would like to disable this
        behavior.

        Returns:
          a boolean controlling whether the Generate function should dynamically
          check the number of anchors generated against the mathematically
          expected number of anchors.
        """
        return True

    @abstractmethod
    def num_anchors_per_location(self):
        """Returns the number of anchors per spatial location.

        Returns:
          a list of integers, one for each expected feature map to be passed to
          the `generate` function.
        """
        pass

    def generate(self, feature_map_shape_list, **params):
        """Generates a collection of bounding boxes to be used as anchors.

        TODO(rathodv): remove **params from argument list and make stride and
          offsets (for multiple_grid_anchor_generator) constructor arguments.

        Args:
          feature_map_shape_list: list of (height, width) pairs in the format
            [(height_0, width_0), (height_1, width_1), ...] that the generated
            anchors must align with.  Pairs can be provided as 1-dimensional
            integer tensors of length 2 or simply as tuples of integers.
          **params: parameters for anchor generation op

        Returns:
          boxes_list: a list of BoxLists each holding anchor boxes corresponding to
            the input feature map shapes.

        Raises:
          ValueError: if the number of feature map shapes does not match the length
            of NumAnchorsPerLocation.
        """
        if self.check_num_anchors and (
                len(feature_map_shape_list) != len(self.num_anchors_per_location())):
            raise ValueError('Number of feature maps is expected to equal the length '
                             'of `num_anchors_per_location`.')
        anchors_list = self._generate(feature_map_shape_list, **params)
        return anchors_list

    @abstractmethod
    def _generate(self, feature_map_shape_list, **params):
        """To be overridden by implementations.

        Args:
          feature_map_shape_list: list of (height, width) pairs in the format
            [(height_0, width_0), (height_1, width_1), ...] that the generated
            anchors must align with.
          **params: parameters for anchor generation op

        Returns:
          boxes_list: a list of BoxList, each holding a collection of N anchor
            boxes.
        """
        pass
