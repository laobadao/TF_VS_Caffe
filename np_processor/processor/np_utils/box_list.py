"""Bounding Box List definition.

BoxList represents a list of bounding boxes as tensorflow
tensors, where each bounding box is represented as a row of 4 numbers,
[y_min, x_min, y_max, x_max].  It is assumed that all bounding boxes
within a given list correspond to a single image.  See also
box_list_ops.py for common box related operations (such as area, iou, etc).

Optionally, users can add additional related fields (such as weights).
We assume the following things to be true about fields:
* they correspond to boxes in the box_list along the 0th dimension
* they have inferrable rank at graph construction time
* all dimensions except for possibly the 0th can be inferred
  (i.e., not None) at graph construction time.

Some other notes:
  * Following tensorflow conventions, we use height, width ordering,
  and correspondingly, y,x (or ymin, xmin, ymax, xmax) ordering
  * Tensors are always provided as (flat) [N, 4] tensors.
"""

import numpy as np


class BoxList(object):
    """Box collection."""

    def __init__(self, boxes, name=None):
        """Constructs box collection.

        Args:
          boxes: a tensor of shape [N, 4] representing box corners

        Raises:
          ValueError: if invalid dimensions for bbox data or if bbox data is not in
              float32 format.
        """
        if boxes.ndim != 2 or boxes.shape[-1] != 4:
            raise ValueError('Invalid dimensions for box data.')

        self.data = {'boxes': boxes}

    def num_boxes(self):
        """Returns number of boxes held in collection.

        Returns:
          a tensor representing the number of boxes held in the collection.
        """
        return self.data['boxes'].shape[0]

    def num_boxes_static(self):
        """Returns number of boxes held in collection.

        This number is inferred at graph construction time rather than run-time.

        Returns:
          Number of boxes held in collection (integer) or None if this is not
            inferrable at graph construction time.
        """
        return self.data['boxes'].shape[0]

    def get_all_fields(self):
        """Returns all fields."""
        return self.data.keys()

    def get_extra_fields(self, name=None):
        """Returns all non-box fields (i.e., everything not named 'boxes')."""
        return [k for k in self.data.keys() if k != 'boxes']

    def add_field(self, field, field_data):
        """Add field to box list.

        This method can be used to add related box data such as
        weights/labels, etc.

        Args:
          field: a string key to access the data via `get`
          field_data: a tensor containing the data to store in the BoxList
        """
        self.data[field] = field_data

    def has_field(self, field):
        return field in self.data

    def get(self):
        """Convenience function for accessing box coordinates.

        Returns:
          a tensor with shape [N, 4] representing box coordinates.
        """
        return self.get_field('boxes')

    def set(self, boxes):
        """Convenience function for setting box coordinates.

        Args:
          boxes: a tensor of shape [N, 4] representing box corners

        Raises:
          ValueError: if invalid dimensions for bbox data
        """
        if boxes.ndim != 2 or boxes.shape[-1] != 4:
            raise ValueError('Invalid dimensions for box data.')
        self.data['boxes'] = boxes

    def get_field(self, field):
        """Accesses a box collection and associated fields.

        This function returns specified field with object; if no field is specified,
        it returns the box coordinates.

        Args:
          field: this optional string parameter can be used to specify
            a related field to be accessed.

        Returns:
          a tensor representing the box collection or an associated field.

        Raises:
          ValueError: if invalid field
        """
        if not self.has_field(field):
            raise ValueError('field ' + str(field) + ' does not exist')
        return self.data[field]

    def set_field(self, field, value):
        """Sets the value of a field.

        Updates the field of a box_list with a given value.

        Args:
          field: (string) name of the field to set value.
          value: the value to assign to the field.

        Raises:
          ValueError: if the box_list does not have specified field.
        """
        if not self.has_field(field):
            raise ValueError('field %s does not exist' % field)
        self.data[field] = value

    def get_center_coordinates_and_sizes(self):
        """Computes the center coordinates, height and width of the boxes.

        Args:
          scope: name scope of the function.

        Returns:
          a list of 4 1-D tensors [ycenter, xcenter, height, width].
        """
        box_corners = self.get()
        ymin, xmin, ymax, xmax = np.stack(np.transpose(box_corners))
        width = xmax - xmin
        height = ymax - ymin
        ycenter = ymin + height / 2.
        xcenter = xmin + width / 2.
        return [ycenter, xcenter, height, width]
