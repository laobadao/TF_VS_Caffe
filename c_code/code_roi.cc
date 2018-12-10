    // Set all element of the output tensor to -inf.
    const int N = output.size();
    for (int i = 0; i < N; i++)
    {
      output(i) = -FLT_MAX;
      argmax(i) = -1;
    }

    // For each ROI R = [batch_index x1 y1 x2 y2]: max pool over R
    int index_roi = 0;
    int index_output = 0;
    for (int n = 0; n < num_rois; ++n)
    {
      int roi_batch_ind = bottom_rois_flat(index_roi + 0);
      int roi_start_w = round(bottom_rois_flat(index_roi + 1) * spatial_scale_);
      int roi_start_h = round(bottom_rois_flat(index_roi + 2) * spatial_scale_);
      int roi_end_w = round(bottom_rois_flat(index_roi + 3) * spatial_scale_);
      int roi_end_h = round(bottom_rois_flat(index_roi + 4) * spatial_scale_);
      CHECK_GE(roi_batch_ind, 0);
      CHECK_LT(roi_batch_ind, batch_size);

      int roi_height = std::max(roi_end_h - roi_start_h + 1, 1);
      int roi_width = std::max(roi_end_w - roi_start_w + 1, 1);
      const T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height_);
      const T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width_);

      int index_data = roi_batch_ind * data_height * data_width * num_channels;

      for (int ph = 0; ph < pooled_height_; ++ph)
      {
        for (int pw = 0; pw < pooled_width_; ++pw)
        {
          // Compute pooling region for this output unit:
          //  start (included) = floor(ph * roi_height / pooled_height_)
          //  end (excluded) = ceil((ph + 1) * roi_height / pooled_height_)
          int hstart = static_cast<int>(floor(static_cast<T>(ph)
                                              * bin_size_h));
          int wstart = static_cast<int>(floor(static_cast<T>(pw)
                                              * bin_size_w));
          int hend = static_cast<int>(ceil(static_cast<T>(ph + 1)
                                           * bin_size_h));
          int wend = static_cast<int>(ceil(static_cast<T>(pw + 1)
                                           * bin_size_w));

          hstart = std::min(std::max(hstart + roi_start_h, 0), data_height);
          hend = std::min(std::max(hend + roi_start_h, 0), data_height);
          wstart = std::min(std::max(wstart + roi_start_w, 0), data_width);
          wend = std::min(std::max(wend + roi_start_w, 0), data_width);

          bool is_empty = (hend <= hstart) || (wend <= wstart);

          const int pool_index = index_output + (ph * pooled_width_ + pw) * num_channels;
          if (is_empty)
          {
            for (int c = 0; c < num_channels; ++c)
            {
              output(pool_index + c) = 0;
              argmax(pool_index + c) = -1;
            }
          }

          for (int h = hstart; h < hend; ++h)
          {
            for (int w = wstart; w < wend; ++w)
            {
              for (int c = 0; c < num_channels; ++c)
              {
                const int index = (h * data_width + w) * num_channels + c;
                if (bottom_data_flat(index_data + index) > output(pool_index + c))
                {
                  output(pool_index + c) = bottom_data_flat(index_data + index);
                  argmax(pool_index + c) = index;
                }
              }
            }
          }
        }
      }
      // Increment ROI index
      index_roi += bottom_rois.dim_size(1);
      index_output += pooled_height_ * pooled_width_ * num_channels;
    }

