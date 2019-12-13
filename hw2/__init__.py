from keras import backend

# Force channel ordering
backend.set_image_data_format("channels_last")
