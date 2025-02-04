# Units used in the image metadata for physical sizes
image_units = {
    "nm": 1e9,
    "μm": 1e6,
    "mm": 1e3,
    "cm": 1e2,
    "dm": 10
}

# To filter base image pyramids when finding segmentation mask images (kaggle-1, kaggle-2)
base_image_dirs = ['lab_processed', 'processed_microscopy' , 'processedMicroscopy']