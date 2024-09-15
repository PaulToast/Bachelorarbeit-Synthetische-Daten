# Use object mask to create a square crop around object
def crop_object(image, mask):
    # Convert mask to binary
    mask = np.where(mask, 255, 0).astype(np.uint8)
    
    # Dilate mask with maximum filter
    mask = Image.fromarray(maximum_filter(mask, size=32))
    
    # Get bounding box of object mask
    mask_box = mask.getbbox()

    # Make mask_box square without offsetting the center
    mask_box_width = mask_box[2] - mask_box[0]
    mask_box_height = mask_box[3] - mask_box[1]
    mask_box_size = max(mask_box_width, mask_box_height)
    mask_box_center_x = (mask_box[2] + mask_box[0]) // 2
    mask_box_center_y = (mask_box[3] + mask_box[1]) // 2
    mask_box = (
        mask_box_center_x - mask_box_size // 2,
        mask_box_center_y - mask_box_size // 2,
        mask_box_center_x + mask_box_size // 2,
        mask_box_center_y + mask_box_size // 2
    )
    
    return image.crop(mask_box), np.array(mask.crop(mask_box))