import numpy as np
import cv2

# Single-channel thresholding:
def get_channel_mask(
    image,
    threshold
):
    """ Generate mask based on channel component values

    Args:
        image (numpy 2-d array): selected channel component of input image
        thresholds (2-element tuple): min & max values for thresholding

    Returns:
        generated mask
    """
    # Image dimensions:
    H, W = image.shape

    # Generate mask:
    mask = np.zeros((H, W), dtype=np.uint8)

    channel_min, channel_max = threshold

    mask[
        (channel_min <= image) & (image <= channel_max)
    ] = 1

    return mask

# Color thresholding:
def color_thresh(
    img,
    color_space,
    thresholds,
    morphology_kernel_size,
    morphology_iters = 0
):
    """ Generate mask based on color thresholding

    Args:
        img (numpy ndarray): input image
        thresholds (list of list of int): min & max values for each channel
    """
    if color_space == "RGB":
        conversion = cv2.COLOR_BGR2RGB
    else:
        conversion = cv2.COLOR_BGR2YUV

    # Set up morphological filtering:
    morphology_kernel = np.ones(
        (morphology_kernel_size,morphology_kernel_size),
        np.uint8
    )

    # Convert to HSV:
    converted = cv2.cvtColor(
        cv2.cvtColor(img, cv2.COLOR_RGB2BGR), conversion
    )

    # Get mask for each channel component:
    masks = [
        get_channel_mask(channel_component, threshold)
        for (channel_component, threshold) in zip(cv2.split(converted), thresholds)
    ]

    # Generate final mask:
    mask = masks[0] & masks[1] & masks[2]

    # Morphological filtering:
    if morphology_iters > 0:
        for _ in range(morphology_iters):
            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_CLOSE,
                morphology_kernel
            )
            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_OPEN,
                morphology_kernel
            )
    elif morphology_iters < 0:
        for _ in range(-morphology_iters):
            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_OPEN,
                morphology_kernel
            )
            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_CLOSE,
                morphology_kernel
            )

    return mask
# Define a function to convert from image coords to rover coords
def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the
    # center bottom of the image.
    x_pixel = -(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[1]/2 ).astype(np.float)
    return x_pixel, y_pixel


# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle)
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles

# Define a function to map rover space pixels to world space
def rotate_pix(xpix, ypix, yaw):
    # Convert yaw to radians
    yaw_rad = yaw * np.pi / 180
    xpix_rotated = (xpix * np.cos(yaw_rad)) - (ypix * np.sin(yaw_rad))

    ypix_rotated = (xpix * np.sin(yaw_rad)) + (ypix * np.cos(yaw_rad))
    # Return the result
    return xpix_rotated, ypix_rotated

def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale):
    # Apply a scaling and a translation
    xpix_translated = (xpix_rot / scale) + xpos
    ypix_translated = (ypix_rot / scale) + ypos
    # Return the result
    return xpix_translated, ypix_translated


# Define a function to apply rotation and translation (and clipping)
# Once you define the two functions above this function should work
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world

source = np.array(
    [
        [  14.,  140.],
        [ 301.,  140.],
        [ 200.,   96.],
        [ 118.,   96.]
    ], dtype = np.float32
)
destination = np.array(
    [
        [ 155.,  154.],
        [ 165.,  154.],
        [ 165.,  144.],
        [ 155.,  144.]
    ], dtype = np.float32
)
# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image

    return warped


# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover):
    # Perform perception steps to update Rover()
    # TODO:
    # NOTE: camera image is coming to you in Rover.img
    # 1) Define source and destination points for perspective transform
    # 2) Apply perspective transform
    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    # 4) Update Rover.vision_image (this will be displayed on left side of screen)
        # Example: Rover.vision_image[:,:,0] = obstacle color-thresholded binary image
        #          Rover.vision_image[:,:,1] = rock_sample color-thresholded binary image
        #          Rover.vision_image[:,:,2] = navigable terrain color-thresholded binary image

    # 5) Convert map image pixel values to rover-centric coords
    # 6) Convert rover-centric pixel values to world coordinates
    # 7) Update Rover worldmap (to be displayed on right side of screen)
        # Example: Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1
        #          Rover.worldmap[rock_y_world, rock_x_world, 1] += 1
        #          Rover.worldmap[navigable_y_world, navigable_x_world, 2] += 1

    # 8) Convert rover-centric pixel positions to polar coordinates
    # Update Rover pixel distances and angles
        # Rover.nav_dists = rover_centric_pixel_distances
        # Rover.nav_angles = rover_centric_angles

    # 0) Rover state:
    (x_trans, y_trans) = Rover.pos
    yaw = Rover.yaw

    # 1) Apply color threshold to identify navigable terrain/obstacles/rock samples
    ground = color_thresh(
        Rover.img,
        "RGB", [[160, 255],[160, 255],[160, 255]],
        5, -1
    )
    obstacle = (ground == 0).astype(np.uint8)
    rock = color_thresh(
        Rover.img,
        "YUV", [[  0, 255],[128, 255],[ 20,  80]],
        7, 1
    )

    # 2) Apply perspective transform
    ground = perspect_transform(ground, source, destination)
    obstacle = perspect_transform(obstacle, source, destination)
    rock = perspect_transform(rock, source, destination)

    # 4) Convert thresholded image pixel values to rover-centric coords & world coords
    coords = {
        "ground": {},
        "obstacle": {},
        "rock": {}
    }
    for obj_name, obj_in_pixel in zip(
        ("ground", "obstacle", "rock"),
        (ground, obstacle, rock),
    ):
        # To rover centric:
        coords[obj_name]["rover"] = rover_coords(obj_in_pixel)
        coords[obj_name]["polar"] = to_polar_coords(*coords[obj_name]["rover"])
        # To world centric:
        coords[obj_name]["world"] = pix_to_world(
            coords[obj_name]["rover"][0], coords[obj_name]["rover"][1],
            x_trans, y_trans, yaw,
            world_size = 200,
            scale = 10
        )

    # Bird eye view:
    Rover.vision_image[:,:,0] = 255 * obstacle
    if rock.any():
        Rover.vision_image[:,:,1] = 255 * rock
    else:
        Rover.vision_image[:,:,1] = 0
    Rover.vision_image[:,:,2] = 255 * ground

    # World map inpainting:
    (obstacle_x_world, obstacle_y_world) = coords["obstacle"]["world"]
    Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] += 10
    if rock.any():
        (rock_x_world, rock_y_world) = coords["rock"]["world"]
        Rover.worldmap[rock_y_world, rock_x_world, 1] = 255
    (ground_x_world, ground_y_world) = coords["ground"]["world"]
    Rover.worldmap[ground_y_world, ground_x_world, 2] += 10

    # Update navigation angle:
    Rover.nav_angles = coords["ground"]["polar"][1]

    return Rover
