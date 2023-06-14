import cv2
import numpy as np


def stackFrames(stack_shape: tuple, frames: list | np.ndarray, dtype= np.uint8):
    """
    Stack multiple frames into a single canvas.

    This function takes a list of frames and combines them into a single canvas using numpy arrays. The frames can be either grayscale or colorful images.

    :param stack_shape: A tuple specifying the desired shape of the stacked canvas (number of rows, number of columns).
    :param frames: A list of numpy arrays, where each element represents a frame to be stacked. Each frame should be a numpy ndarray.
    :param dtype: The data type for the canvas. Default is np.uint8

    :return: The stacked canvas as a numpy ndarray.

    :raises ValueError: If the number of frames is too large for the given stack shape.
    :raises TypeError: If any frame in the list is not a numpy array.

    Note: Each element in the frames list should be a numpy array (or ndarray). Grayscale frames should have shape (height, width), while colorful frames should have shape (height, width, channels).
    """

    for frame in frames:
        if not isinstance(frame, np.ndarray):
            raise ValueError(
                f"The element at index {frames.index(frame)} is not an np.ndarray")

    def checkAllGrayscale(frames):
        # Check if all frames have grayscale shape (height, width)
        checks = []
        for frame in frames:
            if len(frame.shape) == 2:
                checks.append(1)

        if len(checks) == len(frames):
            return True

        # If not all frames have the required number of channels
        return False

    def checkAllColorful(frames):
        # Check if all frames have colorful shape (height, width, channels)
        checks = []

        for frame in frames:
            if len(frame.shape) > 2:
                checks.append(frame)

        # Check if all frames have the same number of channels
        if len(checks) == len(frames):
            channels_count = frames[0].shape[2]
            if any(frame.shape[2] == channels_count for frame in frames):
                return {"success": True, "channels_count": channels_count}
        return {"success": False}

    def equalizeFrames(frames):
        # Equalize the frames to have the same number of channels
        equalizedFrames = []
        for frame in frames:
            if len(frame.shape) > 2:
                channels_count = frame.shape[2]
                break

        for frame in frames:
            if len(frame.shape) <= 2:
                frame = np.expand_dims(frame, axis=-1)
                frame = np.repeat(frame, channels_count, -1)
            equalizedFrames.append(frame)

        return {"frames": equalizedFrames, "channels_count": channels_count}

    stack_max_width_count, stack_max_height_count = stack_shape

    canvas = None
    channels = None
    max_frames = stack_max_height_count * stack_max_width_count
    frame_height, frame_width = frames[0].shape[:2]

    areColorful = checkAllColorful(frames)
    if checkAllGrayscale(frames):
        # Create a grayscale canvas
        canvas = np.zeros((stack_max_height_count * frame_height,
                           stack_max_width_count * frame_width), dtype=dtype)
    elif areColorful['success']:
        # Create a colorful canvas with the same number of channels as the frames
        channels = areColorful['channels_count']
        canvas = np.zeros((stack_max_height_count * frame_height,
                          stack_max_width_count * frame_width, channels), dtype=dtype)
    else:
        # Equalize frames to have the same number of channels
        equalizedFrames = equalizeFrames(frames)
        frames = equalizedFrames['frames']
        channels = equalizedFrames['channels_count']
        canvas = np.zeros((stack_max_height_count * frame_height,
                          stack_max_width_count * frame_width, channels), dtype=dtype)

    print(f'Frame: {frame_width}x{frame_height}, channels={channels}')
    print(f'Stack shape: ({stack_max_width_count},{stack_max_height_count})')

    # Check if the number of frames is within the stack's capacity
    if len(frames) > max_frames:
        raise ValueError(
            f"\nThe number of frames is too large for the current stack. Maximum frames for the stack {stack_shape} is {max_frames}, but got: {len(frames)}")
    else:
        for height_index in range(stack_max_height_count):
            for width_index in range(stack_max_width_count):
                start_height = height_index * frame_height
                end_height = (height_index + 1) * frame_height
                start_width = width_index * frame_width
                end_width = (width_index + 1) * frame_width
                try:
                    canvas[start_height:end_height,
                           start_width:end_width] = frames[stack_shape[0] * height_index + width_index]
                except IndexError:
                    pass

    print(f'Output Image: {canvas.shape[:2][::-1]+(canvas.shape[2],)}\n')
    return canvas


# Example usage:

# Load and resize the images
image = cv2.imread("images/giyu.png", cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
print(f"\nInput Image: {image.shape[1]}x{image.shape[0]}")

image2 = cv2.imread("images/tanjiro.png",  cv2.IMREAD_GRAYSCALE)
image2 = cv2.resize(image2, image.shape[:2][::-1])

image3 = cv2.imread("images/sabito.png")
image3 = cv2.resize(image3, image.shape[:2][::-1])

image5 = cv2.imread("images/shinobu.png", cv2.IMREAD_GRAYSCALE)
image5 = cv2.resize(image5, image.shape[:2][::-1])

image4 = cv2.imread("images/tokito.jpg")
image4 = cv2.resize(image4, image.shape[:2][::-1])

image6 = cv2.imread("images/kanao.jpg", cv2.IMREAD_GRAYSCALE)
image6 = cv2.resize(image6, image.shape[:2][::-1])

# Stack the frames
background = stackFrames(stack_shape=(3, 4), frames=[
    image, image2, image3, image4, image5], dtype=np.uint8)

# Display and save the resulting image
cv2.imshow('justTest1', background)
cv2.waitKey(5000)
cv2.imwrite("example_5.png", background, [cv2.IMWRITE_PNG_COMPRESSION, 9])
cv2.destroyAllWindows()
