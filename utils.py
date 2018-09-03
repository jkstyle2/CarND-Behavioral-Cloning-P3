import cv2

def bgr2rgb(image):
    new_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return new_image

def blur_img(image):
    new_image = cv2.GaussianBlur(image, (3,3), 0)
    return new_image

def rgb2yuv(image):
    new_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    return new_image

def bgr2yuv(image):
    new_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    return new_image

# original shape: 160x320x3, input shape for neural net: 70x160x3
def crop_img(image):
    cropped = image[60:130, :]  # crop the rows between 60 and 130 in images. NOTE: each row and column becomes transpose
    return cropped

def resize_img(image, shape=(160,70)):  # As cropped image becomes transpose, the shape for resizing is also transposed.
    resized = cv2.resize(image, shape)
    return resized

def crop_and_resize(image):
    cropped = crop_img(image)
    cropped_resized = resize(cropped)
    return cropped_resized

def flip_img(image):
    return cv2.flip(image, 1)




####################################################33

# def bgr2rgb(image):
#     return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
# def flipimg(image):
#     return cv2.flip(image, 1)
#
# def cropimg(image):
#     cropped = image[60:130, :]
#     return cropped
#
# def resize(image, shape=(160, 70)):
#     return cv2.resize(image, shape)
#
# def crop_and_resize(image):
#     cropped = cropimg(image)
#     resized = resize(cropped)
#     return resized