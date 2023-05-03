#!/usr/bin/env python3

import os
import sys
import time
import pyopencl as cl
import numpy
from PIL import Image, ImageOps
from libs.util import *


# Parse CLI arguments
IMG_PATH = sys.argv[1] if len(
    sys.argv) > 1 else "images/behemoth-black-hole.jpg"
KERNEL_NAME_GREY = "kernel_grey"
KERNEL_NAME_THRESHOLD = "kernel_threshold"
KERNEL_NAME_STARS = "kernel_stars"
# Work group size will be (LOCAL_SIZE, LOCAL_SIZE).
LOCAL_SIZE = int(sys.argv[2]) if len(sys.argv) > 3 else 15

# Suppress kernel caching.
os.environ["PYOPENCL_NO_CACHE"] = "1"
os.environ["PYOPENCL_CTX"] = "0"

# Load input image and add alpha channel.
img = image_to_array(IMG_PATH)
#img = add_alpha_channel(img)

# Add padding with white pixels to the image.
#img = pad_image(img_no_padding, WINDOW_mid, [255, 255, 255, 0])

# Determine height and width of both original and padded image.
(img_h, img_w, depth) = img.shape
print(f"Image dimensions: {img_h}x{img_w} with depth {depth}")

# Flatten the image and the kernel, and make sure the types are correct.
flat_img = img.reshape(img_h * img_w * depth).astype(numpy.uint32)


# save_image_rgb(flat_img.reshape((img_h, img_w, depth)),
#               "images/andromeda_grey.jpg")


# Create the context, queue and program.
context = cl.create_some_context()
queue = cl.CommandQueue(context)

kernel_grey_code = open("kernels/" + KERNEL_NAME_GREY + ".cl").read()
kernel_threshold_code = open("kernels/" + KERNEL_NAME_THRESHOLD + ".cl").read()
kernel_stars_code = open("kernels/" + KERNEL_NAME_STARS + ".cl").read()
program_grey = cl.Program(context, kernel_grey_code).build()
program_threshold = cl.Program(context, kernel_threshold_code).build()
program_stars = cl.Program(context, kernel_stars_code).build()


# Initialize the kernel.
kernel_grey = program_grey.kernel_grey
kernel_threshold = program_threshold.kernel_threshold
kernel_stars = program_stars.kernel_stars

# Execute.

kernel_grey.set_scalar_arg_dtypes(
    [None, None,
        numpy.int32, numpy.int32, numpy.int32]
)

# Create the result image.
h_output_img = numpy.zeros(img_h * img_w * 1).astype(numpy.uint32)

# Create the buffers on the device.
d_input_img = cl.Buffer(
    context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=flat_img
)

d_output_img = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, h_output_img.nbytes)

print("Executing kernel grey...")
kernel_grey(
    queue,
    (img_h, img_w),
    (1, 1),
    d_input_img,
    d_output_img,
    img_w,
    img_h,
    depth
)
print("Done executing kernel grey.")
# queue.flush()

queue.finish()
# Read the array from the device.
cl.enqueue_copy(queue, h_output_img, d_output_img)

result_img = h_output_img.reshape(img_h, img_w)
#image = numpy.asarray(result_img).astype(dtype=numpy.uint32)
save_image_grey(result_img, "output/" + "output_" + KERNEL_NAME_GREY + ".png")

#result_img = h_output_img.reshape(img_h, img_w)
#save_image_grey(result_img, "output_" + KERNEL_NAME_GREY + ".png")

#########################################################
# Execute threshold kernel
#########################################################

queue = cl.CommandQueue(context)
LOCAL_SIZE = 1
global_size = int((img_w * img_h)/LOCAL_SIZE)

kernel_threshold.set_scalar_arg_dtypes(
    [None, None,
        numpy.int32, numpy.int32]
)

d_input_img = cl.Buffer(
    context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_output_img
)
h_input_brightness = numpy.zeros(
    global_size).astype(numpy.uint32)
# Create the buffers on the device for the output brightness
d_output_brightness = cl.Buffer(
    context, cl.mem_flags.READ_WRITE, h_input_brightness.nbytes)


print("Executing kernel threshold...")
kernel_threshold(
    queue,
    (global_size, 1),
    (LOCAL_SIZE, 1),
    d_output_img,
    d_output_brightness,
    img_w,
    img_h
)
print("Done executing kernel threshold.")

queue.finish()
# Read the array from the device.

cl.enqueue_copy(queue, h_input_brightness, d_output_brightness)
print("Brightness: ", h_input_brightness.sum())
threshold = 2 * h_input_brightness.sum()/(img_w * img_h)
threshold = math.floor(threshold)
print("the threshold is: ", threshold)

#########################################################
# Execute stars kernel
#########################################################

queue = cl.CommandQueue(context)
LOCAL_SIZE = 4

global_size = int((img_w * img_h)/LOCAL_SIZE)
print("global size: ", global_size)

kernel_stars.set_scalar_arg_dtypes(
    [None, None,
        numpy.int32, numpy.int32, numpy.int32]
)

d_input_img = cl.Buffer(
    context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_output_img
)
h_input_count = numpy.zeros(
    global_size).astype(numpy.uint32)
# Create the buffers on the device for the output brightness
d_output_count = cl.Buffer(
    context, cl.mem_flags.READ_WRITE, h_input_brightness.nbytes)


# round the threshold


print("Executing stars kernel...")
kernel_stars(
    queue,
    (global_size, 1),
    (LOCAL_SIZE, 1),
    d_input_img,
    d_output_count,
    threshold,
    img_h,
    img_w
)
print("Done executing stars kernel.")

queue.finish()
# Read the array from the device.

cl.enqueue_copy(queue, h_input_count, d_output_count)
print("Count: ", sum(h_input_count))

