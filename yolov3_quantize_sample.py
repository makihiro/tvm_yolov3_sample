import nnvm
import nnvm.frontend.darknet
import nnvm.testing.yolo_detection
import nnvm.testing.darknet
import matplotlib.pyplot as plt
import numpy as np
import tvm
from tvm import rpc
import sys
import cv2
import time

from ctypes import *
from tvm.contrib import util
from tvm.contrib.download import download
from nnvm.testing.darknet import __darknetffi__

# Model name
MODEL_NAME = 'yolov3'

######################################################################
# Download required files
# -----------------------
# Download cfg and weights file if first time.
CFG_NAME = MODEL_NAME + '.cfg'
WEIGHTS_NAME = MODEL_NAME + '.weights'
REPO_URL = 'https://github.com/siju-samuel/darknet/blob/master/'
CFG_URL = REPO_URL + 'cfg/' + CFG_NAME + '?raw=true'
WEIGHTS_URL = 'https://pjreddie.com/media/files/' + WEIGHTS_NAME

download(CFG_URL, CFG_NAME)
download(WEIGHTS_URL, WEIGHTS_NAME)

# Download and Load darknet library
if sys.platform in ['linux', 'linux2']:
    DARKNET_LIB = 'libdarknet2.0.so'
    DARKNET_URL = REPO_URL + 'lib/' + DARKNET_LIB + '?raw=true'
elif sys.platform == 'darwin':
    DARKNET_LIB = 'libdarknet_mac2.0.so'
    DARKNET_URL = REPO_URL + 'lib_osx/' + DARKNET_LIB + '?raw=true'
else:
    err = "Darknet lib is not supported on {} platform".format(sys.platform)
    raise NotImplementedError(err)

download(DARKNET_URL, DARKNET_LIB)

DARKNET_LIB = __darknetffi__.dlopen('./' + DARKNET_LIB)
cfg = "./" + str(CFG_NAME)
weights = "./" + str(WEIGHTS_NAME)
net = DARKNET_LIB.load_network(cfg.encode('utf-8'), weights.encode('utf-8'), 0)
dtype = 'float32'
batch_size = 1

print("Converting darknet to nnvm symbols...")
sym, params = nnvm.frontend.darknet.from_darknet(net, dtype)

######################################################################
# Compile the model on NNVM
# -------------------------
# compile the model
local = True

if local:
    target = 'llvm'
    ctx = tvm.cpu(0)
else:
    target = 'cuda'
    ctx = tvm.gpu(0)

data = np.empty([batch_size, net.c, net.h, net.w], dtype)
shape = {'data': data.shape}

dtype_dict = {}

# convert nnvm to relay
print("convert nnvm symbols into relay function...")
from nnvm.to_relay import to_relay
func, params = to_relay(sym, shape, 'float32', params=params)
# optimization
print("optimize relay graph...")
with tvm.relay.build_config(opt_level=2):
    func = tvm.relay.optimize(func, target, params)
# quantize
print("apply quantization...")
from tvm.relay import quantize
with quantize.qconfig():
   func = quantize.quantize(func, params)

# Relay build
print("Compiling the model...")
print(func.astext(show_meta_data=False))
with tvm.relay.build_config(opt_level=3):
    graph, lib, params = tvm.relay.build(func, target=target, params=params)

# Save the model
tmp = util.tempdir()
lib_fname = tmp.relpath('model.tar')
lib.export_library(lib_fname)

# NNVM
# with nnvm.compiler.build_config(opt_level=2):
#     graph, lib, params = nnvm.compiler.build(sym, target, shape, dtype_dict, params)


[neth, netw] = shape['data'][2:]  # Current image shape is 608x608
######################################################################
# Execute on TVM Runtime
# ----------------------
# The process is no different from other examples.
from tvm.contrib import graph_runtime

if local:
    remote = rpc.LocalSession()
    ctx = remote.cpu(0)
else:
    # The following is my environment, change this to the IP address of your target device
    host = 'localhost'
    port = 9090
    remote = rpc.connect(host, port)
    ctx = remote.gpu(0)

# upload the library to remote device and load it
remote.upload(lib_fname)
rlib = remote.load_module('model.tar')

# create the remote runtime module
m = graph_runtime.create(graph, rlib, ctx)
m.set_input(**params)
thresh = 0.5
nms_thresh = 0.45
coco_name = 'coco.names'
coco_url = 'https://github.com/siju-samuel/darknet/blob/master/data/' + coco_name + '?raw=true'
font_name = 'arial.ttf'
font_url = 'https://github.com/siju-samuel/darknet/blob/master/data/' + font_name + '?raw=true'
download(coco_url, coco_name)
download(font_url, font_name)

with open(coco_name) as f:
    content = f.readlines()

names = [x.strip() for x in content]

# test image demo
test_image = 'dog.jpg'
print("Loading the test image...")
img_url = 'https://github.com/siju-samuel/darknet/blob/master/data/' + \
          test_image + '?raw=true'
download(img_url, test_image)

data = nnvm.testing.darknet.load_image(test_image, netw, neth)
# set inputs
m.set_input('data', tvm.nd.array(data.astype(dtype)))
# execute
print("Running the test image...")

m.run()
# get outputs
tvm_out = []
for i in range(3):
    layer_out = {}
    layer_out['type'] = 'Yolo'
    # Get the yolo layer attributes (n, out_c, out_h, out_w, classes, total)
    layer_attr = m.get_output(i*4+3).asnumpy()
    layer_out['biases'] = m.get_output(i*4+2).asnumpy()
    layer_out['mask'] = m.get_output(i*4+1).asnumpy()
    out_shape = (layer_attr[0], layer_attr[1] // layer_attr[0],
                 layer_attr[2], layer_attr[3])
    layer_out['output'] = m.get_output(i*4).asnumpy().reshape(out_shape)
    layer_out['classes'] = layer_attr[4]
    tvm_out.append(layer_out)

img = nnvm.testing.darknet.load_image_color(test_image)
_, im_h, im_w = img.shape
dets = nnvm.testing.yolo_detection.fill_network_boxes((netw, neth), (im_w, im_h), thresh,
                                                      1, tvm_out)
last_layer = net.layers[net.n - 1]
nnvm.testing.yolo_detection.do_nms_sort(dets, last_layer.classes, nms_thresh)
nnvm.testing.yolo_detection.draw_detections(img, dets, thresh, names, last_layer.classes)

plt.imshow(img.transpose(1, 2, 0))
plt.show()



# video demo
video_demo = False
if video_demo:

    #vcap = cv2.VideoCapture("video.mp4")
    vcap = cv2.VideoCapture(0)

    n_frames = 0
    seconds = 0.0
    fps = 0.0
    while True:
        # Start time
        start = time.time()
        # Capture frame-by-frame
        n_frames = n_frames + 1

        ret, frame = vcap.read()
        img = np.array(frame)
        img = img.transpose((2, 0, 1))
        img = np.divide(img, 255.0)
        img = np.flip(img, 0)
        data = nnvm.testing.darknet._letterbox_image(img, netw, neth)
        # set inputs
        m.set_input('data', tvm.nd.array(data.astype(dtype)))
        # execute
        print("Running the test image...")

        m.run()
        # get outputs
        tvm_out = []

        #tvm_output_list = []
        # for i in range(0, 3):
        #     tvm_output = m.get_output(i)
        #     tvm_output_list.append(tvm_output.asnumpy())
        #print(tvm_output_list)
        #print(m.get_num_outputs())
        #layer_attr = [m.get_output(i).asnumpy() for i in range(m.get_num_outputs())]

        for i in range(3):
            layer_out = {}
            layer_out['type'] = 'Yolo'
            # Get the yolo layer attributes (n, out_c, out_h, out_w, classes, total)
            layer_attr = m.get_output(i*4+3).asnumpy()
            layer_out['biases'] = m.get_output(i*4+2).asnumpy()
            layer_out['mask'] = m.get_output(i*4+1).asnumpy()
            out_shape = (layer_attr[0], layer_attr[1] // layer_attr[0],
                         layer_attr[2], layer_attr[3])
            layer_out['output'] = m.get_output(i*4).asnumpy().reshape(out_shape)
            layer_out['classes'] = layer_attr[4]
            tvm_out.append(layer_out)

        _, im_h, im_w = img.shape
        dets = nnvm.testing.yolo_detection.fill_network_boxes((netw, neth), (im_w, im_h), thresh,
                                                              1, tvm_out)
        last_layer = net.layers[net.n - 1]
        nnvm.testing.yolo_detection.do_nms_sort(dets, last_layer.classes, nms_thresh)
        nnvm.testing.yolo_detection.draw_detections(img, dets, thresh, names, last_layer.classes)
        # End time
        end = time.time()

        # Time elapsed
        seconds = (end - start)
        # Calculate frames per second
        fps = (fps + (1 / seconds)) / 2
        print(fps)
        cv2.putText(img, str(fps), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        cv2.imshow('Video', img.transpose(1, 2, 0))
        #cv2.waitKey(3)
        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
