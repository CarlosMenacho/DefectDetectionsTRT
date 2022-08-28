import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import common
import os
import cv2
import time
import socket
import torchvision.transforms as transforms
import random

TRT_LOGGER = trt.Logger()

def get_engine(onnx_file_path, engine_file_path=""):      
    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(common.EXPLICIT_BATCH) as network, builder.create_builder_config() as config, trt.OnnxParser(network, TRT_LOGGER) as parser, trt.Runtime(TRT_LOGGER) as runtime:
            config.max_workspace_size = 1 << 28 
            builder.max_batch_size = 1
            if not os.path.exists(onnx_file_path):
                print('ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(onnx_file_path))
                exit(0)

            with open(onnx_file_path, 'rb') as model:
                print("parsing ONNX model")
                if not parser.parse(model.read()):
                    print("Failed to parse onnx model")
                    for errors in range(parser.num_errors):
                        print (parser.get_error(errors))

                    return None

            network.get_input(0).shape = [1, 3, 224, 224]
            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
            plan = builder.build_serialized_network(network, config)
            engine = runtime.deserialize_cuda_engine(plan)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(plan)
            return engine

def preprocess_img(img_path):
    input_img = cv2.imread(img_path)
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    input_img = cv2.resize(input_img,dsize=(224,224), interpolation = cv2.INTER_CUBIC)
    input_img = cv2.normalize(input_img.astype('float32'), None, -0.5, .5, cv2.NORM_MINMAX)

    transform = transforms.ToTensor()
    tensor = transform(input_img)
    tensor = np.expand_dims(tensor,axis=0)
    return tensor

def show_sample(img_path, predicted_img):
    img = cv2.imread(img_path)
    if predicted_img in img_path:
        cv2.putText(img=img, text=str(predicted_img),
            org=(30, 40), fontFace=cv2.FONT_HERSHEY_TRIPLEX, 
            fontScale=1, color=(0, 255, 0),thickness=1)
    else:
        cv2.putText(img=img, text=str(predicted_img),
            org=(30, 40), fontFace=cv2.FONT_HERSHEY_TRIPLEX, 
            fontScale=1, color=(0, 0, 255),thickness=1)
    
    
    cv2.imshow("names",img)
    cv2.waitKey(2000)


def get_infer():
    onnx_file_path = 'resnet50_dip.onnx'
    engine_file_path = "resnet50_dip.trt"
    def_paths = '/home/carlos/diplo_ws/test/def_front/'
    ok_paths = '/home/carlos/diplo_ws/test/ok_front/'
    img_ok_paths = os.listdir(ok_paths)
    img_def_paths = os.listdir(def_paths)
    
    trt_outputs = []
    with get_engine(onnx_file_path, engine_file_path) as engine, engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        if bool(random.getrandbits(1))== True:
            img_path = ok_paths+random.choice(img_ok_paths)
            inputs_img = preprocess_img(img_path)
        else:
            img_path = def_paths+random.choice(img_def_paths)
            inputs_img = preprocess_img(img_path)
        inputs[0].host = inputs_img
        start = time.time()
        trt_outputs = common.do_inference_v2(context, bindings = bindings, inputs = inputs, outputs = outputs, stream = stream)
        print("inference time: ", time.time()-start)
        if(trt_outputs[0][1]> 0.5):
            print("Pieza normal: ", img_path)
            out_str = "good"
            show_sample(img_path, "ok")
        else:
            print("Pieza deformada: ",img_path )
            out_str = "bad"
            show_sample(img_path, "def")

    return out_str

