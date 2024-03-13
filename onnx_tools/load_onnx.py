import onnx
import torch.onnx
import numpy as np
# import cv2
import onnxruntime as rt



def load_onnx():
    onnx_model = onnx.load('./nnunet.onnx')  # load onnx model
    onnx.checker.check_model(onnx_model)  # check onnx model
    print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model


def test_speed():
    x = np.load('x.npy')
    sess = rt.InferenceSession('nnunet-v11.onnx')
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    output = sess.run([output_name], {input_name: x})

    print(output)

if __name__ == '__main__':
    test_speed()