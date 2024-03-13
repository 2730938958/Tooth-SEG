import json
import argparse
import os

import numpy as np
import acl
import cv2
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

from constant import ACL_MEM_MALLOC_HUGE_FIRST, \
    ACL_MEMCPY_HOST_TO_DEVICE, ACL_MEMCPY_DEVICE_TO_HOST, \
    ACL_SUCCESS, IMG_EXT, NPY_FLOAT32



def check_ret(message, ret):
    """检查返回数据"""
    if ret != ACL_SUCCESS:
        raise Exception("{} failed ret={}"
                        .format(message, ret))


class Net(object):
    """推理网络定义"""
    def __init__(self, device_id, model_path):
        self.device_id = device_id
        self.model_path = model_path
        self.model_id = None
        self.context = None

        self.input_data = []
        self.output_data = []
        self.model_desc = None
        self.load_input_dataset = None
        self.load_output_dataset = None

        self.init_resource()


    def release_resource(self):
        """释放ACL资源"""
        print("Releasing resources stage:")
        ret = acl.mdl.unload(self.model_id)
        check_ret("acl.mdl.unload", ret)
        if self.model_desc:
            acl.mdl.destroy_desc(self.model_desc)
            self.model_desc = None

        while self.input_data:
            item = self.input_data.pop()
            ret = acl.rt.free(item["buffer"])
            check_ret("acl.rt.free", ret)

        while self.output_data:
            item = self.output_data.pop()
            ret = acl.rt.free(item["buffer"])
            check_ret("acl.rt.free", ret)

        if self.context:
            ret = acl.rt.destroy_context(self.context)
            check_ret("acl.rt.destroy_context", ret)
            self.context = None

        ret = acl.rt.reset_device(self.device_id)
        check_ret("acl.rt.reset_device", ret)
        print('Resources released successfully.')

    def init_resource(self):
        """初始化ACL资源"""
        acl.init()
        print("init resource stage:")
        ret = acl.rt.set_device(self.device_id)
        check_ret("acl.rt.set_device", ret)

        self.context, ret = acl.rt.create_context(self.device_id)
        check_ret("acl.rt.create_context", ret)

        # load_model
        self.model_id, ret = acl.mdl.load_from_file(self.model_path)
        check_ret("acl.mdl.load_from_file", ret)
        print("model_id:{}".format(self.model_id))

        self.model_desc = acl.mdl.create_desc()
        self._get_model_info()
        print("init resource success")

    def _get_model_info(self,):
        """获取模型信息"""
        ret = acl.mdl.get_desc(self.model_desc, self.model_id)
        check_ret("acl.mdl.get_desc", ret)
        input_size = acl.mdl.get_num_inputs(self.model_desc)
        output_size = acl.mdl.get_num_outputs(self.model_desc)

    def _gen_data_buffer(self, size, des):
        """准备模型输入输出数据"""
        func = buffer_method[des]
        for i in range(size):
            # check temp_buffer dtype
            temp_buffer_size = func(self.model_desc, i)
            temp_buffer, ret = acl.rt.malloc(temp_buffer_size,
                                             ACL_MEM_MALLOC_HUGE_FIRST)
            check_ret("acl.rt.malloc", ret)

            if des == "in":
                self.input_data.append({"buffer": temp_buffer,
                                        "size": temp_buffer_size})
            elif des == "out":
                self.output_data.append({"buffer": temp_buffer,
                                         "size": temp_buffer_size})

    def _data_interaction(self, dataset, policy=ACL_MEMCPY_HOST_TO_DEVICE):
        """在host和device之间通过复制内存进行数据传输"""
        temp_data_buffer = self.input_data \
            if policy == ACL_MEMCPY_HOST_TO_DEVICE \
            else self.output_data
        if len(dataset) == 0 and policy == ACL_MEMCPY_DEVICE_TO_HOST:
            for item in self.output_data:
                temp, ret = acl.rt.malloc_host(item["size"])
                if ret != 0:
                    raise Exception("can't malloc_host ret={}".format(ret))
                dataset.append({"size": item["size"], "buffer": temp})

        for i, item in enumerate(temp_data_buffer):
            if policy == ACL_MEMCPY_HOST_TO_DEVICE:
                if "bytes_to_ptr" in dir(acl.util):
                    bytes_data = dataset[i].tobytes()
                    ptr = acl.util.bytes_to_ptr(bytes_data)
                else:
                    ptr = acl.util.numpy_to_ptr(dataset[i])
                ret = acl.rt.memcpy(item["buffer"],
                                    item["size"],
                                    ptr,
                                    item["size"],
                                    policy)
                check_ret("acl.rt.memcpy", ret)

            else:
                ptr = dataset[i]["buffer"]
                ret = acl.rt.memcpy(ptr,
                                    item["size"],
                                    item["buffer"],
                                    item["size"],
                                    policy)
                check_ret("acl.rt.memcpy", ret)

    def _gen_dataset(self, type_str="input"):
        """准备模型输入输出数据集"""
        dataset = acl.mdl.create_dataset()

        temp_dataset = None
        if type_str == "in":
            self.load_input_dataset = dataset
            temp_dataset = self.input_data
        else:
            self.load_output_dataset = dataset
            temp_dataset = self.output_data

        for item in temp_dataset:
            data = acl.create_data_buffer(item["buffer"], item["size"])
            _, ret = acl.mdl.add_dataset_buffer(dataset, data)

            if ret != ACL_SUCCESS:
                ret = acl.destroy_data_buffer(data)
                check_ret("acl.destroy_data_buffer", ret)

    def _data_from_host_to_device(self, images):
        """将数据从host传到device"""
        print("data interaction from host to device")
        # copy images to device
        self._data_interaction(images, ACL_MEMCPY_HOST_TO_DEVICE)
        # load input data into model
        self._gen_dataset("in")
        # load output data into model
        self._gen_dataset("out")
        print("data interaction from host to device success")

    def _data_from_device_to_host(self):
        """将数据从device传到host"""
        print("data interaction from device to host")
        res = []
        # copy device to host
        self._data_interaction(res, ACL_MEMCPY_DEVICE_TO_HOST)
        print("data interaction from device to host success")
        result = self.get_result(res)
        pred_dict = self._print_result(result)
        # free host memory
        for item in res:
            ptr = item['buffer']
            ret = acl.rt.free_host(ptr)
            check_ret('acl.rt.free_host', ret)
        return pred_dict

    def run(self, images):
        """执行数据传输和模型推理"""
        self._data_from_host_to_device(images)
        self.forward()
        pred_dict = self._data_from_device_to_host()
        return pred_dict

    def forward(self):
        """模型推理"""
        print('execute stage:')
        ret = acl.mdl.execute(self.model_id,
                              self.load_input_dataset,
                              self.load_output_dataset)
        check_ret("acl.mdl.execute", ret)
        self._destroy_databuffer()
        print('execute stage success')

def main():
    device = 0
    model_path = 'nnunet.om'
    data_path = './data'
    data_list = [os.path.join(data_path, data)
                   for data in os.listdir(data_path)]

    net = Net(device, model_path)
    # for data in data_list:
    #     x = preprocess_data(data)
    #     pred_dict = net.run([x])

    print("*****run finish******")
    net.release_resource()

main()