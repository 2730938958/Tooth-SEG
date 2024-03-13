
import torch
from torch import nn
import pickle
from nnunet.training.model_restore import load_model_and_checkpoint_files
import os
import numpy as np

if __name__ == "__main__":

    parameters = []
    for i, f in enumerate(use_folds):
        f = int(f) if f != 'all' else f
        checkpoint = torch.load(join(model_training_output_dir, f'fold_{f}', checkpoint_name),
                                map_location=torch.device('cpu'))
        if i == 0:
            trainer_name = checkpoint['trainer_name']
            configuration_name = checkpoint['init_args']['configuration']
            inference_allowed_mirroring_axes = checkpoint['inference_allowed_mirroring_axes'] if \
                'inference_allowed_mirroring_axes' in checkpoint.keys() else None

        parameters.append(checkpoint['network_weights'])

    with torch.no_grad():
        prediction = None

        if prediction is None:
            for params in parameters:
                # messing with state dict names...
                if not isinstance(self.network, OptimizedModule):
                    self.network.load_state_dict(params)
                else:
                    self.network._orig_mod.load_state_dict(params)







    # convert .model to .onnx
    Model_path = "C:\Users\tanyu.mobi\Desktop\nnUNet-master\nnUNet-master\nnUNet_results\Dataset001_Teeth\nnUNetTrainer__nnUNetPlans__3d_lowres\fold_0"

    folds = '0'
    checkpoint_name = "checkpoint_final"

    trainer, params = load_model_and_checkpoint_files(Model_path, folds=folds, mixed_precision=True,
                                                      checkpoint_name=checkpoint_name)
    net = trainer.network

    checkpoint = torch.load(os.path.join(Model_path, folds, checkpoint_name + ".model"))

    net.load_state_dict(checkpoint['state_dict'])
    net.eval()
    # (1,1,10,20,30)是我i任意写的，你要按照自己的输入数组维度更换
    dummy_input = torch.randn(1, 1, 10, 20, 30).to("cuda")
    torch.onnx.export(
        net,
        dummy_input,
        'dynamic-1.onnx',
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )