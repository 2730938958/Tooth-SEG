from nnunetv2.inference.predict_from_raw_data import predict_entry_point

if __name__=='__main__':
    """
    please add argument according to predict_entry_point() and set environment variables according to your path
    
    export nnUNet_raw="/home/nnUNet/nnUNet_raw"
    export nnUNet_results="/home/nnUNet/nnUNet_results"
    export nnUNet_preprocessed="/home/nnUNet/nnUNet_preprocessed"
    
    -i /home/nnUNet/imagesTs -o /home/nnUNet/infer -d 1 -c 3d_lowres -device cpu -f 0 --save_probabilities
    """
    predict_entry_point()