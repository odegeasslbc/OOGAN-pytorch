training_config_templete = {
    "DATASET": "dsprites",  # you can choose from {"dsprites", any other words you put here}
    "DATA_ROOT": "/path/tp/the/datafolder", # follow the pytorch's torchvision.dataset.Imagefolder's path format
    "DATALOADER_WORKERS":8, # pytorch multi-thread cpu data loading config
    "NOISE_ANNEL": False, # usually not necessary
 
    "SAVE_FOLDER": './', # path to put the "train_results" folder, which will contain trained models and images during training
    "MAX_ITERATION": 100000, # usually training will collapse after 50000 iterations
    "BATCH_SIZE": 10,
    "CUDA_ID": 0, # using cuda by default, 0 is the first GPU

    "NGF": 64, # determines the generator width (the channel number multiplier)
    "NDF": 64, # determines the discriminator width (the channel number multiplier)
    "NC": 1, # the image channel, usually 1 for grey scale or 3 for rgb image
    "USE_PROB_C": True, # determines Q is deterministic or probabilitisc
    "ONE_HOT": True, # weather use One-hot sampling trick or not
    "G_TYPE": "OOGAN", # choose from {"OOGAN", "InfoGAN"}
    "D_TYPE": "OOGAN", # choose from {"OOGAN", "InfoGAN"}
    "Z_DIM": 10, # noise dimension, usually 100 on 128x128 images
    "C_DIM": 6, # the control vector c dimension, usually 100 on 128x128 images
    "IM_SIZE": 64, # can choose from {32, 64, 128, 256} 
    "LR": 1e-4, # usually no need to change it
    "LAMBDA": 1, # controls the weights of the mutual information part in OOGAN's loss, usually just set to 1 is fine
    "GAMMA": 1,  # controls the weights of the one-hot trick part in OOGAN's loss, usually just set to 1 is fine
    "TRIAL_NAME": 'trial_dsprite1' # it determines the save folder's name, please make sure to set a new name when you start a new training
}

training_config_celeba = {
    "DATASET": "celebA",  # you can choose from {"dsprites", any other words you put here}
    "DATA_ROOT": "./celeba/", # follow the pytorch's torchvision.dataset.Imagefolder's path format
    "DATALOADER_WORKERS":8, # pytorch multi-thread cpu data loading config
    "NOISE_ANNEL": False, # usually not necessary
 
    "SAVE_FOLDER": './', # path to put the "train_results" folder, which will contain trained models and images during training
    "MAX_ITERATION": 50000, # usually training will collapse after 50000 iterations
    "BATCH_SIZE": 8,
    "CUDA_ID": 0, # using cuda by default

    "NGF": 64, # determines the generator width (the channel number multiplier)
    "NDF": 64, # determines the discriminator width (the channel number multiplier)
    "NC": 3, # the image channel, usually 1 for grey scale or 3 for rgb image
    "USE_PROB_C": True, # Q is deterministic or probabilitisc
    "ONE_HOT": True, # weather use One-hot sampling trick or not
    "G_TYPE": "OOGAN", # choose from {"OOGAN", "InfoGAN"}
    "D_TYPE": "OOGAN", # choose from {"OOGAN", "InfoGAN"}
    "Z_DIM": 100, # noise dimension, usually 100 on 128x128 images
    "C_DIM": 16, # the control vector c dimension, usually 8 on 128x128 images
    "IM_SIZE": 256, # can choose from {32, 64, 128, 256} 
    "LR": 1e-4, # usually no need to change it
    "LAMBDA": 1, # controls the weights of the mutual information part in OOGAN's loss, usually just set to 1 is fine
    "GAMMA": 1,  # controls the weights of the one-hot trick part in OOGAN's loss, usually just set to 1 is fine
    "TRIAL_NAME": 'trial_celeba_oogan_2' # it determines the save folder's name, please make sure to set a new name when you start a new training
}

training_config_3dchair = {
    "DATASET": "3Dchair",  # you can choose from {"dsprites", any other words you put here}
    "DATA_ROOT": "/media/bingchen/database/3Dchair/rendered_chairs", # follow the pytorch's torchvision.dataset.Imagefolder's path format
    "DATALOADER_WORKERS":8, # pytorch multi-thread cpu data loading config
    "NOISE_ANNEL": False, # usually not necessary
 
    "SAVE_FOLDER": './', # path to put the "train_results" folder, which will contain trained models and images during training
    "MAX_ITERATION": 50000, # usually training will collapse after 50000 iterations
    "BATCH_SIZE": 20,
    "CUDA_ID": 0, # using cuda by default

    "NGF": 64, # determines the generator width (the channel number multiplier)
    "NDF": 64, # determines the discriminator width (the channel number multiplier)
    "NC": 3, # the image channel, usually 1 for grey scale or 3 for rgb image
    "USE_PROB_C": True, # determines Q is deterministic or probabilitisc
    "ONE_HOT": True, # weather use One-hot sampling trick or not
    "G_TYPE": "OOGAN", # choose from {"OOGAN", "InfoGAN"}
    "D_TYPE": "OOGAN", # choose from {"OOGAN", "InfoGAN"}
    "Z_DIM": 50, # noise dimension, usually 100 on 128x128 images
    "C_DIM": 10, # the control vector c dimension, usually 100 on 128x128 images
    "IM_SIZE": 64, # can choose from {32, 64, 128, 256} 
    "LR": 1e-4, # usually no need to change it
    "LAMBDA": 1, # controls the weights of the mutual information part in OOGAN's loss, usually just set to 1 is fine
    "GAMMA": 1,  # controls the weights of the one-hot trick part in OOGAN's loss, usually just set to 1 is fine
    "TRIAL_NAME": 'trial_3dchair_oogan_1' # it determines the save folder's name, please make sure to set a new name when you start a new training
}

training_config_dsprites = {
    "DATASET": "dsprites",  # you can choose from {"dsprites", any other words you put here}
    "DATA_ROOT": "/media/bingchen/wander/research_disentangle/data/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz", # follow the pytorch's torchvision.dataset.Imagefolder's path format
    "DATALOADER_WORKERS":8, # pytorch multi-thread cpu data loading config
    "NOISE_ANNEL": True, # usually not necessary
 
    "SAVE_FOLDER": './', # path to put the "train_results" folder, which will contain trained models and images during training
    "MAX_ITERATION": 50000, # usually training will collapse after 50000 iterations
    "BATCH_SIZE": 64,
    "CUDA_ID": 0, # using cuda by default

    "NGF": 64, # determines the generator width (the channel number multiplier)
    "NDF": 64, # determines the discriminator width (the channel number multiplier)
    "NC": 1, # the image channel, usually 1 for grey scale or 3 for rgb image
    "USE_PROB_C": True, # determines Q is deterministic or probabilitisc
    "ONE_HOT": True, # weather use One-hot sampling trick or not
    "G_TYPE": "InfoGAN", # choose from {"OOGAN", "InfoGAN"}
    "D_TYPE": "OOGAN", # choose from {"OOGAN", "InfoGAN"}
    "Z_DIM": 10, # noise dimension, usually 100 on 128x128 images
    "C_DIM": 6, # the control vector c dimension, usually 100 on 128x128 images
    "IM_SIZE": 32, # can choose from {32, 64, 128, 256} 
    "LR": 1e-4, # usually no need to change it
    "LAMBDA": 1, # controls the weights of the mutual information part in OOGAN's loss, usually just set to 1 is fine
    "GAMMA": 1,  # controls the weights of the one-hot trick part in OOGAN's loss, usually just set to 1 is fine
    "TRIAL_NAME": 'trial_dsprite1_iogan_1' # it determines the save folder's name, please make sure to set a new name when you start a new training
}




training_config = training_config_celeba