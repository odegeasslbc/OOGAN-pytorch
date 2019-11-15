# OOGAN-pytorch
Pytorch implementation of the paper: [OOGAN: Disentangling GAN with One-Hot Sampling and Orthogonal Regularization](https://arxiv.org/abs/1905.10836)


# OOGAN based on vanilla GAN

1. code to define the networks: oogan_models.py, oogan_modules.py
2. to train the model on your data, first prepare your images inside a root folder with subfolders containing your images, then edit the "config.py" for all hyperparameter settings, templetes are provided inside, 
   then run 
    ```python
    python train.py
    ```  
   the training will print log on terminal, and save the generated models and images during training.  
3. to generate images from trained model, first edit the model path in "generate.py", then run:
   ```python
   python generate.py
   ```
   
# OOGAN based on styleGAN

1. code: ./oogan_stylegan/oo_stylegan_train.py, 
         ./oogan_stylegan/oo_stylegan_modules.py
2. to train the model on your data
   ```python
   python oo_stylegan_train.py /path/to/image_root
   ```  
   the training will print log on terminal and save the generated models and images during training.
