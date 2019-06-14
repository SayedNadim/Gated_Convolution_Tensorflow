# GatedConvolution-TF
An unofficial TensorFlow implementation of Free-Form Image Inpainting with Gated Convolution (https://arxiv.org/abs/1806.03589).

#### This is still an ongoing implementation. Please feel free to suggest bugs or improvements. ####

### Dependencies
1. Python 3
2. Tensorflow >= 1.4
3. Neuralgym (https://github.com/JiahuiYu/neuralgym)

### How to train
1. Download Places2 dataset (http://places2.csail.mit.edu/download.html). High resolution images are suggested.
2. Create flist. Modify 'flist_maker' file and run 'python flist_maker'.
3. Modify 'inpaint.yml' file. Set 'RANDOMCROP' to 'True' if you have downloaded high resolution images. If set to 'True', the model will randomly crop 256x256 portion from the images. If set to 'False', the model will resize the images to 256x256 or use 256x256 images.
4. In case of resuming training, modify 'LOG_DIR' and start training from a checkpoint, e.g. 'LOG_DIR: 20190614190532339796_nadim-PMBSB09A-Samsung-DeskTop_places2_MASKED_sn_pgan_full_model_places2_256'

### How to test
1. Change 'npmask' function in 'test.py' to make different sized masks for testing.
2. Run 'python test.py --image IMAGE_LOCATION --checkpoint_dir CHECKPOINT_LOCATION' 
(e.g. python test.py --image 1.jpg --checkpoint_dir ./model_logs/20190614132226313697_nadim-PMBSB09A-Samsung-DeskTop_places2_MASKED_sn_pgan_full_model_places2_256/)

### To do list
- [x] Modify inpaint_ops.py. - Added modified discriminator convolution layers and mask generation function as per suggestions from https://github.com/JiahuiYu/generative_inpainting/issues/62. 
- [ ] Slim model by 25% -  Still training with original width of the model. You can slim the model by 25% by setting cnum = 32 * 0.75.
- [ ] Provide pretrained model - Will provide after finishing training.
- [ ] Provide loss graph - Will update soon.
- [ ] Provide test results - Still training.

### Code References
1. https://github.com/avalonstrel/GatedConvolution - The Baseline Code (Kudos!!)
2. https://github.com/JiahuiYu/generative_inpainting - Main Code 
3. https://github.com/JiahuiYu/generative_inpainting/issues/62 - Model suggestions
4. https://github.com/pfnet-research/sngan_projection/blob/master/updater.py#L21 - Hinge Loss
5. https://github.com/shepnerd/inpainting_gmcnn/blob/master/tensorflow/net/ops.py - Mask Generator Function
