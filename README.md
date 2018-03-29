
# Generating Images from text
In this, we have worked on generating bird images conditioned on the binary segmentation masks and text embeddings


### Dependencies
+ Tensorflow >= 1.0
+ Scipy
+ Scikit-image
+ torchfile

### Preprocessing
+ Download the [birds](https://drive.google.com/file/d/0B0ywwgffWnLLLUc2WHYzM0Q2eWc/view?usp=sharing) caption data in Torch format
+ Download the [birds](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) image data
+ Extract and place the images and segmentation masks in a directory `CUB`
+ Run the script to create tfrecords for training the model: 

    ```
    python preprocess_data.py
    ```

### Training
+ For training the model:

  ```
  python main.py --batch_size=64 --output_size=128
  ```
