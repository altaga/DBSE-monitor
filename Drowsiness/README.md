# Laptop Test:

To test the code on a computer, you should have previously followed the instructions at https://github.com/altaga/DBSE-monitor#laptop-test.

Open the Notebook.ipynb file in a jupyter notebook or the notebook.py file in any IDE that supports python.

NOTE: It is highly recommended to use jupyter notebook to run this code due to its ease of use.

## Model Creation:

Inside the "https://github.com/altaga/DBSE-monitor/tree/master/Drowsiness/Model" folder our model called "BlinkModel.t7" already exists, which is the one I use for all tests.

However the model can be trained by yourself with the code called "train.py" in the folder "https://github.com/altaga/DBSE-monitor/tree/master/Drowsiness/train".

The database that was used, is a database with 4846 images of left and right eyes, open and closed, where approximately half are open and closed so that the network was able to identify the state of the eyes, the database is in the following folder, it is a **.zip** file unzip before starting the training:

https://github.com/altaga/DBSE-monitor/blob/master/Drowsiness/train/dataset/dataset_B_Eye_Images.zip

The training has the following parameters as input.

- input image shape: (24, 24)
- validation_ratio: 0.1
- batch_size: 64
- epochs: 40
- learning rate: 0.001
- loss function: cross entropy loss
- optimizer: Adam

In the first part of the code you can modify the parameters, according to your training criteria.

Example how i train the model with VS code.
<img src="https://i.ibb.co/c1rBQvQ/image.png" width="1000">