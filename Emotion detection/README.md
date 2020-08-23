# Laptop Test:

To test the code on a computer, you should have previously followed the instructions at https://github.com/altaga/DBSE-monitor#laptop-test.

Open the Notebook.ipynb file in a jupyter notebook or the notebook.py file in any IDE that supports python.

NOTE: It is highly recommended to use jupyter notebook to run this code due to its ease of use.

## Model Creation:

Inside the "https://github.com/altaga/DBSE-monitor/tree/master/Emotion%20detection/model" folder our model called "emotions.t7" already exists, which is the one I use for all tests, however the model can be trained by yourself with the code called "train.py" in the folder "https://github.com/altaga/DBSE-monitor/tree/master/Emotion%20detection/train".

The database that was used, is a database with 28710 images of 'Angry','Disgust','Fear','Happy','Sad','Surprised','Neutral' people in CSV format, so that the network was able to identify the state of person face, the database is in the following folder link, download the CSV file into dataset folder:

www.kaggle.com/altaga/emotions

- The model is saved in the train/model folder every 10 epoch.

Example how i train the model with VS code.
<img src="https://i.ibb.co/nsn5sSy/image.png" width="1000">