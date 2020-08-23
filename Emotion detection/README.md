# Laptop Test:

To test the code on a computer, previamente debiste seguir las intrucciones de https://github.com/altaga/DBSE-monitor#laptop-test.

Abre el archivo Notebook.ipynb en un jupyter notebook o el archivo Notebook.py en cualquier IDE que permita ejecutar python.

## Model Creation:

Inside the "https://github.com/altaga/DBSE-monitor/tree/master/Emotion%20detection/model" folder our model called "emotions.t7" already exists, which is the one I use for all tests, however the model can be trained by yourself with the code called "train.py" in the folder "https://github.com/altaga/DBSE-monitor/tree/master/Emotion%20detection/train".

The database that was used, is a database with 28710 images of 'Angry','Disgust','Fear','Happy','Sad','Surprised','Neutral' people in CSV format, so that the network was able to identify the state of person face, the database is in the following folder link, download the CSV file into dataset folder:

www.kaggle.com/altaga/emotions

- The model is saved in the train/model folder every 10 epoch.