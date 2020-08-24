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

# How does it work:

Let's go through a revision of the algorithms and procedures of the CV system (Emotion recognition).

ALL the code is well explained in "Notebook.ipynb" file.

Please take a look at it for extensive explanation and documentation.

The emotion monitor uses the following libraries:

- OpenCV:
- Image processing. 
    - (OpenCV) Haarcascades implementation. 
    - (OpenCV) Face detection
    - (Pytorch) Emotion detection
- VLC: 
    - Music player.

The emotion detection algorithm is as follows:

- Detection that there is a face of a person behind the wheel:

<img src="https://i.ibb.co/ZMvwvfp/Face.png" width="600">

- Once we have detected the face, we cut it out of the image so that we can use them as input for our convolutional PyTorch network.

<img src="https://i.ibb.co/xDgvMBD/Neutral.png" width="600">

- The model is designed to detect the emotion of the face, this emotion will be saved in a variable to be used by our song player.

<img src="https://i.ibb.co/1Q7M3ks/image.png" width="600">

- According to the detected emotion we will randomly reproduce a song from one of our playlists:

    - If the person is angry we will play a song that generates calm
    - If the person is sad, a song for the person to be happy
    - If the person is neutral or happy we will play some of their favorite songs

Note: If the detected emotion has not changed, the playlist will continue without changing the song.