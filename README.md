# Music-Genre-Classification
Training different classification models on GTZAN Genre collection dataset<br>


Music shops and especially online platforms such as Spotify, Apple Music, etc. require a music genre classification algorithm when suggesting and keeping different kinds of genres in their platforms. There are quite a number of music tracks in these platforms and organizing every music tracks by their genres is significantly difficult. By organizing each music track by its genre, the platforms can develop further systems to suggest users new tracks by predefined genres.

The dataset we have used for our project is GTZAN Genre Collection dataset. This
dataset includes 1000 music tracks for 10 different genres: blues, classical, country,
disco, hip-hop, jazz, reggae, rock, metal, and pop. Each genre has 100 audio tracks
inside and each of the tracks consists of 30 seconds. <br>
The audio tracks are preprocessed with the LibROSA package of python that
extracts features from spectrogram of audio tracks. For our methods, we have used
these extracted features and the pictures of the spectrogram.<br><br>
Link to dataset: http://marsyas.info/downloads/datasets.html

In our term project, we have decided to find the best music classification algorithm by genres. The best is defined by the highest accuracy. The accuracies are given for each music genre, average accuracy and top 3 accuracies. For the purpose of classification, we have used the following methods: 1D Convolutional Neural Network (1D CNN), 2D Convolutional Neural Network (2D CNN), Long Short-Term Memory (LSTM) and SVM.

The results (average accuracies) obtained from these methods are the following:<br> 
1D CNN with Raw Waveform: 51% and Top 3 accuracy: 79%<br>
1D CNN with Mel Spec: 64% and Top 3 accuracy: 86%<br> 
2D CNN with Mel Spec: 41% and Top 3 accuracy: 74%<br> 
Long Short-Term Memory (LSTM): 62% and Top 3 accuracy: 85%<br> 
SVM with RBF kernels: 64% 

![alt text](https://github.com/mmoksas68/Music-Genre-Classification/blob/master/result-images/1.PNG?raw=true)
<br>
![alt text](https://github.com/mmoksas68/Music-Genre-Classification/blob/master/result-images/2.PNG?raw=true)
<br>
![alt text](https://github.com/mmoksas68/Music-Genre-Classification/blob/master/result-images/3.PNG?raw=true)
<br>
![alt text](https://github.com/mmoksas68/Music-Genre-Classification/blob/master/result-images/4.PNG?raw=true)
<br>
![alt text](https://github.com/mmoksas68/Music-Genre-Classification/blob/master/result-images/5.PNG?raw=true)

