# Music-Genre-Classification
Training different classification models on GTZAN Genre collection dataset

Music shops and especially online platforms such as Spotify, Apple Music, etc. require a music genre classification algorithm when suggesting and keeping different kinds of genres in their platforms. There are quite a number of music tracks in these platforms and organizing every music tracks by their genres is significantly difficult. By organizing each music track by its genre, the platforms can develop further systems to suggest users new tracks by predefined genres.

In our term project, we have decided to find the best music classification algorithm by genres. The best is defined by the highest accuracy. The accuracies are given for each music genre, average accuracy and top 3 accuracies. For the purpose of classification, we have used the following methods: 1D Convolutional Neural Network (1D CNN), 2D Convolutional Neural Network (2D CNN), Long Short-Term Memory (LSTM) and SVM.

The results (average accuracies) obtained from these methods are the following: 1D CNN with Raw Waveform: 51% and ​Top 3 accuracy: 79% 
1D CNN with Mel Spec: 64% and Top 3 accuracy: 86% 
2D CNN with Mel Spec: 41% and Top 3 accuracy: 74% 
Long Short-Term Memory (LSTM): 62% and Top 3 accuracy: 85% 
SVM with RBF kernels: 64% 
