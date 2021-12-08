# Automated Transfer of Musical Genre in Symbolic Form of Representation with the use of Deep Neural Networks

The extensive use of AI in music, as in all fields of artistic creation, has led to increased demand from creators and scholars for tools that will allow for new ways of
production and analysis of works. A potentially feasible tool is to transfer music tracks from genre to genre. The exploration of the multiple synergies of AI with music and in particular this project (genre to genre) has become a frequent subject of research in the past, using audio files. 

The approach presented in this thesis differs from the previous ones to the extent that it chooses to work with MIDI, which are symbolic data files that provide similar information to sheet music. This option requires the analysis of music at a compositional level, as variables that appear in sound files, such as timbre, are absent. As such, besides the obvious use for creative experimentation, such a transfer could offer more concrete information on the compositional differences between genres.

<p align="center">
  <img src="https://github.com/OrpheasK/thesis/blob/main/var/Part_of_sheet_music_from_arban.JPG"  width="32%" />
  <img src="https://github.com/OrpheasK/thesis/blob/main/var/unnamed.jpg"  width="38%" />
</p>
<p align="center">
  <i> Similarly to sheet music, MIDI files can be considered to contain a set of instructions for the performance of music by digital instruments</i>
</p>

The process we followed was to choose to simplify the encoding of information in a format sufficient to train a neural network with deep machine learning architectures. We organized experiments with a generic autoencoder model using separate decoders for each genre and a universal encoder. The model follows the protocol of an ordinary encoder-decoder model which is a way of organizing recurrent neural networks (RNN) for use in sequence-to-sequence prediction problems using LSTMs with multiple layers. A Teacher Forcing strategy was used to optimise the production of useful results. 

<p align="center">
  <img src="https://github.com/OrpheasK/thesis/blob/main/var/model%202%20eng.png" />
</p>
<p align="center">
  <i> Model architecture - the inputs and outputs of the model in this figure represent instances of the vector representation that was used for MIDI data</i>
</p>

The model was applied with common general architecture characteristics in two forms of representation. Experiments were produced, which were evaluated through a classifier, while those that gave interesting results were also subjected to human evaluation. Errors occurred at the completion of the work, which are explained by the quality and organization of the dataset and the wide range of the selected musical genres analyzed.

It has been concluded that the results of the project have delivered a good estimation on how it can be developed further, while providing useful information for targeted future research that can lead to the production of an easy-to-use processing tool for creative and academic purposes.
