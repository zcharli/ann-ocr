# Optical Character Recognition through Aritificial Neural Networks
Inspired and modeled after the central nervous systems of the brain, Artificial Neural
Networks (ANN) are a family of AI techniques used to produce reasoning, enable intelligent
game playing, classification, and machines learning. ANNs are a powerful breed of
supervised learning systems that are capable of receiving large inputs and adapt through
continuous refinement of its goal on each neuron in the network. It is this adaptive nature
that allows the network to learn in real time which sets ANNs apart from other AI
machine learning techniques.

## Dependencies
- Python 2.7
- SciPy
- NumPy

## Run
Start training with the following command, this will create a `mnist.pklz` file to save the results in.
Use this file on load to test new hand writing digits (using feedforward)
```sh
python -i main.py 

```
```python
annRecognitionNetwork = net.RecognitionNetwork(mnistplzData, True)
recognizedNumber = annRecognitionNetwork.feedforward(image_MatrixBuffer);
```
