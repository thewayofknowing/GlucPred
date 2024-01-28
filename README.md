# Glucose Prediction using Federated Graph Neural Networks 

The codebase is based on the architecture by [Zhang et al](https://ieeexplore.ieee.org/document/8903252), Spatio-Temporal Graph Attention Networks for time series prediction. The model is deployed in a Federated Learning setup, where the dataset is distributed among various clients and learning happens on the client's end. One global model is then learnt which is used for test time predictions. 

### How to run the code
#### Setting up a virtual environment

We recommend setting up a virtual environment and installing the requisite libraries using the following commands :
 
 ```
  python3 -m venv env
  source env/bin/activate
  python3 -m pip install -r requirements.txt
  ```

#### Training
The model training happens with the [<code>train.py</code>](train.py) file. 

```
  python3 train.py --clients 3 --epochs 100 --local_epochs 10 --lr 0.005 
  ```

You can change the default behaviour by using the following arguments :

* ```--clients:```        Number of Simulated Clients, Default: 3
* ```--epochs:```         Number of rounds of training (Global). Default: 100
* ```--local_epochs:```   Number of rounds of training (Local Client). Default: 10
* ```--lr:```             Learning rate, set to 0.005 by default.
* ```--decay:```          Weight Decay, set to 0 by default

#### Testing
The selected model can be used to get test performance metrics and plot the prediction curves using the following command :
```
  python3 test.py --model runs/model_512_0.0003_1e-06.pth --plot 1
  ```
It accepts the following arguments: 
* ```--model:```       Model file location (Required)
* ```--plot:```        Whether to plot a sample prediction graph, Default: 0

#### Credits
Code snippets have been accumulated from the following repositories and credited for the same:
* [Federated Learning (PyTorch)](https://github.com/AshwinRJ/Federated-Learning-PyTorch/tree/master)
* [ST-GAT Traffic Prediction](https://github.com/jswang/stgat_traffic_prediction/tree/main)


