# Trajectory Prediction for Autonomous Vehicles
ERSP @ UCSD https://ersp.eng.ucsd.edu/

Group Members: Dongwook Kim, Ryan Choi, Sherwin Motlagh, Yingkai Lee

Research Advisor: Jedrzej(Jacob) Kozerawski

SUMO_setup.md contains instructions on setting up SUMO.
Then, you should be able to try out our current version of the script by downloading first.net.xml, first.sumocfg, and test.py. Run test.py.



### To train the MLP model on a GPU:
```
python attention_prediction.py --end_epoch 120 --obs_len 20 --pred_len 30 --gpu 0 --train_batch_size 512 --mlp
```

### To train the LSTM model on a GPU:
```
python attention_prediction.py --end_epoch 120 --obs_len 20 --pred_len 30 --gpu 0 --train_batch_size 512
```
