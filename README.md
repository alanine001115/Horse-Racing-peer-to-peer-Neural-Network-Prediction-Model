This study aims to advance predictive modeling in betting exchanges by bridging the strengths
of deep sequence modeling and ensemble-based approaches. Convolutional neural networks
(CNN) and long short-term memory (LSTM) networks are employed to capture localized pat-
terns and temporal dependencies in sequential odds and volume data, hoping to identify hidden
pattern that increase the model prediction power. XGBoost is employed to model static covari-
ates and provide interpretability, while CNN-LSTM networks are trained on engineered tabular
features derived from selected variables and optimized using a validation set. Furthermore, the
resulting hybrid system is evaluated on a dataset comprising 1701 horse races conducted on
the Betfair Exchange platform. The results demonstrate that the CNN model consistently
outperforms a tree-based ensemble approach in both the odds movement and arbitrage de-
tection tasks. Moreover, the combination of the two models increase the predictive power in
some tasks. Also, the XGBoost model provided some interpretablity to CNN-LSTM models by
showing feature importance.
