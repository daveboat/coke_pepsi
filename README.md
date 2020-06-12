# Toy model for coke/pepsi classification

This is a toy model using tensorflow 2.0 for soft drink classification between coke and pepsi cans and bottles, meant for demonstrative purposes. Images were collected from Google image search.

```model.py``` contains the model definition
```train.py``` is the main training file. Hyperparameters are hardcoded.
```convert_tflite.py``` runs a script which converts the saved model to a tflite file

Because only ~70 images are used, all training data is included in the repo. The trained model, in saved_model format, and in tflite format, are also included in this repo.

### Training log

Below is a training log for 50 epochs:

```
Epoch 1, Loss: 0.7484723925590515, Accuracy: 60.000003814697266, Test Loss: 0.7023062109947205, Test Accuracy: 53.333335876464844
Epoch 2, Loss: 0.7400655746459961, Accuracy: 53.333335876464844, Test Loss: 0.6836736798286438, Test Accuracy: 53.333335876464844
Epoch 3, Loss: 0.6902545094490051, Accuracy: 48.88888931274414, Test Loss: 0.6944419741630554, Test Accuracy: 46.66666793823242
Epoch 4, Loss: 0.6925384402275085, Accuracy: 48.88888931274414, Test Loss: 0.6771779656410217, Test Accuracy: 60.000003814697266
Epoch 5, Loss: 0.6740819811820984, Accuracy: 55.55555725097656, Test Loss: 0.656101644039154, Test Accuracy: 53.333335876464844
Epoch 6, Loss: 0.6563219428062439, Accuracy: 66.66667175292969, Test Loss: 0.6177938580513, Test Accuracy: 60.000003814697266
Epoch 7, Loss: 0.3819253146648407, Accuracy: 84.44444274902344, Test Loss: 0.28830215334892273, Test Accuracy: 93.33333587646484
Epoch 8, Loss: 0.4367862939834595, Accuracy: 82.22222137451172, Test Loss: 0.44751861691474915, Test Accuracy: 80.0
Epoch 9, Loss: 0.4134106934070587, Accuracy: 88.8888931274414, Test Loss: 0.26702770590782166, Test Accuracy: 93.33333587646484
Epoch 10, Loss: 0.2825040817260742, Accuracy: 93.33333587646484, Test Loss: 0.22878111898899078, Test Accuracy: 100.0
Epoch 11, Loss: 0.1658862829208374, Accuracy: 97.77777862548828, Test Loss: 0.154488667845726, Test Accuracy: 93.33333587646484
Epoch 12, Loss: 0.24487002193927765, Accuracy: 93.33333587646484, Test Loss: 0.3519488275051117, Test Accuracy: 86.66666412353516
Epoch 13, Loss: 0.25000637769699097, Accuracy: 84.44444274902344, Test Loss: 0.47134777903556824, Test Accuracy: 80.0
Epoch 14, Loss: 0.20285126566886902, Accuracy: 91.11111450195312, Test Loss: 0.13951818645000458, Test Accuracy: 100.0
Epoch 15, Loss: 0.19104258716106415, Accuracy: 97.77777862548828, Test Loss: 0.16895979642868042, Test Accuracy: 93.33333587646484
Epoch 16, Loss: 0.05736321955919266, Accuracy: 100.0, Test Loss: 0.28337863087654114, Test Accuracy: 93.33333587646484
Epoch 17, Loss: 0.15777914226055145, Accuracy: 93.33333587646484, Test Loss: 0.09994355589151382, Test Accuracy: 93.33333587646484
Epoch 18, Loss: 0.03653470799326897, Accuracy: 97.77777862548828, Test Loss: 0.0982230007648468, Test Accuracy: 93.33333587646484
Epoch 19, Loss: 0.09954585880041122, Accuracy: 93.33333587646484, Test Loss: 0.02471625804901123, Test Accuracy: 100.0
Epoch 20, Loss: 0.06976577639579773, Accuracy: 95.55555725097656, Test Loss: 0.16978169977664948, Test Accuracy: 93.33333587646484
Epoch 21, Loss: 0.0688447579741478, Accuracy: 95.55555725097656, Test Loss: 0.09541510790586472, Test Accuracy: 100.0
Epoch 22, Loss: 0.16378824412822723, Accuracy: 91.11111450195312, Test Loss: 0.012079094536602497, Test Accuracy: 100.0
Epoch 23, Loss: 0.0491531565785408, Accuracy: 100.0, Test Loss: 0.08242759108543396, Test Accuracy: 100.0
Epoch 24, Loss: 0.07256065309047699, Accuracy: 93.33333587646484, Test Loss: 0.014410089701414108, Test Accuracy: 100.0
Epoch 25, Loss: 0.054267726838588715, Accuracy: 97.77777862548828, Test Loss: 0.014437764883041382, Test Accuracy: 100.0
Epoch 26, Loss: 0.014979813247919083, Accuracy: 100.0, Test Loss: 0.009632942266762257, Test Accuracy: 100.0
Epoch 27, Loss: 0.035872772336006165, Accuracy: 97.77777862548828, Test Loss: 0.005337590351700783, Test Accuracy: 100.0
Epoch 28, Loss: 0.08602884411811829, Accuracy: 97.77777862548828, Test Loss: 0.004631286486983299, Test Accuracy: 100.0
Epoch 29, Loss: 0.04752058908343315, Accuracy: 97.77777862548828, Test Loss: 0.3451300859451294, Test Accuracy: 86.66666412353516
Epoch 30, Loss: 0.10340915620326996, Accuracy: 95.55555725097656, Test Loss: 0.04207894578576088, Test Accuracy: 100.0
Epoch 31, Loss: 0.05640830472111702, Accuracy: 100.0, Test Loss: 0.03375643491744995, Test Accuracy: 100.0
Epoch 32, Loss: 0.0075826495885849, Accuracy: 100.0, Test Loss: 0.0023693253751844168, Test Accuracy: 100.0
Epoch 33, Loss: 0.003913493826985359, Accuracy: 100.0, Test Loss: 0.07982256263494492, Test Accuracy: 93.33333587646484
Epoch 34, Loss: 0.0014794141752645373, Accuracy: 100.0, Test Loss: 0.04896518215537071, Test Accuracy: 93.33333587646484
Epoch 35, Loss: 0.0031059186439961195, Accuracy: 100.0, Test Loss: 0.00011615739640546963, Test Accuracy: 100.0
Epoch 36, Loss: 0.00037152308505028486, Accuracy: 100.0, Test Loss: 0.14112089574337006, Test Accuracy: 93.33333587646484
Epoch 37, Loss: 0.0011328798718750477, Accuracy: 100.0, Test Loss: 0.14973342418670654, Test Accuracy: 93.33333587646484
Epoch 38, Loss: 0.00011892782640643418, Accuracy: 100.0, Test Loss: 0.23382961750030518, Test Accuracy: 93.33333587646484
Epoch 39, Loss: 0.0011291601695120335, Accuracy: 100.0, Test Loss: 0.0011561453575268388, Test Accuracy: 100.0
Epoch 40, Loss: 0.0015463603194803, Accuracy: 100.0, Test Loss: 0.04765807464718819, Test Accuracy: 100.0
Epoch 41, Loss: 0.0007707966724410653, Accuracy: 100.0, Test Loss: 0.06439944356679916, Test Accuracy: 93.33333587646484
Epoch 42, Loss: 0.001605882658623159, Accuracy: 100.0, Test Loss: 0.16670644283294678, Test Accuracy: 93.33333587646484
Epoch 43, Loss: 0.0012983831111341715, Accuracy: 100.0, Test Loss: 0.000518298358656466, Test Accuracy: 100.0
Epoch 44, Loss: 0.00048058066749945283, Accuracy: 100.0, Test Loss: 0.08776756376028061, Test Accuracy: 93.33333587646484
Epoch 45, Loss: 0.0002973123046103865, Accuracy: 100.0, Test Loss: 0.0013820636086165905, Test Accuracy: 100.0
Epoch 46, Loss: 0.0006491061067208648, Accuracy: 100.0, Test Loss: 0.048505742102861404, Test Accuracy: 93.33333587646484
Epoch 47, Loss: 8.846184937283397e-05, Accuracy: 100.0, Test Loss: 0.05628155171871185, Test Accuracy: 93.33333587646484
Epoch 48, Loss: 0.0008474474889226258, Accuracy: 100.0, Test Loss: 0.032403670251369476, Test Accuracy: 100.0
Epoch 49, Loss: 0.0015330800088122487, Accuracy: 100.0, Test Loss: 0.016267243772745132, Test Accuracy: 100.0
Epoch 50, Loss: 1.5821033230167814e-05, Accuracy: 100.0, Test Loss: 0.49899783730506897, Test Accuracy: 93.33333587646484
```