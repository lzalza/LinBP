# Code for A Theoretical View of Linear Backpropagation and Its Convergence

## Files:
1. sim: The simulation experiments
> two_layer_attack.py: Adversarial attack on two layer student-teacher network
> one_layer_sgd.py: SGD training for one layer student-teacher network
2. attack_cifar: Adversarial attack on CIFAR-10
(Implement on the base of https://github.com/Kaminyou/PGD-Implemented-Adversarial-attack-on-CIFAR10)
3. model_training_mnist: Model training on MNIST
4. model_training_cifar10: Model training on CIFAR-10
(Implement on the base of https://github.com/kuangliu/pytorch-cifar)
5. adv_train: Adversarial training on CIFAR-10
(Implement on the base of https://github.com/ndb796/Pytorch-Adversarial-Training-CIFAR)

## Requirements:
python == 3.6
torch == 1.8
torchvision == 0.4.2
numpy == 1.16.4

## Preparation:
1. MNIST data should be copied into model_training_mnist/mnist.
2. CIFAR-10 data should be copied into model_training_cifar10/dataset and adv_train/data.
3. The pretrained model (except roubst WRN model) should be downloaded from https://github.com/huyvnphan/PyTorch_CIFAR10.
4. The pretrained robust WRN model can be found on RobustBench and the download links is https://drive.google.com/uc?id=1lBVvLG6JLXJgQP2gbsTxNHl6s3YAopqk&export=download. 
5. Move the pretrained models to attack_cifar/models/state_dicts.
6. In adversarial attack experiments, CIFAR-10 test images should be classified and put into the right folders in attack_cifar/imgs, e.g., imgs/airplane, imgs/automobile...


## Execution:
1. Simulation experiments:
cd sim
For adversarial attack python: two_layer_attack.py
For SGD training: python one_layer_sgd.py
2. Adversarial attack on CIFAR-10:
cd attack_cifar
python main.py -I imgs -T PGD -M resnet50 -e 0.0313 -k 5 -g 0 (-lin) #To attack resnet50 using LinBP (or not).
3. Model training on MNIST:
cd model_training_mnist
python MLP.py -lr 0.001 -e 50 (-lin) #To train MLP using LinBP (or not).
python LeNet.py -lr 0.005 -e 50 (-lin) #To train LeNet-5 using LinBP (or not).
4. Model training on CIFAR-10:
cd model_training_cifar10
python main.py --lr 0.005 --epoch 100 --model mobilenet_v2 (-lin) # To trian mobilenet_v2 using LinBP (or not).
5. Adversarial training on CIFAR-10:
cd adv_train
python pgd_adversarial_training.py --lr 0.01 -K 5 --epoch 100 --model mobilenet_v2 (-lin_t) (-lin_a) #Adversarial training on mobilenet_v2 using LinBP for training (or not) and LinBP for attack (or not).


