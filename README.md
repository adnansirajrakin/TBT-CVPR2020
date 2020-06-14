#  TBT: Targeted Neural Network Attack with Bit Trojan - CVPR2020

Our algorithm efficiently generates a trigger specifically designed to locate vulnerable bits of DNN weights. 
Once the attacker flips these vulnerable bits, the network still operates with normal inference accuracy with benign input. 
However, when the attacker activates the trigger by embedding it with any input, the network is forced to classify all inputs 
into one target class. 
Highlights:
1. TBT can transform a fully functional DNN to classify 92% of test images to a target class with just 84 bit-flips out of 88 million 
weight bits on Resnet-18 for CIFAR10 dataset.
2. We require 6 million x less # of parameter modification in comparison to BadNet.
3. We inject Trojan after deployment of the model at the inference Phase through only flipping several Bits.
4. We do not require any Training information or access to training facilities to inject the Trojan.
More Details: https://dfan.engineering.asu.edu/ai-security-targeted-neural-network-attack-with-bit-trojan/


## Description of The Code.
In the repository, we provide a sample code to implement the targeted  bit trojan attack. The paper can be find in the arxiv link https://arxiv.org/abs/1909.05193. The link to get the associated dependencies can be found in https://drive.google.com/open?id=1FC3XssrjgbI5m-BFniebUY0AiDDPU6e8. Two steps are required to test the Code:

1. Run "TBT.py" file to generate the trigger then inject the trojan into a pre-trained resnet-18 model; file name "Resnet18_8bit.pkl" in the google drive link.

One can tune the values of wb and target class by tuning 'wb' and 'targets' variable. In order to change the TAP change 'start' and 'end' variables which would indicate the start point and end point of the trigger across one dimension.

2. Run "Test.py" to test the effectiveness of the trigger and also count the number of parameters modified and bits-fliped. A sample model with a file name "Resnet18_8bit_final_trojan.pkl" is provided in the google drive link to directly run the "test.py" code on torjan inserted Resnet-18 model with a 'TAP' of 9.6 % and 'wb' of 150 on target class 2.

3. Feel free to play with the following parameters to generate different Triggers including pararmeters:

'wb'= Tunes the number of weights the attacker modifies.
'targets'= sets the target class for the attack.
'start' = start pixel of the trigger for each channel.
'end'= end pixel of the trigger for each channel.

Note: We can not directly control the value of 'nb'. It is determined by the algorithm based on the values of 'wb'.
