# Trageted-Bit-Trojan-TBT
Code: Paper Targeted Bit Trojan Attack
In the repository we provide a sample code to implement the targeted  bit trojan attack. The paper can be find in https://arxiv.org/abs/1909.05193 the link provided. The link to get the associated dependencies can be found in https://drive.google.com/open?id=1FC3XssrjgbI5m-BFniebUY0AiDDPU6e8 following folder. Two step to test the Code:

1. TBT.py would generate the trigger then inject the trojan into a pre-trained resnet-18 model can be found in the google drive link.
At the begin one can tune the values of wb and target class by tuning wb and targets variable. In order to change the TAP change start and end variables which would indicate the start point and end point of the trigger across one dimension.

2. test.py would test the effectiveness of the trigger and also count the number of parameters modified and bits-fliped. A sample model is provided to directly run the test.py code on Resnet-18 with TAP 0f 9.6 % and wb of 150 on target class 2.
