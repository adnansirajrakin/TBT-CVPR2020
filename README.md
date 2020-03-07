# Trageted-Bit-Trojan-TBT
Code: Targeted Bit Trojan Attack (TBT).
In the repository, we provide a sample code to implement the targeted  bit trojan attack. The paper can be find in the arxiv link https://arxiv.org/abs/1909.05193. The link to get the associated dependencies can be found in https://drive.google.com/open?id=1FC3XssrjgbI5m-BFniebUY0AiDDPU6e8. Two steps are required to test the Code:

1. Run "TBT.py" file to generate the trigger then inject the trojan into a pre-trained resnet-18 model which can be found in the google drive link.

One can tune the values of wb and target class by tuning 'wb' and 'targets' variable. In order to change the TAP change 'start' and 'end' variables which would indicate the start point and end point of the trigger across one dimension.

2. Run "Test.py" would test the effectiveness of the trigger and also count the number of parameters modified and bits-fliped. A sample model is provided to directly run the "test.py" code on torjan inserted Resnet-18 model with 'TAP' of 9.6 % and 'wb' of 150 on target class 2.

3. Feel free to play with the following parameters to generate different Triggers including pararmeters:

'wb'= Tunes the number of weights the attacker modifies.
'targets'= sets the target class for the attack.
'start' = start pixel of the trigger for each channel.
'end'= end pixel of the trigger for each channel.
