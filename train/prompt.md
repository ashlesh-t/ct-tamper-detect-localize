ROLE: ML engineer
Overview: This is an code for localisation of tampered Lung CT basically an tumour site is removed using GAN manipulation and standard diffusion models, the area its cutcube and did many layers of additinal edits which gave tthis tampeed data !

OBjective: An robust methodology to make precise localisation of tampered area , and also i do have an classification layer that will tall removed or inserted , and my aim i s to localise removed section in the CT slice, i guess patientwise data split is not that necessary , as we do slice wise localisation

Data Overview: currently i do pass data like TB(Real CT scans without tumours) and FB (false benign basically tumour site is removed ) and these 2 things are not related at all , like they are not of same patients , also i dont have mapped slices like (Ream and its tampered), they areof completelt different , even though they are Lung CT , the true bening is an data set and this tampered sis another and we dont have anythings isimilar

I have few questions:
1. In pretrain section do we need to train on real CT without tumout or with tumour?
2. what id we use Silu or Gelu or leky relu insted of relu and apply dreat slyle sota
3. we do have csv , use it effectivee

below is the metrics that i currently get which ar every very bad
================================================================================
Advanced CT Forensic Localization
================================================================================
Device: cuda
Loading CT Removal data...
Using CSV: /kaggle/input/ct-removal-processed/2/data_v2.csv
CSV loaded: 1844 rows
Sample rows:
   type                                               path    x    y  \
0   FB  1.3.6.1.4.1.14519.5.2.1.6279.6001.931383239747...  295  338   
1   FB  1.3.6.1.4.1.14519.5.2.1.6279.6001.931383239747...  154  286   
2   FB  1.3.6.1.4.1.14519.5.2.1.6279.6001.935683764293...  370  371   
3   FB  1.3.6.1.4.1.14519.5.2.1.6279.6001.935683764293...  147  358   
4   FB  1.3.6.1.4.1.14519.5.2.1.6279.6001.939152384493...  422  363   

        scanner  cur_slice  
0  LightSpeed16         44  
1  LightSpeed16         79  
2  LightSpeed16        118  
3  LightSpeed16        129  
4  Sensation 16         67  
Found 1688 total removal slices
Found 449 removal patients
Found 14527 TB (real) samples for pretraining
Train: 1102 slices from 314 patients
Val: 586 slices from 135 patients

Initializing model...
Total parameters: 8,871,527
Pretraining on 1102 real + 1102 fake samples (balanced, no duplication)

================================================================================
PHASE 1: Improved Self-Supervised Pretraining
================================================================================
Features: Two-view augmentation + SimCLR loss + Cosine annealing
Loading pretrain checkpoint from /kaggle/working/ct_forensic_output/pretrain_checkpoint.pth
Resuming pretraining from epoch 60, best loss: 0.2338
Pretrain history saved to: /kaggle/working/ct_forensic_output/pretrain_history.json

Improved pretraining complete! Encoder learned robust real vs fake representations.
Pretrained model saved to: /kaggle/working/ct_forensic_output/pretrained_encoder.pth

================================================================================
PHASE 2: Supervised Fine-Tuning
================================================================================
/usr/local/lib/python3.11/dist-packages/torch/optim/lr_scheduler.py:62: UserWarning: The verbose parameter is deprecated. Please use get_last_lr() to access the learning rate.
  warnings.warn(

Sanity check: First 5 train mask sums
Mask 0 sum: 0.00
Mask 1 sum: 303.59
Mask 2 sum: 0.00
Mask 3 sum: 129.85
Mask 4 sum: 6419.46

Starting fine-tuning...

Epoch 1/100
--------------------------------------------------------------------------------
Training: 100%|██████████| 138/138 [01:46<00:00,  1.30it/s, loss=0.9932, dice=0.0000, aux=0.204]
Validation: 100%|██████████| 74/74 [00:16<00:00,  4.51it/s, loss=0.9736, dice=0.0000]

Epoch 1 Summary:
 Train - Loss: 1.0114, Dice: 0.0535, IoU: 0.0521, Prec: 0.0117, Rec: 0.0788
 Val - Loss: 0.9355, Dice: 0.0377, IoU: 0.0266, Prec: 0.0745, Rec: 0.0189
 LR: 1.00e-04
 *** New best model! Dice: 0.0377 ***

Epoch 2/100
--------------------------------------------------------------------------------
Training: 100%|██████████| 138/138 [01:45<00:00,  1.31it/s, loss=1.0093, dice=0.0000, aux=0.011]
Validation: 100%|██████████| 74/74 [00:16<00:00,  4.46it/s, loss=0.9756, dice=0.0358]

Epoch 2 Summary:
 Train - Loss: 0.9533, Dice: 0.0562, IoU: 0.0298, Prec: 0.0394, Rec: 0.2479
 Val - Loss: 0.9312, Dice: 0.0681, IoU: 0.0377, Prec: 0.0440, Rec: 0.2598
 LR: 1.00e-04
 *** New best model! Dice: 0.0681 ***

Epoch 3/100
--------------------------------------------------------------------------------
Training: 100%|██████████| 138/138 [01:45<00:00,  1.30it/s, loss=1.0643, dice=0.0000, aux=0.168]
Validation: 100%|██████████| 74/74 [00:16<00:00,  4.52it/s, loss=0.9650, dice=0.0323]

Epoch 3 Summary:
 Train - Loss: 0.9453, Dice: 0.0743, IoU: 0.0402, Prec: 0.0486, Rec: 0.3381
 Val - Loss: 0.9367, Dice: 0.0664, IoU: 0.0369, Prec: 0.0548, Rec: 0.1919
 LR: 1.00e-04

Epoch 4/100
--------------------------------------------------------------------------------
Training: 100%|██████████| 138/138 [01:45<00:00,  1.31it/s, loss=0.9028, dice=0.1431, aux=0.174]
Validation: 100%|██████████| 74/74 [00:16<00:00,  4.51it/s, loss=1.0224, dice=0.0059]

Epoch 4 Summary:
 Train - Loss: 0.9400, Dice: 0.0795, IoU: 0.0436, Prec: 0.0545, Rec: 0.2765
 Val - Loss: 0.9441, Dice: 0.0620, IoU: 0.0404, Prec: 0.0399, Rec: 0.1142
 LR: 1.00e-04

Epoch 5/100
--------------------------------------------------------------------------------
Training: 100%|██████████| 138/138 [01:45<00:00,  1.31it/s, loss=0.9946, dice=0.0000, aux=0.009]
Validation: 100%|██████████| 74/74 [00:16<00:00,  4.53it/s, loss=0.9839, dice=0.0364]

Epoch 5 Summary:
 Train - Loss: 0.9294, Dice: 0.0865, IoU: 0.0471, Prec: 0.0608, Rec: 0.2911
 Val - Loss: 0.9424, Dice: 0.0634, IoU: 0.0411, Prec: 0.0432, Rec: 0.1110
 LR: 1.00e-04

Epoch 6/100
--------------------------------------------------------------------------------
Training: 100%|██████████| 138/138 [01:45<00:00,  1.31it/s, loss=0.9790, dice=0.0000, aux=0.007]
Validation: 100%|██████████| 74/74 [00:16<00:00,  4.54it/s, loss=1.0434, dice=0.0000]

Epoch 6 Summary:
 Train - Loss: 0.9252, Dice: 0.0916, IoU: 0.0509, Prec: 0.0683, Rec: 0.2697
 Val - Loss: 0.9508, Dice: 0.0569, IoU: 0.0317, Prec: 0.0458, Rec: 0.1255
 LR: 1.00e-04

Epoch 7/100
--------------------------------------------------------------------------------
Training: 100%|██████████| 138/138 [01:45<00:00,  1.31it/s, loss=1.0392, dice=0.0000, aux=0.061]
Validation: 100%|██████████| 74/74 [00:16<00:00,  4.60it/s, loss=1.0396, dice=0.0000]

Epoch 7 Summary:
 Train - Loss: 0.9254, Dice: 0.0971, IoU: 0.0542, Prec: 0.0712, Rec: 0.3200
 Val - Loss: 0.9430, Dice: 0.0610, IoU: 0.0345, Prec: 0.0477, Rec: 0.1331
 LR: 1.00e-04

Epoch 8/100
--------------------------------------------------------------------------------
Training: 100%|██████████| 138/138 [01:45<00:00,  1.31it/s, loss=0.7799, dice=0.1892, aux=0.009]
Validation: 100%|██████████| 74/74 [00:16<00:00,  4.54it/s, loss=1.0142, dice=0.0538]

Epoch 8 Summary:
 Train - Loss: 0.9196, Dice: 0.0963, IoU: 0.0537, Prec: 0.0727, Rec: 0.2725
 Val - Loss: 0.9453, Dice: 0.0585, IoU: 0.0323, Prec: 0.0413, Rec: 0.1669
 LR: 1.00e-04

Epoch 9/100
--------------------------------------------------------------------------------
Training: 100%|██████████| 138/138 [01:45<00:00,  1.31it/s, loss=1.0122, dice=0.0000, aux=0.008]
Validation: 100%|██████████| 74/74 [00:16<00:00,  4.54it/s, loss=1.0565, dice=0.0000]

Epoch 9 Summary:
 Train - Loss: 0.9218, Dice: 0.0985, IoU: 0.0550, Prec: 0.0722, Rec: 0.2833
 Val - Loss: 0.9521, Dice: 0.0569, IoU: 0.0314, Prec: 0.0466, Rec: 0.1182
 LR: 1.00e-04

Epoch 10/100
--------------------------------------------------------------------------------
Training: 100%|██████████| 138/138 [01:45<00:00,  1.31it/s, loss=0.7722, dice=0.2547, aux=0.048]
Validation: 100%|██████████| 74/74 [00:16<00:00,  4.60it/s, loss=0.9467, dice=0.1002]

Epoch 10 Summary:
 Train - Loss: 0.9109, Dice: 0.1078, IoU: 0.0600, Prec: 0.0811, Rec: 0.3222
 Val - Loss: 0.9429, Dice: 0.0565, IoU: 0.0305, Prec: 0.0425, Rec: 0.1573
 LR: 1.00e-04

Epoch 11/100
--------------------------------------------------------------------------------
Training: 100%|██████████| 138/138 [01:45<00:00,  1.31it/s, loss=1.0138, dice=0.0430, aux=0.056]
Validation: 100%|██████████| 74/74 [00:16<00:00,  4.60it/s, loss=0.9892, dice=0.0917]

Epoch 11 Summary:
 Train - Loss: 0.9159, Dice: 0.1037, IoU: 0.0579, Prec: 0.0778, Rec: 0.2997
 Val - Loss: 0.9492, Dice: 0.0712, IoU: 0.0449, Prec: 0.0445, Rec: 0.1462
 LR: 1.00e-04
 *** New best model! Dice: 0.0712 ***

Epoch 12/100
--------------------------------------------------------------------------------
Training: 100%|██████████| 138/138 [01:45<00:00,  1.31it/s, loss=0.8830, dice=0.0705, aux=0.009]
Validation: 100%|██████████| 74/74 [00:16<00:00,  4.56it/s, loss=1.0397, dice=0.0027]

Epoch 12 Summary:
 Train - Loss: 0.9056, Dice: 0.1132, IoU: 0.0647, Prec: 0.0864, Rec: 0.3203
 Val - Loss: 0.9541, Dice: 0.0666, IoU: 0.0423, Prec: 0.0471, Rec: 0.0975
 LR: 1.00e-04

Epoch 13/100
--------------------------------------------------------------------------------
Training: 100%|██████████| 138/138 [01:45<00:00,  1.31it/s, loss=0.8167, dice=0.1226, aux=0.010]
Validation: 100%|██████████| 74/74 [00:16<00:00,  4.55it/s, loss=1.0384, dice=0.0000]

Epoch 13 Summary:
 Train - Loss: 0.9082, Dice: 0.1117, IoU: 0.0624, Prec: 0.0832, Rec: 0.3343
 Val - Loss: 0.9544, Dice: 0.0619, IoU: 0.0396, Prec: 0.0456, Rec: 0.0845
 LR: 1.00e-04

Epoch 14/100
--------------------------------------------------------------------------------
Training: 100%|██████████| 138/138 [01:45<00:00,  1.31it/s, loss=0.9335, dice=0.0000, aux=0.005]
Validation: 100%|██████████| 74/74 [00:16<00:00,  4.46it/s, loss=1.0909, dice=0.0000]

Epoch 14 Summary:
 Train - Loss: 0.8994, Dice: 0.1232, IoU: 0.0697, Prec: 0.0919, Rec: 0.2926
 Val - Loss: 0.9638, Dice: 0.0858, IoU: 0.0596, Prec: 0.0524, Rec: 0.1090
 LR: 1.00e-04
 *** New best model! Dice: 0.0858 ***

Epoch 15/100
--------------------------------------------------------------------------------
Training: 100%|██████████| 138/138 [01:45<00:00,  1.31it/s, loss=0.8638, dice=0.2187, aux=0.105]
Validation: 100%|██████████| 74/74 [00:16<00:00,  4.62it/s, loss=0.9758, dice=0.0885]

Epoch 15 Summary:
 Train - Loss: 0.8945, Dice: 0.1222, IoU: 0.0692, Prec: 0.0924, Rec: 0.3169
 Val - Loss: 0.9462, Dice: 0.0636, IoU: 0.0354, Prec: 0.0480, Rec: 0.1399
 LR: 1.00e-04

Epoch 16/100
--------------------------------------------------------------------------------
Training: 100%|██████████| 138/138 [01:45<00:00,  1.31it/s, loss=1.0158, dice=0.0000, aux=0.016]
Validation: 100%|██████████| 74/74 [00:16<00:00,  4.60it/s, loss=1.0019, dice=0.0692]

Epoch 16 Summary:
 Train - Loss: 0.8956, Dice: 0.1187, IoU: 0.0671, Prec: 0.0889, Rec: 0.3226
 Val - Loss: 0.9541, Dice: 0.0864, IoU: 0.0596, Prec: 0.0463, Rec: 0.1204
 LR: 1.00e-04
 *** New best model! Dice: 0.0864 ***

Epoch 17/100
--------------------------------------------------------------------------------
Training: 100%|██████████| 138/138 [01:45<00:00,  1.31it/s, loss=0.9401, dice=0.0477, aux=0.008]
Validation: 100%|██████████| 74/74 [00:16<00:00,  4.55it/s, loss=0.9791, dice=0.1043]

Epoch 17 Summary:
 Train - Loss: 0.9031, Dice: 0.1117, IoU: 0.0625, Prec: 0.0864, Rec: 0.3110
 Val - Loss: 0.9545, Dice: 0.0651, IoU: 0.0413, Prec: 0.0433, Rec: 0.1101
 LR: 1.00e-04

Epoch 18/100
--------------------------------------------------------------------------------
Training: 100%|██████████| 138/138 [01:45<00:00,  1.31it/s, loss=0.8642, dice=0.0000, aux=0.008]
Validation: 100%|██████████| 74/74 [00:16<00:00,  4.54it/s, loss=1.0254, dice=0.0702]

Epoch 18 Summary:
 Train - Loss: 0.8882, Dice: 0.1347, IoU: 0.0778, Prec: 0.1028, Rec: 0.3145
 Val - Loss: 0.9635, Dice: 0.0622, IoU: 0.0399, Prec: 0.0521, Rec: 0.0792
 LR: 1.00e-04

Epoch 19/100
--------------------------------------------------------------------------------
Training: 100%|██████████| 138/138 [01:45<00:00,  1.31it/s, loss=1.0228, dice=0.0125, aux=0.128]
Validation: 100%|██████████| 74/74 [00:16<00:00,  4.56it/s, loss=1.0481, dice=0.0000]

Epoch 19 Summary:
 Train - Loss: 0.8905, Dice: 0.1296, IoU: 0.0738, Prec: 0.0953, Rec: 0.3358
 Val - Loss: 0.9585, Dice: 0.1571, IoU: 0.1348, Prec: 0.0442, Rec: 0.0827
 LR: 1.00e-04
 *** New best model! Dice: 0.1571 ***

Epoch 20/100
--------------------------------------------------------------------------------
Training: 100%|██████████| 138/138 [01:45<00:00,  1.30it/s, loss=0.8807, dice=0.1828, aux=0.065]
Validation: 100%|██████████| 74/74 [00:16<00:00,  4.54it/s, loss=0.9886, dice=0.0768]

Epoch 20 Summary:
 Train - Loss: 0.8894, Dice: 0.1284, IoU: 0.0734, Prec: 0.0988, Rec: 0.3423
 Val - Loss: 0.9470, Dice: 0.0603, IoU: 0.0334, Prec: 0.0464, Rec: 0.1347
 LR: 1.00e-04

Epoch 21/100
--------------------------------------------------------------------------------
Training: 100%|██████████| 138/138 [01:45<00:00,  1.30it/s, loss=0.9901, dice=0.0000, aux=0.003]
Validation: 100%|██████████| 74/74 [00:16<00:00,  4.62it/s, loss=1.0823, dice=0.0000]

Epoch 21 Summary:
 Train - Loss: 0.8932, Dice: 0.1287, IoU: 0.0731, Prec: 0.0975, Rec: 0.3272
 Val - Loss: 0.9691, Dice: 0.0616, IoU: 0.0398, Prec: 0.0451, Rec: 0.0742
 LR: 1.00e-04

Epoch 22/100
--------------------------------------------------------------------------------
Validation: 100%|██████████| 74/74 [00:16<00:00,  4.50it/s, loss=1.0496, dice=0.0000]aux=0.009]

Epoch 22 Summary:
 Train - Loss: 0.8866, Dice: 0.1297, IoU: 0.0742, Prec: 0.0954, Rec: 0.3117
 Val - Loss: 0.9534, Dice: 0.0680, IoU: 0.0433, Prec: 0.0437, Rec: 0.1153
 LR: 1.00e-04

Epoch 23/100
--------------------------------------------------------------------------------
Training: 100%|██████████| 138/138 [01:45<00:00,  1.30it/s, loss=0.9355, dice=0.0508, aux=0.009]
Validation: 100%|██████████| 74/74 [00:16<00:00,  4.59it/s, loss=1.0403, dice=0.0350]

Epoch 23 Summary:
 Train - Loss: 0.8763, Dice: 0.1499, IoU: 0.0870, Prec: 0.1143, Rec: 0.3584
 Val - Loss: 0.9529, Dice: 0.0743, IoU: 0.0474, Prec: 0.0477, Rec: 0.1246
 LR: 1.00e-04

Epoch 24/100
--------------------------------------------------------------------------------
Training: 100%|██████████| 138/138 [01:45<00:00,  1.31it/s, loss=0.8233, dice=0.1338, aux=0.011]
Validation: 100%|██████████| 74/74 [00:16<00:00,  4.59it/s, loss=1.0232, dice=0.0449]

Epoch 24 Summary:
 Train - Loss: 0.8747, Dice: 0.1486, IoU: 0.0857, Prec: 0.1137, Rec: 0.3746
 Val - Loss: 0.9657, Dice: 0.1174, IoU: 0.0947, Prec: 0.0481, Rec: 0.0741
 LR: 1.00e-04

Epoch 25/100
--------------------------------------------------------------------------------
Training: 100%|██████████| 138/138 [01:45<00:00,  1.30it/s, loss=0.9238, dice=0.0000, aux=0.009]
Validation: 100%|██████████| 74/74 [00:16<00:00,  4.57it/s, loss=1.0786, dice=0.0000]

Epoch 25 Summary:
 Train - Loss: 0.8770, Dice: 0.1465, IoU: 0.0845, Prec: 0.1114, Rec: 0.3296
 Val - Loss: 0.9634, Dice: 0.0658, IoU: 0.0420, Prec: 0.0439, Rec: 0.1087
 LR: 1.00e-04

Epoch 26/100
--------------------------------------------------------------------------------
Training: 100%|██████████| 138/138 [01:45<00:00,  1.30it/s, loss=0.9638, dice=0.0000, aux=0.010]
Validation: 100%|██████████| 74/74 [00:16<00:00,  4.57it/s, loss=0.9493, dice=0.1546]

Epoch 26 Summary:
 Train - Loss: 0.8740, Dice: 0.1480, IoU: 0.0854, Prec: 0.1156, Rec: 0.3350
 Val - Loss: 0.9519, Dice: 0.0600, IoU: 0.0328, Prec: 0.0457, Rec: 0.1391
 LR: 1.00e-04

Epoch 27/100
--------------------------------------------------------------------------------
Training: 100%|██████████| 138/138 [01:45<00:00,  1.30it/s, loss=0.8793, dice=0.1400, aux=0.051]
Validation: 100%|██████████| 74/74 [00:16<00:00,  4.44it/s, loss=0.9729, dice=0.1141]

Epoch 27 Summary:
 Train - Loss: 0.8658, Dice: 0.1579, IoU: 0.0920, Prec: 0.1213, Rec: 0.3650
 Val - Loss: 0.9545, Dice: 0.0576, IoU: 0.0315, Prec: 0.0438, Rec: 0.1251
 LR: 1.00e-04

Epoch 28/100
--------------------------------------------------------------------------------
Training: 100%|██████████| 138/138 [01:45<00:00,  1.30it/s, loss=0.6846, dice=0.2318, aux=0.015]
Validation: 100%|██████████| 74/74 [00:16<00:00,  4.52it/s, loss=1.0755, dice=0.0000]

Epoch 28 Summary:
 Train - Loss: 0.8649, Dice: 0.1595, IoU: 0.0921, Prec: 0.1222, Rec: 0.3759
 Val - Loss: 0.9634, Dice: 0.0628, IoU: 0.0402, Prec: 0.0414, Rec: 0.0920
 LR: 1.00e-04

Epoch 29/100
--------------------------------------------------------------------------------
Training: 100%|██████████| 138/138 [01:45<00:00,  1.31it/s, loss=0.8409, dice=0.2383, aux=0.005]
Validation: 100%|██████████| 74/74 [00:16<00:00,  4.54it/s, loss=1.0993, dice=0.0000]

Epoch 29 Summary:
 Train - Loss: 0.8665, Dice: 0.1590, IoU: 0.0926, Prec: 0.1267, Rec: 0.3577
 Val - Loss: 0.9993, Dice: 0.1400, IoU: 0.1263, Prec: 0.0478, Rec: 0.0273
 LR: 1.00e-04




dont give code just discuss pros and cons of current aparroach? is it too complicated , do i need to do any data visualisation , any better apprach ? refer any papers or anythign , and discuss