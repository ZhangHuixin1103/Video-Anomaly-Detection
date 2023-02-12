# Video-Anomaly-Detection

## Zoom Link

https://tamu.zoom.us/j/9876585850

## Shared Doc for Meeting

https://docs.google.com/document/d/1gzF_MX23FH7GLBQUonJaQUyu-8-0hov1BHAe2_zI-zY/

## TO DO

**benchmarking protocol:**

- annotation format

  - How to define anomaly? Are there any subclass-level labels (e.g., robbery with gun) within superclass-level labels (e.g., anomalous activity)?

  - How to label the anomaly? For example, what does granularity look like?

  - What is the distribution, e.g., in terms of activity labels.

- training setup

  - Note that anomaly detection in videos has a setup that we train only on normal events but strive to detect anomalous events during testing. In other words, the anomalous events are not seen during training. So one type of methods is to learn normal event distribution and find out-of-distribution testing examples as the anomaly.

  - Another setup can be that we train a model on both labeled normal and anomalous events, and hope the trained model to detect the same types of anomalous events during testing.

  - Extending the above, we can do [open-set anomaly detection](https://arxiv.org/pdf/2208.11113.pdf) where we have normal and some known anomalous labeled events for training, but must detect unknown anomalous events in testing. Yet, we hope the trained model will be able to detect/recognize unknown/unseen anomalous events.

- testing setup

  - (very coarse level) video clip level (or temporal segments) recognition

  - (coarse level) frame-wise recognition w.r.t normal-vs-anomalous recognition, or even K-way classification (different types of normal and anomalous events)

  - (fine-level) anomalous region segmentation/detection, i.e., bounding box level detection or mask level segmentation; in terms of what level of granularity, e.g., K-way vs. binary (normal-abnormal)

- evaluation metrics

  - AUC-ROC

  - precision, accuracy, mean average precision, recall, top-K accuracy, etc

## Dataset

|   Name   |   Link   |  # of videos  |  Length  | Avg Frames |   Label   |  Anomalies  |
| :------: | :------: | :-----------: | :------: | :------------: | :-------: | :---------: |
| [ShanghaiTech](https://svip-lab.github.io/dataset/campus_dataset.html) | [GitHub](https://github.com/StevenLiuWen/ano_pred_cvpr2018) | 437 (330 / 107) | -- | 726 | Pixel-level; Training videos all normal | 13 scenes, 130 abnormal events |
| [UCSD Peds1&2](http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm) | [Paper](http://www.svcl.ucsd.edu/publications/conference/2010/cvpr2010/cvpr_anomaly_2010.pdf) | 70 (34 / 36), 28 (16 / 12) | 5 min | 200, 163 | Frame-level, Partial Pixel-level | Biker, Skater, Cart, Wheelchair, People walking across a walkway / in the grass|
| [CUHK Avenue](http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html) | [Paper](https://www.cv-foundation.org/openaccess/content_iccv_2013/papers/Lu_Abnormal_Event_Detection_2013_ICCV_paper.pdf) | 37 (16 / 21) | 30 min | 839 | Pixel-level; Training videos all normal | Strange action, Wrong direction, Abnormal object |
| [UBnormal](https://github.com/lilygeorgescu/UBnormal/) | [Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Acsintoae_UBnormal_New_Benchmark_for_Supervised_Open-Set_Video_Anomaly_Detection_CVPR_2022_paper.pdf) | 543 (268 / 64 / 211) | 2 hr | 436 | Pixel-level; Include disjoint sets of anomaly types in training and testing | Crawl, Dance, Run injured, Steal, Fight, Sleep, Smoke, Lay down, Car accident, Seizure|
| [UCF-Crime](https://www.crcv.ucf.edu/research/real-world-anomaly-detection-in-surveillance-videos/) | [GitHub](https://github.com/WaqasSultani/AnomalyDetectionCVPR2018) | 1900 (1610 / 290) | 128 hr | 7247 | Weakly labeled, Video-level; Frame labels available only for testing videos | Abuse, Arrest, Arson, Assault, Road Accident, Burglary, Explosion, Fighting, Robbery, Shooting, Stealing, Shoplifting, Vandalism |
| [UBI-Fights](http://socia-lab.di.ubi.pt/EventDetection/) | [GitHub](https://github.com/DegardinBruno/human-self-learning-anomaly) | 1000 | 80 hr | -- | Frame-level | Fight |
| [XD-Violence](https://roc-ng.github.io/XD-Violence/) | [GitHub](https://github.com/Roc-Ng/XDVioDet) | 4754 | 217 hr | -- | Weakly labeled; Assign multi labels (1 ≤ #labels ≤ 3) to each video owing to co-occurrence | Abuse, Car Accident, Explosion, Fighting, Riot, Shooting |

## Paper

|   Name   |  GitHub  |  Dataset  | Supervised |  Grained  |  Details  | Evaluation |
| :------: | :------: | :-------: | :--------: | :-------: | :-------: | :--------: |
| [MIST: Multiple Instance Self-Training Framework for Video Anomaly Detection](https://arxiv.org/pdf/2104.01633.pdf) | [MIST_VAD](https://github.com/fjchange/MIST_VAD) | UCF-Crime, ShanghaiTech | Weak | Fine | MIST can refine task-specific discriminative representations with only video-level annotations. MIST is composed of 1) a multiple instance clip-level pseudo label generator, and 2) a self-guided attention boosted feature encoder to automatically focus on anomalous regions in frames while extracting task-specific representations. We adopt a self-training scheme to optimize both components and finally obtain a task-specific feature encoder. | AUC-ROC, False Alarm Rate (FAR) |
| [Anomaly Detection in Video via Self-Supervised and Multi-Task Learning](https://arxiv.org/pdf/2011.07491.pdf) | [AED-SSMTL](https://github.com/lilygeorgescu/AED-SSMTL) | CUHK Avenue, ShanghaiTech, UCSD Ped2 | Self | Fine | No anomalous events when training! Use self-supervised and multi-task learning at the object level. First utilize a pre-trained detector for objects. Then train a 3D CNN by jointly learning multiple proxy tasks: three self-supervised ((i) discrimination of forward/backward moving objects (arrow of time), (ii) discrimination of objects in consecutive/intermittent frames (motion irregularity) and (iii) reconstruction of object-specific appearance information) and one based on knowledge distillation. | Frame-level AUC, Region-based detection criterion (RBDC), Trackbased detection criterion (TBDC) |
| [Self-Supervised Predictive Convolutional Attentive Block for Anomaly Detection](https://arxiv.org/pdf/2111.09099.pdf) | [SSPCAB](https://github.com/ristea/sspcab) | MVTec AD; CHUK Avenue, ShanghaiTech | Self | Fine | Only learn from normal training samples, but evaluate on both normal and abnormal test samples. SSPCAB is a novel neural block composed of a masked convolutional layer and a channel attention module, which predicts a masked region in the convolutional receptive field. It's trained in a self-supervised manner, via a reconstruction loss of its own. | AUROC; AUC, RBDC, TBDC |
| [Bayesian Nonparametric Submodular Video Partition for Robust Anomaly Detection](https://arxiv.org/pdf/2203.12840.pdf) | [BN-SVP](https://github.com/ritmininglab/bn-svp) | ShanghaiTech, CUHK Avenue, UCF-Crime |  |  |  |  |
| []() | []() |  |  |  |  |  |
| []() | []() |  |  |  |  |  |
| []() | []() |  |  |  |  |  |
| []() | []() |  |  |  |  |  |
| []() | []() |  |  |  |  |  |
| []() | []() |  |  |  |  |  |
| []() | []() |  |  |  |  |  |
| []() | []() |  |  |  |  |  |
| []() | []() |  |  |  |  |  |
