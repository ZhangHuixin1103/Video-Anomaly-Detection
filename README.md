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
| [ShanghaiTech](https://svip-lab.github.io/dataset/campus_dataset.html) | [GitHub](https://github.com/StevenLiuWen/ano_pred_cvpr2018) | 437 (330 / 107) | -- | 726 | Pixel-level | 13 scenes, 130 abnormal events |
| [UCSD Peds1&2](http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm) | [Paper](http://www.svcl.ucsd.edu/publications/conference/2010/cvpr2010/cvpr_anomaly_2010.pdf) | 70 (34 / 36), 28 (16 / 12) | 5 min | 200, 163 | Frame-level | Biker, Skater, Cart, Wheelchair, People walking across a walkway / in the grass|
| [CUHK Avenue](http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html) | [Paper](https://www.cv-foundation.org/openaccess/content_iccv_2013/papers/Lu_Abnormal_Event_Detection_2013_ICCV_paper.pdf) | 37 (16 / 21) | 30 min | 839 | Train videos capture normal situations; Test videos include both normal and abnormal events | Strange action, Wrong direction, Abnormal object |
| [UBnormal](https://github.com/lilygeorgescu/UBnormal/) | [Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Acsintoae_UBnormal_New_Benchmark_for_Supervised_Open-Set_Video_Anomaly_Detection_CVPR_2022_paper.pdf) | 543 (268 / 64 / 211) | 2 hr | 436 | Pixel level; Include disjoint sets of anomaly types in training and testing | Crawl, Dance, Run injured, Steal, Fight, Sleep, Smoke, Lay down, Car accident, Seizure|
| [UCF-Crime](https://www.crcv.ucf.edu/research/real-world-anomaly-detection-in-surveillance-videos/) | [GitHub](https://github.com/WaqasSultani/AnomalyDetectionCVPR2018) | 1900 | 128 hr | 7247 | Weakly labeled, video-level | Abuse, Arrest, Arson, Assault, Road Accident, Burglary, Explosion, Fighting, Robbery, Shooting, Stealing, Shoplifting, and Vandalism |
| [UBI-Fights](http://socia-lab.di.ubi.pt/EventDetection/) | [GitHub](https://github.com/DegardinBruno/human-self-learning-anomaly) | 1000 | 80 hr | -- | Frame-level | Fight |
| [XD-Violence](https://roc-ng.github.io/XD-Violence/) | [GitHub](https://github.com/Roc-Ng/XDVioDet) |
