# Video-Anomaly-Detection

## Shared Doc for Meeting

https://docs.google.com/document/d/1gzF_MX23FH7GLBQUonJaQUyu-8-0hov1BHAe2_zI-zY/

## TO DO

**benchmarking protocol:**

- annotation format

  - How to define anomaly? Are there any subclass-level labels (e.g., robbery with gun) within superclass-level labels (e.g., anomalous activity)?

  - How to label the anomaly? For example, what does granularity look like?

  - What is the distribution, e.g., in terms of activity labels?

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

>Existing mainstream VAD techniques are based on either the one-class formulation, which assumes all training data are normal, or weakly-supervised, which requires only video-level normal/anomaly labels.

>Unsupervised models learned solely from normal videos are applicable to any testing anomalies but suffer from a high false positive rate. In contrast, weakly supervised methods are effective in detecting known anomalies but could fail in an open world.

|   Name   |  GitHub  |  Dataset  | Supervised |  Grained  |  Details  | Evaluation |
| :------: | :------: | :-------: | :--------: | :-------: | :-------: | :--------: |
| [MIST: Multiple Instance Self-Training Framework for Video Anomaly Detection](https://arxiv.org/pdf/2104.01633.pdf) | [MIST_VAD](https://github.com/fjchange/MIST_VAD) | UCF-Crime, ShanghaiTech | Weakly | Fine | MIST can refine task-specific discriminative representations with only video-level annotations. MIST is composed of 1) a multiple instance clip-level pseudo label generator, and 2) a self-guided attention boosted feature encoder to automatically focus on anomalous regions in frames while extracting task-specific representations. We adopt a self-training scheme to optimize both components and finally obtain a task-specific feature encoder. | AUC-ROC, False Alarm Rate (FAR) |
| [Anomaly Detection in Video via Self-Supervised and Multi-Task Learning](https://arxiv.org/pdf/2011.07491.pdf) | [AED-SSMTL](https://github.com/lilygeorgescu/AED-SSMTL) | CUHK Avenue, ShanghaiTech, UCSD Ped2 | Self | Coarse | No anomalous events when training! Use self-supervised and multi-task learning at the object level. First utilize a pre-trained detector for objects. Then train a 3D CNN by jointly learning multiple proxy tasks: three self-supervised ((i) discrimination of forward/backward moving objects (arrow of time), (ii) discrimination of objects in consecutive/intermittent frames (motion irregularity) and (iii) reconstruction of object-specific appearance information) and one based on knowledge distillation. | Frame-level AUC, Region-based detection criterion (RBDC), Track-based detection criterion (TBDC) |
| [Self-Supervised Predictive Convolutional Attentive Block for Anomaly Detection](https://arxiv.org/pdf/2111.09099.pdf) | [SSPCAB](https://github.com/ristea/sspcab) | MVTec AD; CHUK Avenue, ShanghaiTech | Self | Fine | Only learn from normal training samples, but evaluate on both normal and abnormal test samples. SSPCAB is a novel neural block composed of a masked convolutional layer and a channel attention module, which predicts a masked region in the convolutional receptive field. It's trained in a self-supervised manner, via a reconstruction loss of its own. | AUROC; AUC, RBDC, TBDC |
| [Bayesian Nonparametric Submodular Video Partition for Robust Anomaly Detection](https://arxiv.org/pdf/2203.12840.pdf) | [BN-SVP](https://github.com/ritmininglab/bn-svp) | ShanghaiTech, CUHK Avenue, UCF-Crime | Weakly | Coarse | A Bayesian nonparametric submodularity diversified MIL model for robust video anomaly detection in practical settings that involve outlier and multimodal scenarios. | Frame-level AUC-ROC, Top-k (Avg Top-k) |
| [Generative Cooperative Learning for Unsupervised Video Anomaly Detection](https://arxiv.org/pdf/2203.03962.pdf) | -- | UCF-Crime, ShanghaiTech | Un | Coarse | An unsupervised anomaly detection approach (GCL) using unlabeled training videos, which can be deployed without providing any manual annotations. GCL exploits the low frequency of anomalies towards building a cross-supervision between a generator and a discriminator. Both networks get trained in a cooperative fashion, thereby allowing unsupervised learning. | Frame-level AUC-ROC |
| [Dance with Self-Attention: A New Look of Conditional Random Fields on Anomaly Detection in Videos](https://openaccess.thecvf.com/content/ICCV2021/papers/Purwanto_Dance_With_Self-Attention_A_New_Look_of_Conditional_Random_Fields_ICCV_2021_paper.pdf) | -- | UCF-Crime, ShanghaiTech | Weakly | Coarse | The network learns multi-scale CNN features with a relation-aware feature extractor. Then a CRF is employed to model the relationships of the global and local features with a newly devised self-attention. With such a combination, short- and long-term temporal dependencies across frames can be learned. And a contrastive multi-instance learning scheme can broaden the gap between the normal and abnormal instance. | Frame-level AUC, FAR |
| [A Hybrid Video Anomaly Detection Framework via Memory-Augmented Flow Reconstruction and Flow-Guided Frame Prediction](https://arxiv.org/pdf/2108.06852.pdf) | [HF2-VAD](https://github.com/LiUzHiAn/hf2vad) | UCSD Ped2, CUHK Avenue, ShanghaiTech | Un | Coarse | HF2-VAD is a Hybrid framework that integrates Flow Reconstruction and Frame Prediction. First, network of Multi-Level Memory modules in an Autoencoder with Skip Connections can memorize normal patterns for optical flow reconstruction, so that abnormal events can be identified with larger flow reconstruction errors. Then Conditional Variational Autoencoder captures the correlation between video frame and optical flow, to predict the next frame given several previous frames. Reconstruction can influence prediction quality! | Frame-level AUROC |
| [Weakly-supervised Video Anomaly Detection with Robust Temporal Feature Magnitude Learning](https://arxiv.org/pdf/2101.10030.pdf) | [RTFM](https://github.com/tianyu0207/RTFM) | ShanghaiTech, UCF-Crime, XD-Violence, UCSD Peds | Weakly | Coarse | RTFM learns a temporal feature magnitude mapping function that 1) detects the rare abnormal snippets from videos containing many normal ones, and 2) guarantees a large margin between normal and abnormal snippets. RTFM also adapts dilated convolutions and self-attention mechanisms to capture long- and short-range temporal dependencies. | Frame-level AUC, Average Precision (AP) |
| [Towards Open Set Video Anomaly Detection](https://arxiv.org/pdf/2208.11113.pdf) | [Towards-OpenVAD](https://github.com/YUZ128pitt/Towards-OpenVAD) | XD-Violence, UCF-Crime, ShanghaiTech | Weakly | Coarse | Both known anomalies and novel ones exist in testing! We develop a method for OpenVAD problem by integrating evidential deep learning and normalizing flows into a MIL framework. It inherits advantages of both unsupervised NFs and weakly-supervised MIL framework. | AUC-ROC, AUC-PR |
| [Self-Supervised Sparse Representation for Video Anomaly Detection](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136730727.pdf) | [S3R](https://github.com/louisYen/S3R) | ShanghaiTech, UCF-Crime, XD-Violence | Weakly | Coarse | S3R framework models the feature-level anomaly through the offline trained dictionary and self-supervised learning. With the learned dictionary, S3R facilitates two coupled modules, en-Normal (reconstruct normal-event features) and de-Normal (filter out the normal-event features). The self-supervised techniques also enable generating samples of pseudo normal/anomaly to train the anomaly detector. | Frame-level AUC, Average Precision (AP) |
| [Video Anomaly Detection by Solving Decoupled Spatio-Temporal Jigsaw Puzzles](https://arxiv.org/pdf/2207.10172.pdf) | [Jigsaw-VAD](https://github.com/gdwang08/Jigsaw-VAD) | UCSD Ped2, CUHK Avenue, ShanghaiTech | Self | Fine | Solve spatio-temporal jigsaw puzzles, which is cast as a multi-label fine-grained classification problem. Our advantages: 1) spatio-temporal jigsaw puzzles are decoupled in terms of spatial and temporal dimensions, responsible for capturing highly discriminative appearance and motion features, respectively; 2) full permutations are used to provide abundant jigsaw puzzles, allowing the network to distinguish subtle spatio-temporal differences between normal and abnormal; and 3) the pretext task is tackled in an end-to-end manner without relying on pre-trained models. | Frame-level AUROC |
| [CLAWS: Clustering Assisted Weakly Supervised Learning with Normalcy Suppression for Anomalous Event Detection](https://arxiv.org/pdf/2011.12077.pdf) | [CLAWS](https://github.com/xaggi/claws_eccv) | UCF-Crime, ShanghaiTech | Weakly | Coarse | Contribution: 1) a random batch based training procedure to reduce inter-batch correlation, 2) a normalcy suppression mechanism to minimize anomaly scores of the normal regions of a video by taking into account the overall information available in one training batch, and 3) a clustering distance based loss to contribute towards mitigating the label noise and to produce better anomaly representations by encouraging our model to generate distinct normal and anomalous clusters. | Frame-level AUC, False Alarm Rate (FAR) |
| [Few-shot Scene-adaptive Anomaly Detection](https://arxiv.org/pdf/2007.07843.pdf) | [Few-shot Scene-adaptive](https://github.com/yiweilu3/Few-shot-Scene-adaptive-Anomaly-Detection) | ShanghaiTech, UCF-Crime, UCSD Peds, CUHK Avenue, UR Fall | Un | Coarse | Given a few frames captured from a previously unseen scene, try to produce an anomaly detection model specifically adapted to this scene. During meta-training, we have access to videos from multiple scenes. We use these videos to construct a collection of tasks, where each task is a few-shot scene-adaptive anomaly detection task. | AUC-ROC |
| [Clustering Driven Deep Autoencoder for Video Anomaly Detection](https://cse.buffalo.edu/~jsyuan/papers/2020/ECCV2020-2341-CameraReady.pdf) | -- | UCSD Ped2, CUHK Avenue, ShanghaiTech | Un | Coarse | We design a convolution autoencoder architecture to separately capture spatial and temporal informative representation. The spatial part reconstructs the last individual frame, while the temporal part takes consecutive frames as input and RGB difference as output to simulate the generation of optical flow. Besides, we design a deep k-means cluster to force the appearance and the motion encoder to extract common factors of variation within the dataset. We use both reconstruction error and cluster distance to evaluate the anomaly. | Frame-level AUC-ROC |
| [Few-Shot Fast-Adaptive Anomaly Detection](https://openreview.net/pdf?id=bAE1y8wG-ng) | -- | MVTec AD; ShanghaiTech, UCSD Peds, CUHK Avenue |  | Fine | Energy Based Model (EBM) learns to associate low energies to correct values and higher energies to incorrect values. At its core, the EBM employs Langevin Dynamics (LD) in generating these incorrect samples based on an iterative optimization procedure, alleviating the intractable problem of modeling the world of anomalies. Then, in order to avoid training an anomaly detector for every task, we utilize an adaptive sparse coding layer. This update of the sparse coding layer needs to be achievable with just a few shots. A meta learning scheme simulates such a few shot setting during training. | mIoU; Frame-level AUC-ROC |
| [Attribute-based Representations for Accurate and Interpretable Video Anomaly Detection](https://arxiv.org/pdf/2212.00789.pdf) | [Accurate-Interpretable-VAD](https://github.com/talreiss/Accurate-Interpretable-VAD) | UCSD Ped2, CUHK Avenue, ShanghaiTech | Self | Coarse | In every frame, we represent each object using velocity and pose representations, which is followed by density-based anomaly scoring. Combine interpretable attribute-based representations with implicit deep representation. | Frame-level AUROC |
| [Real-world Video Anomaly Detection by Extracting Salient Features in Videos](https://arxiv.org/pdf/2209.06435.pdf) | -- | UCF-Crime, ShanghaiTech, XD-Violence | Weakly | Coarse | While it is indeed important to learn all segments together, the temporal orders are irrelevant to high accuracy. So do not use MIL framework, but instead use a lightweight model with a self-attention mechanism to automatically extract features that are important for determining normal/abnormal from all input segments. | Frame-level AUC-ROC, Average Precision (AP) |

## Performance Comparison

### UCF-Crime

|   Model   | Conference | Supervised |  Feature  |  AUC (%)  |  FAR (%)  |
| :-------: | :--------: | :--------: | :-------: | :-------: | :-------: |
| [MIST](https://arxiv.org/pdf/2104.01633.pdf) | CVPR 21 | Weakly | I3D RGB | 82.30 | 0.13 |
| [BN-SVP](https://arxiv.org/pdf/2203.12840.pdf) | CVPR 22 | Weakly | I3D | 83.39 | -- |
| [GCL](https://arxiv.org/pdf/2203.03962.pdf) | CVPR 22 | Un | ResNext | 71.04 | -- |
| [CRF](https://openaccess.thecvf.com/content/ICCV2021/papers/Purwanto_Dance_With_Self-Attention_A_New_Look_of_Conditional_Random_Fields_ICCV_2021_paper.pdf) | ICCV 21 | Weakly | Relation-aware RGB | 85.00 | 0.024 |
| [RTFM](https://arxiv.org/pdf/2101.10030.pdf) | ICCV 21 | Weakly | I3D RGB | 84.30 | -- |
| [CLAWS](https://arxiv.org/pdf/2011.12077.pdf) | ECCV 20 | Weakly | C3D RGB | 83.03 | -- |
| [OpenVAD](https://arxiv.org/pdf/2208.11113.pdf) | ECCV 22 | Weakly | -- | 80.14 | -- |
| [S3R](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136730727.pdf) | ECCV 22 | Weakly | I3D | 85.99 | -- |
| [Real-World](https://arxiv.org/pdf/2209.06435.pdf) | ICIP 22 | Weakly | I3D RGB | 84.91 | -- |

### ShanghaiTech

|   Model   | Conference | Supervised |  Feature  |  AUC (%)  |  FAR (%)  |
| :-------: | :--------: | :--------: | :-------: | :-------: | :-------: |
| [MIST](https://arxiv.org/pdf/2104.01633.pdf) | CVPR 21 | Weakly | I3D RGB | 94.83 | 0.05 |
| [SSMTL](https://arxiv.org/pdf/2011.07491.pdf) | CVPR 21 | Self | -- | 83.5 | -- |
| [SSPCAB](https://arxiv.org/pdf/2111.09099.pdf) | CVPR 22 | Self | -- | Micro 83.6, Macro 89.5 | -- |
| [BN-SVP](https://arxiv.org/pdf/2203.12840.pdf) | CVPR 22 | Weakly | C3D | 96.00 | -- |
| [GCL](https://arxiv.org/pdf/2203.03962.pdf) | CVPR 22 | Un | ResNext | 78.93 | -- |
| [CRF](https://openaccess.thecvf.com/content/ICCV2021/papers/Purwanto_Dance_With_Self-Attention_A_New_Look_of_Conditional_Random_Fields_ICCV_2021_paper.pdf) | ICCV 21 | Weakly | Relation-aware RGB | 96.85 | 0.004 |
| [HF2-VAD](https://arxiv.org/pdf/2108.06852.pdf) | ICCV 21 | Un | -- | 76.2 | -- |
| [RTFM](https://arxiv.org/pdf/2101.10030.pdf) | ICCV 21 | Weakly | I3D RGB | 97.21 | -- |
| [CLAWS](https://arxiv.org/pdf/2011.12077.pdf) | ECCV 20 | Weakly | C3D RGB | 89.67 | 0.12 |
| [Few-Shot](https://arxiv.org/pdf/2007.07843.pdf) | ECCV 20 | Un | -- | 77.9 | -- |
| [Cluster](https://cse.buffalo.edu/~jsyuan/papers/2020/ECCV2020-2341-CameraReady.pdf) | ECCV 20 | Un | -- | 73.3 | -- |
| [OpenVAD](https://arxiv.org/pdf/2208.11113.pdf) | ECCV 22 | Weakly | -- | 93.99 | -- |
| [S3R](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136730727.pdf) | ECCV 22 | Weakly | I3D | 97.48 | -- |
| [Jigsaw](https://arxiv.org/pdf/2207.10172.pdf) | ECCV 22 | Self | -- | 84.3 | -- |
| [Accurate-Interpretable](https://arxiv.org/pdf/2212.00789.pdf) | -- | Self | -- | Micro 85.9, Macro 89.6 | -- |
| [Real-World](https://arxiv.org/pdf/2209.06435.pdf) | ICIP 22 | Weakly | I3D RGB | 95.72 | -- |

### CUHK Avenue

|   Model   | Conference | Supervised |  Feature  |  AUC (%)  |
| :-------: | :--------: | :--------: | :-------: | :-------: |
| [SSMTL](https://arxiv.org/pdf/2011.07491.pdf) | CVPR 21 | Self | -- | 86.9 |
| [SSPCAB](https://arxiv.org/pdf/2111.09099.pdf) | CVPR 22 | Self | -- | Micro 92.9, Macro 93.5 |
| [BN-SVP](https://arxiv.org/pdf/2203.12840.pdf) | CVPR 22 | Weakly | C3D | 80.87 |
| [HF2-VAD](https://arxiv.org/pdf/2108.06852.pdf) | ICCV 21 | Un | -- | 91.1 |
| [Few-Shot](https://arxiv.org/pdf/2007.07843.pdf) | ECCV 20 | Un | -- | 85.8 |
| [Cluster](https://cse.buffalo.edu/~jsyuan/papers/2020/ECCV2020-2341-CameraReady.pdf) | ECCV 20 | Un | -- | 86.0 |
| [Jigsaw](https://arxiv.org/pdf/2207.10172.pdf) | ECCV 22 | Self | -- | 92.2 |
| [Accurate-Interpretable](https://arxiv.org/pdf/2212.00789.pdf) | -- | Self | -- | Micro 93.3, Macro 96.2 |

### UCSD Ped2

|   Model   | Conference | Supervised |  Feature  |  AUC (%)  |
| :-------: | :--------: | :--------: | :-------: | :-------: |
| [SSMTL](https://arxiv.org/pdf/2011.07491.pdf) | CVPR 21 | Self | -- | 92.4 |
| [HF2-VAD](https://arxiv.org/pdf/2108.06852.pdf) | ICCV 21 | Un | -- | 99.3 |
| [RTFM](https://arxiv.org/pdf/2101.10030.pdf) | ICCV 21 | Weakly | I3D RGB | 98.6 |
| [Few-Shot](https://arxiv.org/pdf/2007.07843.pdf) | ECCV 20 | Un | -- | 96.2 |
| [Cluster](https://cse.buffalo.edu/~jsyuan/papers/2020/ECCV2020-2341-CameraReady.pdf) | ECCV 20 | Un | -- | 96.5 |
| [Jigsaw](https://arxiv.org/pdf/2207.10172.pdf) | ECCV 22 | Self | -- | 99.0 |
| [Accurate-Interpretable](https://arxiv.org/pdf/2212.00789.pdf) | -- | Self | -- | Micro 99.1, Macro 99.9 |

### XD-Violence

|   Model   | Conference | Supervised |  Feature  |  AP (%)  |
| :-------: | :--------: | :--------: | :-------: | :------: |
| [RTFM](https://arxiv.org/pdf/2101.10030.pdf) | ICCV 21 | Weakly | I3D RGB | 77.81 |
| [OpenVAD](https://arxiv.org/pdf/2208.11113.pdf) | ECCV 22 | Weakly | -- | AUC-PR (%): 69.61 |
| [S3R](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136730727.pdf) | ECCV 22 | Weakly | I3D | 80.26 |
| [Real-World](https://arxiv.org/pdf/2209.06435.pdf) | ICIP 22 | Weakly | I3D RGB + VGGish | 82.89 |
