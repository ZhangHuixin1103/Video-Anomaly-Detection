# Video-Anomaly-Detection

## Shared Doc for Meeting

https://docs.google.com/document/d/1gzF_MX23FH7GLBQUonJaQUyu-8-0hov1BHAe2_zI-zY/

## TO DO

**benchmarking protocol:**

- annotation format

- - How to define anomaly? Are there any subclass-level labels (e.g., robbery with gun) within superclass-level labels (e.g., anomalous activity)?

- - How to label the anomaly? For example, what does granularity look like?

- - What is the distribution, e.g., in terms of activity labels.

- training setup

- - Note that anomaly detection in videos has a setup that we train only on normal events but strive to detect anomalous events during testing. In other words, the anomalous events are not seen during training. So one type of methods is to learn normal event distribution and find out-of-distribution testing examples as the anomaly.

- - Another setup can be that we train a model on both labeled normal and anomalous events, and hope the trained model to detect the same types of anomalous events during testing.

- - Extending the above, we can do open-set anomaly detection where we have normal and some known anomalous labeled events for training, but must detect unknown anomalous events in testing. Yet, we hope the trained model will be able to detect/recognize unknown/unseen anomalous events.

- testing setup

- - (very coarse level) video clip level (or temporal segments) recognition

- - (coarse level) frame-wise recognition w.r.t normal-vs-anomalous recognition, or even K-way classification (different types of normal and anomalous events)

- - (fine-level) anomalous region segmentation/detection, i.e., bounding box level detection or mask level segmentation; in terms of what level of granularity, e.g., K-way vs. binary (normal-abnormal)

- evaluation metrics

- - precision, accuracy, mean average precision, recall, top-K accuracy, etc


