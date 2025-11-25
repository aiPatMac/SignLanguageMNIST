## Sign Language MNIST Dataset Description

https://www.kaggle.com/datasets/datamunge/sign-language-mnist

The original MNIST image dataset of handwritten digits is a popular benchmark for image-based machine learning methods, but researchers have renewed efforts to update it and develop drop-in replacements that are more challenging for computer vision and more representative of real-world applications.

As noted in one recent replacement called the **Fashion-MNIST** dataset, the Zalando researchers quoted the startling claim that:

> “Most pairs of MNIST digits (784 total pixels per sample) can be distinguished pretty well by just one pixel.”

To stimulate the community to develop more drop-in replacements, the **Sign Language MNIST** is presented here and follows the same CSV format with labels and pixel values in single rows. The American Sign Language letter database of hand gestures represents a multi-class problem with **24 classes of letters** (excluding **J** and **Z**, which require motion).

---

## Dataset Format

The dataset format is patterned closely after the classic MNIST.
Each training and test case includes:

* A **label (0–25)** mapping to letters **A–Z**

  * *No cases exist for 9 = J or 25 = Z because these require motion*
* **784 pixel values** representing a 28×28 grayscale image (values 0–255)

### Dataset Size

* **Training data:** 27,455 samples
* **Test data:** 7,172 samples

These sizes are roughly half of standard MNIST.

### Data Creation Process

The original hand gesture images came from multiple users repeating gestures against different backgrounds. The dataset was expanded from 1,704 uncropped color images.

The preprocessing pipeline (using **ImageMagick**) included:

* Cropping to hands-only
* Grayscaling
* Resizing
* Augmenting each sample with **50+ variations**

Augmentation techniques included:

* Filters: *Mitchell, Robidoux, Catrom, Spline, Hermite*
* 5% random pixelation
* ±15% brightness/contrast adjustments
* 3° rotation

Because of the small image size, these modifications significantly affect resolution and class separation in interesting, controllable ways.

This dataset was inspired by **Fashion-MNIST** and the machine learning pipeline for gestures by **Sreehari**.

---

## Motivation and Applications

A robust visual recognition algorithm could:

* Provide new benchmarks for modern ML methods (e.g., CNNs)
* Help the deaf and hard-of-hearing communicate more effectively using computer vision applications

According to the **National Institute on Deafness and Other Communication Disorders (NIDCD)**, American Sign Language (ASL) is a complete, complex language and the primary language for many deaf North Americans. It is the leading minority language in the U.S. after Spanish, Italian, German, and French.

One practical application could involve implementing computer vision on inexpensive hardware such as a **Raspberry Pi**, using **OpenCV** and **Text-to-Speech**, enabling improved and automated ASL translation systems.