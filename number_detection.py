import os, sys
import math
import abc
import numpy as np
import cv2
import torch
import sklearn
import joblib
from train_cnn import Net


class Classifier(object):
    def __init__(self, f):
        assert os.path.isfile(f)
        self.model = None
        self.load(f)

    @abc.abstractmethod
    def load(self, f):
        raise NotImplementedError()

    @abc.abstractmethod
    def predict(self, images):
        raise NotImplementedError()


class SklearnClassifier(Classifier):
    def load(self, f):
        self.model = joblib.load(f)

    def preprocess(self, images):
        preprocessed_images = []
        for image in images:
            image = cv2.resize(image, (28, 28))
            image = image.flatten()
            image = np.divide(image, 255.0)
            preprocessed_images.append(image)
        return np.array(preprocessed_images)

    def predict(self, images):
        preprocessed_images = self.preprocess(images)
        return self.model.predict(preprocessed_images)


class PytorchClassifier(Classifier):
    def load(self, f):
        self.model = Net()
        self.model.load_state_dict(torch.load(f))
        self.model.eval()

    def preprocess(self, images):
        preprocessed_images = []
        clahe = cv2.createCLAHE()
        for image in images:
            image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)
            image = np.divide(image, 255.0)
            image = (image - 0.1307) / 0.3081
            preprocessed_images.append(image)
        return torch.Tensor(np.expand_dims(np.array(preprocessed_images), axis=1))

    def predict(self, images):
        preprocessed_images = self.preprocess(images)
        probabilities = (
            self.model.forward(preprocessed_images).detach().numpy().tolist()
        )
        predictions = [probs.index(max(probs)) for probs in probabilities]
        for prediction, probability in zip(predictions, probabilities):
            print(prediction, probability)
        return predictions


class NumberDetection(object):
    def __init__(self, img, classifier):
        assert isinstance(img, np.ndarray), isinstance(classifier, Classifier)
        self.img = img
        self.contours = None
        self.binary_img = None
        self.debug_img = img.copy()
        self.classifier = classifier

    def get_binary_image(self, lower=[0, 0, 0], upper=[50, 50, 50]):
        self.binary_img = cv2.inRange(self.img, np.array(lower), np.array(upper))
        kernel = np.ones((7, 7), np.uint8)
        self.binary_img = cv2.dilate(self.binary_img, kernel, iterations=1)
        return self.binary_img

    def get_contours(self):
        self.contours, hierarchies = cv2.findContours(
            self.binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        contours_to_remove = []
        for i, h in enumerate(hierarchies[0]):
            # If contour has a parent remove it
            if h[-1] != -1:
                contours_to_remove.append(i)
        self.contours = [
            c for i, c in enumerate(self.contours) if i not in contours_to_remove
        ]
        return self.contours

    def draw_classification(self, cx, cy, shape_class):
        cv2.putText(
            self.debug_img,
            shape_class,
            (cx, cy - 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

    def draw_bounding_box(self, x, y, w, h):
        cv2.rectangle(self.debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    def find_numbers(self, offset=10):
        self.get_binary_image()
        self.get_contours()
        crops, center_locations = [], []
        for n, cnt in enumerate(self.contours):
            cv2.drawContours(self.debug_img, [cnt], 0, (0, 0, 255), 2)
            area = cv2.contourArea(cnt)
            print(area)
            x, y, w, h = cv2.boundingRect(cnt)
            self.draw_bounding_box(x, y, w, h)
            crop = self.binary_img[
                y - offset : y + h + offset, x - offset : x + w + offset
            ]
            if crop.shape[1]:
                crops.append(crop)
                center_locations.append([x + (w / 2), y + (h / 2)])
                print(center_locations[-1])
        return crops, center_locations

    def detect_numbers(self):
        crops, center_locations = self.find_numbers()
        classifications = self.classifier.predict(crops)
        for classification, center_location in zip(classifications, center_locations):
            self.draw_classification(
                int(center_location[0]), int(center_location[1]), str(classification)
            )
        return classifications, center_locations


if __name__ == "__main__":
    img = cv2.imread("numbers.jpg")
    img = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
    # classifier = SklearnClassifier("mnist_mlp.p")
    classifier = PytorchClassifier("mnist_cnn.pt")
    number_detection = NumberDetection(img, classifier)
    number_detection.detect_numbers()

    combined = np.hstack(
        (
            number_detection.debug_img,
            cv2.cvtColor(number_detection.binary_img, cv2.COLOR_GRAY2BGR),
        )
    )
    cv2.imshow("combined", combined)
    cv2.waitKey(0)
