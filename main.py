from PIL import Image
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
import numpy as np
import random
import string
import os


class GaussianNBPredictor:
    __train_data_X = []
    __train_data_Y = []
    __test_data_X = []
    __test_data_Y = []
    __images = []
    __image = None
    __color_labels = {}

    def __init__(self, sample_name: str) -> None:
        self.__images = [Image.open('./Dataset/' + pic).convert('L').load() for pic in os.listdir('./Dataset')]
        self.__image = Image.open('./Dataset/' + sample_name)
        self.__color_labels = {0: (0, 0, 255), 1: (255, 0, 0), 2: (0, 255, 0), 3: (255, 255, 0), 4: (255, 255, 255),
                               5: (130, 255, 240)}

    @staticmethod
    def __is_train_pixel(x: int, y: int) -> bool:
        # Water
        if (135 <= x <= 185 and 300 <= y <= 350) or (340 <= x <= 360 and 600 <= y <= 630):
            return True
        # Mountain
        if 850 <= x <= 900 and 275 <= y <= 350:
            return True
        # Agriculture
        if 410 <= x <= 445 and 210 <= y <= 270:
            return True
        # Arid
        if 250 <= x <= 295 and 195 <= y <= 270:
            return True
        # Snow
        if (609 <= x <= 616 and 375 <= y <= 381) or (837 <= x <= 843 and 837 <= y <= 843) or (
                596 <= x <= 614 and 370 <= y <= 380) or (614 <= x <= 620 and 374 <= y <= 386):
            return True
        # Salt
        if (260 <= x <= 270 and 733 <= y <= 740) or (228 <= x <= 239 and 204 <= y <= 211):
            return True
        return False

    @staticmethod
    def __label_train_pixel(x: int, y: int) -> int:
        # Water
        if (135 <= x <= 185 and 300 <= y <= 350) or (340 <= x <= 360 and 600 <= y <= 630):
            return 0
        # Mountain
        if 850 <= x <= 900 and 275 <= y <= 350:
            return 1
        # Agriculture
        if 410 <= x <= 445 and 210 <= y <= 270:
            return 2
        # Arid
        if 250 <= x <= 295 and 195 <= y <= 270:
            return 3
        # Snow
        if (609 <= x <= 616 and 375 <= y <= 381) or (837 <= x <= 843 and 837 <= y <= 843) or (
                596 <= x <= 614 and 370 <= y <= 380) or (614 <= x <= 620 and 374 <= y <= 386):
            return 4
        # Salt
        if (260 <= x <= 270 and 733 <= y <= 740) or (228 <= x <= 239 and 204 <= y <= 211):
            return 5

    @staticmethod
    def __is_test_pixel(x: int, y: int) -> bool:
        # Water
        if (165 <= x <= 205 and 160 <= y <= 200) or (255 <= x <= 270 and 705 <= y <= 715):
            return True
        # Mountain
        if 205 <= x <= 255 and 300 <= y <= 360:
            return True
        # Agriculture
        if 350 <= x <= 380 and 310 <= y <= 340:
            return True
        # Arid
        if 280 <= x <= 310 and 320 <= y <= 350:
            return True
        # Snow
        if 837 <= x <= 843 and 837 <= y <= 843:
            return True
        # Salt
        if 260 <= x <= 270 and 733 <= y <= 740:
            return True
        return False

    @staticmethod
    def __label_test_pixel(x: int, y: int) -> int:
        # Water
        if (165 <= x <= 205 and 160 <= y <= 200) or (255 <= x <= 270 and 705 <= y <= 715):
            return 0
        # Mountain
        if 205 <= x <= 255 and 300 <= y <= 360:
            return 1
        # Agriculture
        if 350 <= x <= 380 and 310 <= y <= 340:
            return 2
        # Arid
        if 280 <= x <= 310 and 320 <= y <= 350:
            return 3
        # Snow
        if 837 <= x <= 843 and 837 <= y <= 843:
            return 4
        # Salt
        if 260 <= x <= 270 and 733 <= y <= 740:
            return 5

    def __collect_pixel_values_from_images(self, x: int, y: int) -> list:
        values = []
        for image in self.__images:
            values.append(image[x, y])
        return values

    def __fill_data_axis(self) -> None:
        for i in range(self.__image.size[0]):
            for j in range(self.__image.size[1]):
                if self.__is_train_pixel(i, j):
                    self.__train_data_X.append(self.__collect_pixel_values_from_images(i, j))
                    self.__train_data_Y.append(self.__label_train_pixel(i, j))
                if self.__is_test_pixel(i, j):
                    self.__test_data_X.append(self.__collect_pixel_values_from_images(i, j))
                    self.__test_data_Y.append(self.__label_test_pixel(i, j))

    def __save_output_image(self) -> None:
        path = os.getcwd() + '\\Output'
        if os.path.isdir(path):
            self.__image.save(
                path + '\\' + ''.join(random.choices(string.ascii_uppercase + string.digits, k=10)) + '.jpg')
            return
        os.mkdir(path)
        self.__image.save(path + '\\' + ''.join(random.choices(string.ascii_uppercase + string.digits, k=10)) + '.jpg')

    def __make_output_image(self, model: GaussianNB) -> None:
        pixels = self.__image.load()
        for i in range(self.__image.size[0]):
            for j in range(self.__image.size[1]):
                if pixels[i, j] == (0, 0, 0):
                    continue
                elif self.__is_train_pixel(i, j):
                    pixels[i, j] = self.__color_labels[self.__label_train_pixel(i, j)]
                elif self.__is_test_pixel(i, j):
                    pixels[i, j] = self.__color_labels[self.__label_test_pixel(i, j)]
                else:
                    pixels[i, j] = self.__color_labels[
                        model.predict([self.__collect_pixel_values_from_images(i, j)])[0]]
        self.__image.show()
        self.__save_output_image()

    def __create_confusion_matrix(self, actual_data: list, predicted_data: np.ndarray) -> np.ndarray:
        return confusion_matrix(actual_data, predicted_data, labels=list(self.__color_labels.keys()))

    def driver(self) -> None:
        print('Detecting dataset pixels...')
        self.__fill_data_axis()
        print('Fitting the Gaussian Naive Bayes model...')
        gnb = GaussianNB().fit(self.__train_data_X, self.__train_data_Y)
        print('Predicting the area covering type...')
        self.__make_output_image(gnb)
        print('Prediction completed.')
        print()
        print('Test data confusion matrix:')
        print(self.__create_confusion_matrix(self.__test_data_Y, gnb.predict(self.__test_data_X)))
        print('Train data confusion matrix:')
        print(self.__create_confusion_matrix(self.__train_data_Y, gnb.predict(self.__train_data_X)))


if __name__ == '__main__':
    gnb_predictor = GaussianNBPredictor('pic-7.jpg')
    gnb_predictor.driver()
