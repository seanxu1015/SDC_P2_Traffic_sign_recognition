import pickle
import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.utils import shuffle


class DataUtils:
    
    def __init__(self, train_file, test_file, batch_size=128, rs=161210, 
                 val_size=0.1, augment=False,normalizer=None,):
        train_images, train_labels = self.readfile(train_file)
        test_images, test_labels = self.readfile(test_file)
        train_num = len(train_labels)
        # train_images, train_labels = self.shuffle(train_images, train_labels, rs)
        # test_images, test_labels = self.shuffle(test_images, test_labels, rs)
        images = np.vstack([train_images, test_images])
        labels = np.hstack([train_labels, test_labels])
        if augment:
            images, labels = self.augment(images, labels)
        images = self.normalization(images, normalizer)
        images, labels = shuffle(images, labels, random_state=rs)
        n1 = int(len(labels) * 0.6)
        n2 = int(len(labels) * 0.8)
        train_images = images[:n1]
        train_labels = labels[:n1]
        val_images = images[n1:n2]
        val_labels = labels[n1:n2]
        test_images = images[n2:]
        test_labels = labels[n2:]
        self.train_images, self.train_labels = self.fit_batches(train_images, 
                train_labels, batch_size)
        self.val_images, self.val_labels = self.fit_batches(val_images, 
                val_labels, batch_size)
        self.test_images, self.test_labels = self.fit_batches(test_images, 
                test_labels,  batch_size)

    def readfile(self, filename):
        with open(filename, 'rb') as f:
            content = pickle.load(f)
        return content['features'], content['labels'] 
        
    def normalization(self, images, normalizer):
        if normalizer is None:
            def normalizer(data):
                return (data - np.mean(data)).astype(np.float32) / np.std(data)
            images[:, :, :, 0] = normalizer(images[:, :, :, 0])
            images[:, :, :, 1] = normalizer(images[:, :, :, 1])
            images[:, :, :, 2] = normalizer(images[:, :, :, 2])
        else:
            images = normalizer(images)
        return images
            
    def fit_batches(self, images, labels, batch_size):
        num_batches = int(len(labels)/batch_size) + 1
        res_images = []
        res_labels = []
        for i in range(num_batches):
            j = i * batch_size
            k = (i + 1) * batch_size
            if k > len(labels):
                k = len(labels)
                j = k - batch_size
            res_images.append(images[j:k])
            res_labels.append(labels[j:k])
            
        return np.array(res_images), np.array(res_labels)
    
    def augment(self, images, labels):
        label_count = [0 for _ in range(43)]
        for i in range(43):
            label_count[i] = np.sum(labels==i)
        augmented_images, augmented_labels = data_augment(images, labels, label_count)
        augmented_images = np.array(augmented_images)
        augmented_labels = np.array(augmented_labels)
        images = np.vstack([images, augmented_images])
        labels = np.hstack([labels, augmented_labels])
        return images, labels



def transform_image(img,ang_range,shear_range,trans_range):
    '''
    This function transforms images to generate new images.
    The function takes in following arguments,
    1- Image
    2- ang_range: Range of angles for rotation
    3- shear_range: Range of values to apply affine transform to
    4- trans_range: Range of values to apply translations over. 
    
    A Random uniform distribution is used to generate different parameters for transformation
    
    '''
    # Rotation

    ang_rot = np.random.uniform(ang_range)-ang_range/2
    rows,cols,ch = img.shape    
    Rot_M = cv2.getRotationMatrix2D((cols/2,rows/2),ang_rot,1)

    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    tr_y = trans_range*np.random.uniform()-trans_range/2
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])

    # Shear
    pts1 = np.float32([[5,5],[20,5],[5,20]])

    pt1 = 5+shear_range*np.random.uniform()-shear_range/2
    pt2 = 20+shear_range*np.random.uniform()-shear_range/2

    pts2 = np.float32([[pt1,5],[pt2,pt1],[5,pt2]])

    shear_M = cv2.getAffineTransform(pts1,pts2)
        
    img = cv2.warpAffine(img,Rot_M,(cols,rows))
    img = cv2.warpAffine(img,Trans_M,(cols,rows))
    img = cv2.warpAffine(img,shear_M,(cols,rows))
    
    return img


def data_augment(images, labels, label_count, ratio=2):
    assert len(images) == len(labels)
    cat_num = np.max(label_count) * ratio
    cats = [(cat_num-i)//i for i in label_count]
    augmented_images = []
    augmented_labels = []
    for i in range(len(labels)):
        for _ in range(cats[labels[i]]):
            img = transform_image(images[i], 150, 10, 5)
            augmented_images.append(img)
            augmented_labels.append(labels[i])
    return augmented_images, augmented_labels
