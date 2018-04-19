import os
from scipy import ndimage
import numpy as np
from six.moves import cPickle as Pickle
import csv

DATA_FOLDER = 'data'
image_size = 32
pixel_depth = 255
pickle_extension = '.pickle'
num_classes = 48
image_per_class = 500


def get_folders(path):
    data_folders = [os.path.join(path, d) for d in sorted(os.listdir(path))
                    if os.path.isdir(os.path.join(path, d))]

    if len(data_folders) != num_classes:
        raise Exception(
            'Expected %d folders, one per class. Found %d instead.' % (
                num_classes, len(data_folders)))

    return data_folders


def load_letter(folder, min_num_images):
    """Load the data for a single letter label."""
    image_files = os.listdir(folder)
    dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                         dtype=np.float32)
    print(folder)
    image_index = -1
    for image_index, image in enumerate(image_files):
        image_file = os.path.join(folder, image)
        try:
            image_data = 1 * (ndimage.imread(image_file).astype(float) > pixel_depth / 2)
            if image_data.shape != (image_size, image_size):
                raise Exception('Unexpected image shape: %s' % str(image_data.shape))
            dataset[image_index, :, :] = image_data
        except IOError as err:
            print('Could not read:', image_file, ':', err, '- it\'s ok, skipping.')

    num_images = image_index + 1
    dataset = dataset[0:num_images, :, :]
    if num_images < min_num_images:
        raise Exception('Many fewer images than expected: %d < %d' % (num_images, min_num_images))

    print('Full dataset tensor:', dataset.shape)
    print('Mean:', np.mean(dataset))
    print('Standard deviation:', np.std(dataset))
    return dataset


def maybe_pickle(data_folders, min_num_images_per_class, force=False):
    dataset_names = []
    for folder in data_folders:
        set_filename = folder + pickle_extension
        dataset_names.append(folder)
        if os.path.exists(set_filename) and not force:
            # You may override by setting force=True.
            print('%s already present - Skipping pickling.' % set_filename)
        else:
            # print('Pickling %s.' % set_filename)
            dataset = load_letter(folder, min_num_images_per_class)
            try:
                with open(set_filename, 'wb') as f:
                    Pickle.dump(dataset, f, Pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print('Unable to save data to', set_filename, ':', e)

    return dataset_names


def make_arrays(nb_rows, img_size):
    if nb_rows:
        dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
        labels = np.ndarray(nb_rows, dtype=np.int32)
    else:
        dataset, labels = None, None
    return dataset, labels


def merge_datasets(pickle_files, train_size, test_size=0, valid_size=0):
    num_classes = len(pickle_files)
    print(num_classes)
    valid_dataset, valid_labels = make_arrays(valid_size, image_size)
    test_dataset, test_labels = make_arrays(test_size, image_size)
    train_dataset, train_labels = make_arrays(train_size, image_size)
    valid_size_per_class = valid_size // num_classes
    test_size_per_class = test_size // num_classes
    train_size_per_class = train_size // num_classes

    print(valid_size_per_class, test_size_per_class, train_size_per_class)

    start_valid, start_test, start_train = 0, valid_size_per_class, (valid_size_per_class + test_size_per_class)
    end_valid = valid_size_per_class
    end_test = end_valid + test_size_per_class
    end_train = end_test + train_size_per_class

    print(start_valid, end_valid)
    print(start_test, end_test)
    print(start_train,end_train)

    s_valid, s_test, s_train = 0, 0, 0
    e_valid, e_test, e_train = valid_size_per_class, test_size_per_class, train_size_per_class
    temp = []
    for label, pickle_file in enumerate(pickle_files):
        temp.append([label, pickle_file[-4:]])
        try:
            with open(pickle_file + pickle_extension, 'rb') as f:
                letter_set = Pickle.load(f)
                # let's shuffle the letters to have random validation and training set
                np.random.shuffle(letter_set)
                if valid_dataset is not None:
                    valid_letter = letter_set[:end_valid, :, :]
                    valid_dataset[s_valid:e_valid, :, :] = valid_letter
                    valid_labels[s_valid:e_valid] = label
                    s_valid += valid_size_per_class
                    e_valid += valid_size_per_class

                if test_dataset is not None:
                    test_letter = letter_set[start_test:end_test, :, :]
                    test_dataset[s_test:e_test, :, :] = test_letter
                    test_labels[s_test:e_test] = label
                    s_test += test_size_per_class
                    e_test += test_size_per_class

                train_letter = letter_set[start_train:end_train, :, :]
                train_dataset[s_train:e_train, :, :] = train_letter
                train_labels[s_train:e_train] = label
                s_train += train_size_per_class
                e_train += train_size_per_class
        except Exception as e:
            print('Unable to process data from', pickle_file, ':', e)
            raise
    with open('classes.csv', 'w') as my_csv:
        writer = csv.writer(my_csv, delimiter=',')
        writer.writerows(temp)
    return valid_dataset, valid_labels, test_dataset, test_labels, train_dataset, train_labels


data_folders = get_folders(DATA_FOLDER)
train_datasets = maybe_pickle(data_folders, image_per_class, True)
train_size = int(image_per_class * num_classes * 0.7)
test_size = int(image_per_class * num_classes * 0.2)
valid_size = int(image_per_class * num_classes * 0.1)

valid_dataset, valid_labels, test_dataset, test_labels, train_dataset, train_labels = merge_datasets(
    train_datasets, train_size, test_size, valid_size)

print('Training set', train_dataset.shape, train_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)

pickle_file = 'data.pickle'

try:
    f = open(pickle_file, 'wb')
    save = {
        'train_dataset': train_dataset,
        'train_labels': train_labels,
        'valid_dataset': valid_dataset,
        'valid_labels': valid_labels,
        'test_dataset': test_dataset,
        'test_labels': test_labels,
    }
    Pickle.dump(save, f, Pickle.HIGHEST_PROTOCOL)
    f.close()
except Exception as e:
    print('Unable to save data to', pickle_file, ':', e)
    raise

statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)
