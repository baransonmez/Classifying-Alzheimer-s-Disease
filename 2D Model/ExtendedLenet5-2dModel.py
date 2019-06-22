import os

import nibabel as nib
import torch
import torch.backends.cudnn as cudnn
from matplotlib import cm
from sklearn.metrics import classification_report
from torch import nn, optim
from torchvision import transforms

cudnn.benchmark = True
import random
import numpy as np
from Lenet import LeNet5
import copy
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from inspect import signature
from sklearn.metrics import confusion_matrix


def sample_generator_specific_axis(data_list, axis):
    random.shuffle(data_list)
    mri_number = 0
    for mri, label in data_list:
        mri_data = nib.load(mri).get_data()

        if axis == "z":
            z_slice_limit = len(mri_data)
            for i in range(99, z_slice_limit - 20):
                np_label = np.zeros((1, 2))
                if label == "AD":
                    np_label[0][0] = 1
                else:
                    np_label[0][1] = 1

                yield (composed_transform(mri_data[i]), torch.Tensor(np_label).long(), mri_number)


        elif axis == "y":
            y_slice_limit = len(mri_data[:, :, ])
            y_slices = [mri_data[:, :, i] for i in range(y_slice_limit)]
            for i in range(40, y_slice_limit - 40):
                np_label = np.zeros((1, 2))
                if label == "AD":
                    np_label[0][0] = 1
                else:
                    np_label[0][1] = 1

                yield (composed_transform(y_slices[i]), torch.Tensor(np_label).long(), mri_number)


        elif axis == "x":
            x_slice_limit = len(mri_data[:, ])
            x_slices = [mri_data[:, i] for i in range(x_slice_limit)]
            for i in range(40, x_slice_limit - 40):
                np_label = np.zeros((1, 2))
                if label == "AD":
                    np_label[0][0] = 1
                else:
                    np_label[0][1] = 1
                from PIL import Image
                im = Image.fromarray(np.uint8(cm.gist_earth(x_slices[i]) * 255))
                im.show()
                yield (composed_transform(x_slices[i]), torch.Tensor(np_label).long(), mri_number)

        mri_number += 1


def generate_dataset(full_dataset):
    # random.shuffle(full_dataset)
    rand_full_list = []
    for i in range(int(len(full_dataset) / 2)):
        rand_full_list.append(full_dataset[i])
        rand_full_list.append(full_dataset[-i])
    val_count = int(0.15 * len(rand_full_list))
    test_count = int(0.12 * len(rand_full_list))
    train_data = rand_full_list[:-(val_count + test_count)]
    val_data = rand_full_list[len(train_data):-(test_count)]
    test_data = rand_full_list[len(val_data) + len(train_data):]

    return train_data, val_data, test_data


def generate_mri_label_pairs():
    dataset_path = "dataset/"
    classes = os.listdir(dataset_path)
    full_data = []
    for class_type in classes:

        class_type_path = os.path.join(dataset_path, class_type)

        samples = os.listdir(class_type_path)

        for sample in samples:

            if class_type == "Healty":
                sample_path = os.path.join(class_type_path, sample)
                anats = os.listdir(sample_path)

                for anat in anats:
                    nifti_path = os.path.join(sample_path, anat, "NIFTI")
                    nifti = os.listdir(nifti_path)[0]
                    mri_path = os.path.join(nifti_path, nifti)

            elif class_type == "AD":
                sample_path = os.path.join(class_type_path, sample)

                mri_names = os.listdir(sample_path)

                for mri_name in mri_names:

                    mri_name_path = os.path.join(sample_path, mri_name)
                    anats = os.listdir(mri_name_path)

                    for anat in anats:
                        nifti_path = os.path.join(mri_name_path, anat, "NIFTI")
                        nifti = os.listdir(nifti_path)[0]
                        mri_path = os.path.join(nifti_path, nifti)
            full_data.append((mri_path, class_type))

    return full_data


def train_model():
    full_data = generate_mri_label_pairs()
    train, val, test = generate_dataset(full_data)
    train_losses = []
    val_losses = []
    print(len(test))
    print(len(val))
    axis = "y"
    for epoch in range(epoch_num):

        true = 0
        random.shuffle(train)
        generator = sample_generator_specific_axis(train, axis)
        i = 0
        loss_train = 0
        for (images, labels, mri_check) in generator:

            im_dev = images.to(device, dtype=torch.float).unsqueeze(0)
            true_label = torch.argsort(labels.cpu())[0][0]
            labels_dev = true_label.to(device, dtype=torch.int64).unsqueeze(0)
            optim.zero_grad()
            outputs = cnn(im_dev)
            prediction = np.argmax(outputs.cpu().data)
            if true_label == prediction:
                true += 1
            loss = criterion(outputs, labels_dev)
            loss.backward()
            optim.step()

            loss_train += loss.data
            i = i + 1
        print('Epoch : %d/%d,  Training Loss: %.4f'
              % (epoch + 1, epoch_num, loss_train / i))
        print("Training Acc: " + str(true / i))
        train_losses.append(loss_train / i)

        generator = sample_generator_specific_axis(val, axis)
        valid_loss = 0
        i = 0
        true_val = 0
        true_scene = 0
        old_mri_check = 0
        total_mri_count = 0
        with torch.no_grad():
            cnn.eval()  # no_grad and eval mode for evaluation mode
            for (images, labels, mri_check) in generator:
                if mri_check != old_mri_check:
                    if true_scene >= old_mri_check / 2:
                        true_val += 1
                    true_scene = 0
                    total_mri_count += 1
                optim.zero_grad()
                im_dev = images.to(device, dtype=torch.float).unsqueeze(0)
                true_label = torch.argsort(labels.cpu())[0][0]
                labels_dev = true_label.to(device, dtype=torch.int64).unsqueeze(0)
                optim.zero_grad()
                outputs = cnn(im_dev)
                prediction = np.argmax(outputs.cpu().data)
                if true_label == prediction:
                    true += 1
                loss = criterion(outputs, labels_dev)
                optim.step()
                loss_train += loss.data
                if true_label == prediction:
                    true_scene += 1
                valid_loss += loss.data
                i = i + 1
                old_mri_check = mri_check

            print('Epoch : %d/%d,  Validation Loss: %.4f'
                  % (epoch + 1, epoch_num, valid_loss / i))
            val_losses.append(valid_loss / i)
            print("Validation Acc: " + str(true_val / total_mri_count))

    plt.plot(train_losses)
    plt.ylabel('Train Loss')
    plt.show()

    plt.plot(val_losses)
    plt.ylabel('Validation Loss')
    plt.show()
    torch.save(cnn.state_dict(), str(learning_rate) + "logsoftmax-" + axis)


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = classes["Healty","Alzheimer's Disease"]

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def calculate_test_accuracy(test_data):
    cnn.load_state_dict(torch.load("1.4e-05newExtrasigmodi-x"))  # Assigns model's weights from loaded fine tuned model
    cnn_x = copy.deepcopy(cnn)
    cnn.load_state_dict(torch.load("1.4e-05newExtrasigmodi-y"))  # Assigns model's weights from loaded fine tuned model
    cnn_y = copy.deepcopy(cnn)
    cnn.load_state_dict(torch.load("1.4e-05newExtrasigmodi-z"))  # Assigns model's weights from loaded fine tuned model
    cnn_z = copy.deepcopy(cnn)
    generator_z = sample_generator_specific_axis(test_data, "z")
    generator_y = sample_generator_specific_axis(test_data, "y")
    generator_x = sample_generator_specific_axis(test_data, "x")
    true_test = 0
    total_mri_count = 0
    old_mri_check = 0
    true_scene = 0
    i = 0
    pred_x = []
    true_label_list = []
    with torch.no_grad():
        cnn_x.eval()  # no_grad and eval mode for evaluation mode
        cnn_y.eval()  # no_grad and eval mode for evaluation mode
        cnn_z.eval()  # no_grad and eval mode for evaluation mode
        for (images, labels, mri_check) in generator_x:
            if mri_check != old_mri_check:

                if true_scene >= old_mri_check / 2:
                    true_test += 1
                    pred_x.append(np.argmax(old_label.data.numpy()))
                else:
                    pred_x.append(np.argmax(old_label.data.numpy()))
                true_label_list.append(old_label.data.numpy())

                true_scene = 0
                total_mri_count += 1
                i = i + 1
            optim.zero_grad()
            output_x = np.argmax(cnn_x(images.unsqueeze(0)).data.numpy())

            true_label = torch.argmax(labels, 1)
            if true_label[0] == output_x:
                true_scene += 1

            old_mri_check = mri_check
            old_label = labels

        i = 0
        pred_y = []
        old_mri_check = 0
        for (images, labels, mri_check) in generator_y:
            if mri_check != old_mri_check:
                if true_scene >= old_mri_check / 2:
                    true_test += 1
                    pred_y.append(np.argmax(old_label.data.numpy()))
                else:
                    pred_y.append(np.argmin(old_label.data.numpy()))
                true_scene = 0
                total_mri_count += 1
                i = i + 1
            optim.zero_grad()
            output_y = np.argmax(cnn_y(images.unsqueeze(0)).data.numpy())

            true_label = torch.argmax(labels, 1)
            if true_label[0] == output_y:
                true_scene += 1

            old_mri_check = mri_check
            old_label = labels

        i = 0
        pred_z = []
        old_mri_check = 0
        for (images, labels, mri_check) in generator_z:
            if mri_check != old_mri_check:
                if true_scene >= old_mri_check / 2:
                    true_test += 1
                    pred_z.append(np.argmax(old_label.data.numpy()))
                else:
                    pred_z.append(np.argmin(old_label.data.numpy()))
                true_scene = 0
                total_mri_count += 1
                i = i + 1
            optim.zero_grad()
            output_z = np.argmax(cnn_z(images.unsqueeze(0)).data.numpy())

            true_label = torch.argmax(labels, 1)
            if true_label[0] == output_z:
                true_scene += 1

            old_mri_check = mri_check
            old_label = labels

        true_result = 0
        prediction = {1: 0, 0: 0}
        predict = []
        true_labels = []

        for ind in range(len(pred_x)):

            prediction[pred_x[ind]] += 1.10
            prediction[pred_y[ind]] += 1.10
            prediction[pred_z[ind]] += 1.0
            pred_sorted_keys = sorted(prediction, key=prediction.get, reverse=True)
            predict.append(pred_sorted_keys[0])
            true_labels.append(np.argmax(true_label_list[ind]))
            if pred_sorted_keys[0] == np.argmax(true_label_list[ind]):
                true_result += 1
            prediction = {1: 0, 0: 0}

        print("Test Acc: " + str(true_result / len(pred_x)))
        from sklearn.metrics import precision_recall_fscore_support
        print(precision_recall_fscore_support(true_labels, predict, average='macro'))
        from sklearn.metrics import average_precision_score
        average_precision = average_precision_score(true_labels, predict)

        precision, recall, _ = precision_recall_curve(true_labels, predict)

        step_kwargs = ({'step': 'post'}
                       if 'step' in signature(plt.fill_between).parameters
                       else {})
        plt.step(recall, precision, color='b', alpha=0.2,
                 where='post')
        plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
            average_precision))
        plt.show()

        print(classification_report(true_labels, predict))

        np.set_printoptions(precision=2)

        # Plot non-normalized confusion matrix
        plot_confusion_matrix(true_labels, predict, classes=["healty", "AD dementia"],
                              title='Confusion matrix, without normalization')

        plt.show()


composed_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=1),

    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])
device = torch.device('cuda:0')
learning_rate = 0.000011  # Define learning rate
epoch_num = 30  # Define epoch count2
criterion = nn.CrossEntropyLoss()  # Choose loss function
cnn = LeNet5()
optim = optim.Adam(cnn.parameters(), lr=learning_rate)  # Choose optimizer
cnn.to(device)
train_model()

full_data = generate_mri_label_pairs()
train, val, test = generate_dataset(full_data)
calculate_test_accuracy(val)
calculate_test_accuracy(test)
