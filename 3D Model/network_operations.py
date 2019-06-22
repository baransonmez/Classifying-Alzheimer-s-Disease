from torch import nn
import helper
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
import random
import nibabel as nib
import numpy as np


device = torch.device("cpu")
custom_model_path = "model/model.pth"
torch.manual_seed(0)


def get_trained_network(network, dataset_dict, dataset_size_dict, epoch_size=20):

    loss_criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    model_optimizer = optim.SGD(network.parameters(), lr=0.0000001)

    # Decay LR by a factor of 0.1 every 10 epochs
    opt_lr_scheduler = lr_scheduler.StepLR(model_optimizer, step_size=10, gamma=0.1)

    model_trained = train_network(network, dataset_dict, dataset_size_dict, loss_criterion, model_optimizer, opt_lr_scheduler, epoch_size)

    return model_trained.cpu()


def train_network(model, dataset_loaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs):

    print("Training starts...")

    begin = time.time()
    model.train(True)
    best_model = copy.deepcopy(model.state_dict())
    best_acc = 0
    loss_dict = {"train": [], "val": []}
    acc_dict = {"train": [], "val": []}

    for epoch in range(num_epochs * 2):

        part = "val"
        if epoch % 2 == 0:
            part = "train"
            print('Epoch {}/{}'.format((epoch // 2) + 1, num_epochs))
            print('-' * 10)
        part_loss = 0.0
        part_corrects = 0

        if part == 'train':
            # # decreases learning rate
            scheduler.step()
            # Set model to training mode
            model.train(True)
            # shuffle
            random.shuffle(dataset_loaders["train"])
            for input_file, labels, cls in dataset_loaders[part]:
                # load mri
                x = nib.load(input_file).get_fdata()
                x = np.expand_dims(x, axis=0)
                transfrm = torch.from_numpy(x)
                transfrm = transfrm.float()
                inputs = transfrm.to(device)
                labels = torch.Tensor(labels).long().to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                with torch.set_grad_enabled(True):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    # back propogation
                    loss.backward()
                    optimizer.step()
                # add loss
                part_loss += loss.item() * inputs.size(0)
                part_corrects += torch.sum(preds == cls)
        else:
            # Set model to evaluate mode
            model.eval()
            for input_file, labels, cls in dataset_loaders[part]:
                #load mri
                x = nib.load(input_file).get_fdata()
                x = np.expand_dims(x, axis=0)
                transfrm = torch.from_numpy(x)
                transfrm = transfrm.float()
                inputs = transfrm.to(device)
                labels = torch.Tensor(labels).long().to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                with torch.set_grad_enabled(False):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                # loss
                part_loss += loss.item() * inputs.size(0)
                part_corrects += torch.sum(preds == cls)

        epoch_loss = part_loss / dataset_sizes[part]
        loss_dict[part].append(epoch_loss)
        epoch_acc = part_corrects.double() / dataset_sizes[part]
        acc_dict[part].append(epoch_acc)
        print('{} Loss: {:.4f}'.format(part, epoch_loss))
        print('{} Acc: {:.4f}'.format(part, epoch_acc))

        # deep copy the best model
        if part == 'val' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - begin
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model)
    # save model
    torch.save(model, custom_model_path)
    # visualize learning
    helper.plot_graphs(loss_dict, num_epochs, "loss")
    helper.plot_graphs(acc_dict, num_epochs, "accuracy")

    return model


def freeze_layers(model):
    """" Freeze some layers """
    model.conv1.requires_grad = False
    model.pool1.requires_grad = False
    model.conv2.requires_grad = False
    model.pool2.requires_grad = False
    model.conv3.requires_grad = False
    model.pool3.requires_grad = False

    print("Model is adjusted.")


def fine_tune_last_layers(num_epochs, dataset_loaders, dataset_sizes):
    # load model as feature extractor
    model = torch.load(custom_model_path)
    freeze_layers(model)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer = optim.SGD(model.parameters(), lr=0.0000001)

    # Decay LR by a factor of 0.1 every 10 epochs
    opt_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    print("Fine-tuning starts...")

    begin = time.time()
    model.train(True)
    best_model = copy.deepcopy(model.state_dict())
    best_acc = 0
    loss_dict = {"train": [], "val": []}
    acc_dict = {"train": [], "val": []}

    for epoch in range(num_epochs * 2):

        part = "val"
        if epoch % 2 == 0:
            part = "train"
            print('Epoch {}/{}'.format((epoch // 2) + 1, num_epochs))
            print('-' * 10)
        part_loss = 0.0
        part_corrects = 0

        if part == 'train':
            # Set model to training mode
            model.train(True)
            random.shuffle(dataset_loaders["train"])
            # random.shuffle(dataset_loaders["val"])
            for input_file, labels, cls in dataset_loaders[part]:
                x = nib.load(input_file).get_fdata()
                x = np.expand_dims(x, axis=0)
                transfrm = torch.from_numpy(x)
                transfrm = transfrm.float()
                inputs = transfrm.to(device)
                labels = torch.Tensor(labels).long().to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                with torch.set_grad_enabled(True):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    # back propogation
                    loss.backward()
                    optimizer.step()
                # add loss
                part_loss += loss.item() * inputs.size(0)
                part_corrects += torch.sum(preds == cls)
        else:
            # Set model to evaluate mode
            model.eval()
            for input_file, labels, cls in dataset_loaders[part]:
                x = nib.load(input_file).get_fdata()
                x = np.expand_dims(x, axis=0)
                transfrm = torch.from_numpy(x)
                transfrm = transfrm.float()
                inputs = transfrm.to(device)
                labels = torch.Tensor(labels).long().to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                with torch.set_grad_enabled(False):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                # loss
                part_loss += loss.item() * inputs.size(0)
                print("out ", outputs, " preds ", preds, " cls ", cls)
                part_corrects += torch.sum(preds == cls)

        epoch_loss = part_loss / dataset_sizes[part]
        loss_dict[part].append(epoch_loss)
        epoch_acc = part_corrects.double() / dataset_sizes[part]
        acc_dict[part].append(epoch_acc)
        print('{} Loss: {:.4f}'.format(part, epoch_loss))
        print('{} Acc: {:.4f}'.format(part, epoch_acc))

        # deep copy the best model
        if part == 'val' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - begin
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model)
    # save model
    torch.save(model, custom_model_path)
    # visualize learning
    helper.plot_graphs(loss_dict, num_epochs, "loss")
    helper.plot_graphs(acc_dict, num_epochs, "accuracy")


def test_network(dataset_loaders):
    """" Run tests """
    # load model for testing
    torch.nn.Module.dump_patches = True
    model = torch.load(custom_model_path)
    model.eval()
    model.train(False)
    loss_criterion = nn.CrossEntropyLoss()
    part_loss = 0.0
    part_corrects = 0
    conf_matr = [[0, 0], [0, 0]]
    y = []
    ystar = []
    for input_file, labels, cls in dataset_loaders["test"]:
        x = nib.load(input_file).get_fdata()
        x = np.expand_dims(x, axis=0)
        transfrm = torch.from_numpy(x)
        transfrm = transfrm.float()
        inputs = transfrm.to(device)
        labels = torch.Tensor(labels).long().to(device)
        # zero the parameter gradients
        # forward
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = loss_criterion(outputs, labels)
            # loss
            part_loss += loss.item() * inputs.size(0)
            print("out ", outputs)
            print("Prediction: ", preds[0].item(), " Correct Class: ", cls)
            part_corrects += torch.sum(preds == cls)
            conf_matr[cls][preds[0].item()] += 1
            ystar.append(preds[0].item())
            y.append(cls)

    print("Correct prediction ", part_corrects.item())
    print("Accuracy is ", (part_corrects.double()/len(dataset_loaders["test"])).item())
    helper.color_map_conf_matr(conf_matr)
    helper.calc_metric(y, ystar)
