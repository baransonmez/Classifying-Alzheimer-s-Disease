import dataloader
import network_operations as netop
import network
import sys


dataset_loaders = {}
dataset_sizes = {}


if __name__ == "__main__":
    # first argument is path to main directory
    root_path = sys.argv[1]
    # load train val and test data
    tr_load = dataloader.find_path_pairs(root_path + "Train/")
    dataset_loaders["train"] = tr_load
    dataset_sizes["train"] = len(tr_load)

    val_load = dataloader.find_path_pairs(root_path + "Val/")
    dataset_loaders["val"] = val_load
    dataset_sizes["val"] = len(val_load)

    test_load = dataloader.find_path_pairs(root_path + "Test/")
    dataset_loaders["test"] = test_load
    dataset_sizes["test"] = len(test_load)
    print("Train data size: ", dataset_sizes["train"])
    print("Validation data size: ", dataset_sizes["val"])
    print("Test data size: ", dataset_sizes["test"])

    device = netop.device
    print("Running on " + str(device))
    net = network.CustomNet()
    # train model
    netop.get_trained_network(net, epoch_size=20, dataset_dict=dataset_loaders, dataset_size_dict=dataset_sizes)
    # fine-tune last layers
    netop.fine_tune_last_layers(num_epochs=20, dataset_loaders=dataset_loaders, dataset_sizes=dataset_sizes)
    # test network, this uses the pretrained data
    # once pretrained data is obtained above two lines can be commented
    netop.test_network(dataset_loaders)


