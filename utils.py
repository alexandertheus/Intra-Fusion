import model_architectures as models
import torch
from torchvision import datasets
import torchvision.transforms as transforms


def get_model(config, device=-1, relu_inplace=True):
    """
    :param device: instance of torch.device
    :return: An instance of torch.nn.Module
    """
    num_classes = 100 if config["dataset"] == "Cifar100" else 10

    model = {
        "vgg11_nobias": lambda: models.VGG(
            "VGG11",
            num_classes,
            batch_norm=False,
            bias=False,
            relu_inplace=relu_inplace,
        ),
        "vgg11_half_nobias": lambda: models.VGG(
            "VGG11_half",
            num_classes,
            batch_norm=False,
            bias=False,
            relu_inplace=relu_inplace,
        ),
        "vgg11_doub_nobias": lambda: models.VGG(
            "VGG11_doub",
            num_classes,
            batch_norm=False,
            bias=False,
            relu_inplace=relu_inplace,
        ),
        "vgg11_quad_nobias": lambda: models.VGG(
            "VGG11_quad",
            num_classes,
            batch_norm=False,
            bias=False,
            relu_inplace=relu_inplace,
        ),
        "vgg11": lambda: models.VGG(
            "VGG11", num_classes, batch_norm=False, relu_inplace=relu_inplace
        ),
        "vgg11_bn": lambda: models.VGG(
            "VGG11", num_classes, batch_norm=True, relu_inplace=relu_inplace
        ),
        "resnet18": lambda: models.ResNet18(num_classes=num_classes),
        "resnet18_nobias": lambda: models.ResNet18(
            num_classes=num_classes, linear_bias=False
        ),
        "resnet18_nobias_nobn": lambda: models.ResNet18(
            num_classes=num_classes, use_batchnorm=False, linear_bias=False
        ),
    }[config["model"]]()

    if device != -1:
        # model.to(device)
        model = model.cuda(device)
        if device == "cuda":
            model = torch.nn.DataParallel(model)
            torch.backends.cudnn.benchmark = True

    return model


def get_pretrained_model(config, path, device_id=-1, relu_inplace=True):
    model = get_model(config, device_id, relu_inplace=relu_inplace)

    if device_id != -1:
        state = torch.load(
            path,
            map_location=(
                lambda s, _: torch.serialization.default_restore_location(
                    s, "cuda:" + str(device_id)
                )
            ),
        )
    else:
        state = torch.load(
            path,
            map_location=(
                lambda s, _: torch.serialization.default_restore_location(
                    s, "cpu")
            ),
        )

    print(
        "Loading model at path {} which had accuracy {} and at epoch {}".format(
            path, state["test_accuracy"], state["epoch"]
        )
    )
    model.load_state_dict(state["model_state_dict"])
    return model, state["test_accuracy"] * 100


def find_ignored_layers(model_original, out_features):
    ignored_layers = []
    for m in model_original.modules():
        if isinstance(m, torch.nn.Linear) and m.out_features == out_features:
            ignored_layers.append(m)

    return ignored_layers


def get_cifar10_data_loader():
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            root="./data",
            train=True,
            transform=transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, 4),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
            download=True,
        ),
        batch_size=128,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            root="./data",
            train=False,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        ),
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return {"train": train_loader, "test": val_loader}


def get_cifar100_data_loader():
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(
            root="./data",
            train=True,
            transform=transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, 4),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
            download=True,
        ),
        batch_size=128,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(
            root="./data",
            train=False,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        ),
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return {"train": train_loader, "test": val_loader}


def evaluate(input_model, loaders, gpu_id):
    """
    Computes the accuracy of a given model (input_model) on a given dataset (loaders["test"]).
    """
    if gpu_id != -1:
        input_model = input_model.cuda(gpu_id)
    input_model.eval()

    accuracy_accumulated = 0
    total = 0
    with torch.no_grad():
        for images, labels in loaders["test"]:
            if gpu_id != -1:
                images, labels = images.cuda(), labels.cuda()

            test_output = input_model(images)

            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = (pred_y == labels).sum().item() / float(labels.size(0))
            accuracy_accumulated += accuracy
            total += 1

    return accuracy_accumulated / total
