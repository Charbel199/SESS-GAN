import torchvision.transforms as transforms


def get_transforms(size, number_of_channels):
    transf = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5 for x in range(number_of_channels)], [0.5 for x in range(number_of_channels)]
            )
        ]
    )
    return transf
