import albumentations as A


def get_transforms(size, number_of_channels):
    transf = A.Compose(
        [
            A.Resize(size, size),
            A.Normalize(
                mean=[0.5 for x in range(number_of_channels)],
                std=[0.5 for x in range(number_of_channels)],
                max_pixel_value=1
            )
        ]
    )
    return transf
