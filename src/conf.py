class Config:

    def __init__(self, args):
        self.batch_size: int = args.batch_size
        self.epoch: int = args.epoch
        self.device: str = args.device
        self.size: int = args.size
        self.learning_rate: int = args.learning_rate
        self.noise_dimension: int = args.noise_dimension
        self.number_of_classes: int = args.number_of_classes
        self.features_discriminator: int = args.features_discriminator
        self.features_generator: int = args.features_generator
        self.train_path: int = args.train_path
        self.val_path: int = args.val_path
        self.data_format: int = args.data_format
