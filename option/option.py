import argparse

class Option():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--training_data_path', default='/home/han/Documents/han/TNTUNet/datasets/train/',type=str, help="training data file path")
        self.parser.add_argument('--testing_data_path', type=str, help="testing data file path")
        self.parser.add_argument('--validation_data_path', type=str, help="validation data file path")
        self.parser.add_argument("--image_height", type=int, default=256, help="size of image height") #720
        self.parser.add_argument("--image_width", type=int, default=256, help="size of image width") #1280
        self.parser.add_argument("--channels", type=int, default=4, help="number of image channels")
        self.parser.add_argument("--batch_size", type=int, default=1, help="batch size of training process")
        self.parser.add_argument("--num_classes", type=int, default=2, help="the number of classes")
        self.parser.add_argument("--base_lr", type=float, default=0.01, help="learning rate")
        self.parser.add_argument("--max_epochs", type=int, default=150, help="maximum epoch")
        self.parser.add_argument("--max_iterations", type=int, default=30000, help="maximun iteration")
        self.parser.add_argument("--deterministic", type=int, default=1, help="whether use deterministic training")
        self.parser.add_argument("--seed", type=int, default=42, help="random seed for cuda")
        self.opt = self.parser.parse_args()