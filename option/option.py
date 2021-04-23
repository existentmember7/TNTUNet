import argparse

class Option():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--training_data_path', default='/media/han/D/aicenter_rebar_data/data/train_v3/',type=str, help="training data file path")
        self.parser.add_argument('--testing_data_path', type=str, default="/media/han/D/aicenter_rebar_data/data/test_v3/",help="testing data file path")
        self.parser.add_argument('--validating_data_path', type=str, default="/media/han/D/aicenter_rebar_data/data/validation_v3/",help="validating data file path")
        self.parser.add_argument("--image_height", type=int, default=256, help="size of image height") #720
        self.parser.add_argument("--image_width", type=int, default=256, help="size of image width") #1280
        self.parser.add_argument("--channels", type=int, default=4, help="number of image channels")
        self.parser.add_argument("--batch_size", type=int, default=1, help="batch size of training process")
        self.parser.add_argument("--num_classes", type=int, default=2, help="the number of classes")
        self.parser.add_argument("--base_lr", type=float, default=0.01, help="learning rate")
        self.parser.add_argument("--max_epochs", type=int, default=200, help="maximum epoch")
        self.parser.add_argument("--max_iterations", type=int, default=100000, help="maximun iteration")
        self.parser.add_argument("--deterministic", type=int, default=1, help="whether use deterministic training")
        self.parser.add_argument("--seed", type=int, default=42, help="random seed for cuda")
        self.parser.add_argument("--save_interval", type=int, default=50, help="the model weights saving interval")
        self.parser.add_argument("--model_path", type=str, default="/home/han/Documents/han/TNTUNet/model/", help="the path of saving logs")
        self.parser.add_argument("--model_weight_path", type=str, default="/home/han/Documents/han/TNTUNet/model/epoch_399.pth", help="the path of saving weights")
        self.parser.add_argument("--train", type=bool, default=None, help="true to train")
        self.parser.add_argument("--test", type=bool, default=None, help="true to test")
        self.parser.add_argument("--continue_training", type=bool, default=False, help="whether to continue from the last checkpoint")
        self.parser.add_argument("--ignore_background_class", type=bool, default=True, help="whether mIoU considered background class (0)")
        self.opt = self.parser.parse_args()