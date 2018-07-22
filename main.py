import argparse
from training import training_procedure

parser = argparse.ArgumentParser()

# add arguments
parser.add_argument('--cuda', type=bool, default=True, help="run the following code on a GPU")

parser.add_argument('--batch_size', type=int, default=64, help="batch size for training")
parser.add_argument('--image_size', type=int, default=28, help="height and width of the image")
parser.add_argument('--num_channels', type=int, default=1, help="number of channels in the images")
parser.add_argument('--initial_learning_rate', type=float, default=0.0001, help="starting learning rate")

parser.add_argument('--style_dim', type=int, default=16, help="dimension of style latent space")
parser.add_argument('--class_dim', type=int, default=16, help="dimension of class latent space")
parser.add_argument('--num_classes', type=int, default=10, help="number of classes on which the data set trained")

# loss function coefficient
# 3 reconstruction coef for 64 dim space
parser.add_argument('--reconstruction_coef', type=float, default=2., help="coefficient for reconstruction term")
parser.add_argument('--reverse_cycle_coef', type=float, default=10., help="coefficient for reverse cycle loss term")
parser.add_argument('--kl_divergence_coef', type=float, default=3., help="coefficient for KL-Divergence loss term")

parser.add_argument('--beta_1', type=float, default=0.9, help="default beta_1 val for adam")
parser.add_argument('--beta_2', type=float, default=0.999, help="default beta_2 val for adam")


# paths to save models
parser.add_argument('--encoder_save', type=str, default='encoder', help="model save for encoder")
parser.add_argument('--decoder_save', type=str, default='decoder', help="model save for decoder")

parser.add_argument('--log_file', type=str, default='log.txt', help="text file to save training logs")

parser.add_argument('--load_saved', type=bool, default=False, help="flag to indicate if a saved model will be loaded")
parser.add_argument('--start_epoch', type=int, default=0, help="flag to set the starting epoch for training")
parser.add_argument('--end_epoch', type=int, default=100, help="flag to indicate the final epoch of training")

FLAGS = parser.parse_args()

if __name__ == '__main__':
    training_procedure(FLAGS)
