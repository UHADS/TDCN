import argparse
import torchvision
from torch.utils.data import DataLoader
from data.dataset import DatasetFromHdf5
from tdcn.slover import TDCNet_Trainer

# set Training parameter
parser = argparse.ArgumentParser(description='TDCN')

# Set super parameters
parser.add_argument('--batchSize', type=int, default=32, help='Small batch size')
parser.add_argument('--testBatchSize', type=int, default=1, help='Test batch size')
parser.add_argument('--nEpochs', type=int, default=100, help='Iterations')
parser.add_argument('--imageSize', type=int, default=96, metavar='N')
parser.add_argument('--samplingRate', type=int, default=1, help='sampling_rate = 1/4/10/25')
parser.add_argument('--samplingPoint', type=int, default=10, help='1% - 10 4% - 41 10% - 102 25% - 256 30% - 307 40% - 410 50% - 512')
#parser.add_argument('--trainPath', default='./dataset/train')
parser.add_argument('--valPath', default='dataset/vail')
parser.add_argument('--lr', type=float, default=0.0004, help='Learning Rate. Default=0.001')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--cuda', action='store_true', default=True)
parser.add_argument("--loss_mode", type=str, 
                        choices=["MSE", "L1", "SmoothL1"], default= "L1")
parser.add_argument('--resume', type=int, default= 0)
parser.add_argument('--resume_dir', type=str, default="./epochs/model_path.pth")

args = parser.parse_args()
kwargs = {'num_workers': 9, 'pin_memory': True} if args.cuda else {}


transforms_test = torchvision.transforms.Compose([
    torchvision.transforms.Grayscale(num_output_channels=1),
    torchvision.transforms.ToTensor(),
])

train_dataset = DatasetFromHdf5("train2000.h5")
val_dataset = torchvision.datasets.ImageFolder(args.valPath, transform=transforms_test)
train_loader = DataLoader(train_dataset, batch_size=args.batchSize, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.testBatchSize, shuffle=False)

# saved_model select
model = TDCNet_Trainer(args, train_loader, val_loader)


model.run()
