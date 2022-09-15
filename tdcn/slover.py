import torch
from math import log10
from tdcn.TDCNet import TDCNet
from u1tils.utility import progress_bar
import torch.backends.cudnn as cudnn

results = {'loss': [], 'psnr': []}


class TDCNet_Trainer(object):
    def __init__(self, config, training_loader, testing_loader):
        super(TDCNet_Trainer, self).__init__()
        self.criterion = torch.nn.L1Loss()
        self.criterionMSE = torch.nn.MSELoss()
        self.CUDA = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.CUDA else 'cpu')
        self.model = None
        self.lr = config.lr
        self.output_path = './epochs'
        self.nEpochs = config.nEpochs
        self.sampling_rate = config.samplingRate
        self.sampling_point = config.samplingPoint
        self.optimizer = None
        self.scheduler = None
        self.seed = config.seed
        self.training_loader = training_loader
        self.testing_loader = testing_loader
        self.loss_mode = config.loss_mode
        self.resume_dir =  config.resume_dir
        self.resume = config.resume
        self.step = 0  

    def build_model(self):
        self.model = TDCNet(base_filter=self.sampling_point).to(self.device)
        self.model.weight_init(mean=0.0, std=0.02)
        torch.manual_seed(self.seed)
        if self.CUDA:
            torch.cuda.manual_seed(self.seed)
            cudnn.benchmark = True
            self.criterion.cuda()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)

    def train(self):
        self.model.train()
        train_loss = 0
        if self.loss_mode in ["MSE"]: 
            self.loss_fn = self.criterionMSE
        elif self.loss_mode in ["L1"]: 
            self.loss_fn =  self.criterion
        elif self.loss_mode in ["SmoothL1"]:
            self.loss_fn = torch.nn.SmoothL1Loss()

        for batch_num, (data, target) in enumerate(self.training_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            a= self.model(data)
            loss = self.loss_fn(a, data)        
            train_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            progress_bar(batch_num, len(self.training_loader), 'Loss: %.4f' % (train_loss / (batch_num + 1)))
        return format(train_loss / len(self.training_loader))

    def test(self):
        self.model.eval()
        avg_psnr = 0
        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.testing_loader):
                data, target = data.to(self.device), target.to(self.device)
                prediction = self.model(data)
                mse = self.criterionMSE(prediction, data)
                psnr = 10 * log10(1 / mse.item())
                avg_psnr += psnr
                progress_bar(batch_num, len(self.testing_loader), 'PSNR: %.4f' % (avg_psnr / (batch_num + 1)))
        return format(avg_psnr / len(self.testing_loader))

    def run(self):
        self.build_model()
        if self.resume > 0:
           checkpoint = torch.load(self.resume_dir)
           self.optimizer.load_state_dict(checkpoint['optimizer'])
           self.model.load_state_dict(checkpoint['model'])
           self.scheduler.load_state_dict(checkpoint['scheduler'])
           self.step = checkpoint['epoch']

        for epoch in range(self.step + 1, self.nEpochs+1 ):
            print("\n===> Epoch {} starts:".format(epoch))
            loss = self.train()
            results['loss'].append(loss)
            avg_psnr = self.test()
            results['psnr'].append(avg_psnr)
            self.scheduler.step()
            checkpoint = {
                       'epoch': epoch,
                       'model': self.model.state_dict(),
                       'optimizer': self.optimizer.state_dict(),
                       'scheduler': self.scheduler.state_dict(),
                     }
            model_out_path = self.output_path + "/model_path" + str(epoch) + ".pth"
            torch.save(checkpoint, model_out_path)

        
