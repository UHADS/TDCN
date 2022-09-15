import os
import glob
import torch
from u1tils import utility
import argparse
import scipy.io as sio
from time import time
import importlib

# set the path of test model
model_path = './model/s0.1.pth'
parser = argparse.ArgumentParser(description='TDCN')
parser.add_argument('--model', type=str, default='TDCNet' )
parser.add_argument('--cpu', action='store_true',help='use cpu only')
parser.add_argument('--sampling_rate', type=str, default='10', help='save reconstruct images')
parser.add_argument('--sampling_point', type=int, default=102, help='save reconstruct images')#1% - 10 4% - 41 10% - 102 25% - 256 30% - 307 40% - 410 50% - 512
parser.add_argument('--dir_data', type=str, default='./dataset/', help='dataset directory')
parser.add_argument('--dir', type=str, default='./res_images/', help='save reconstruct images')
parser.add_argument('--data_test', type=str, default='Set5', help='test dataset name,Set5+Set14+BSDS100+Set11')
parser.add_argument('--save_results', default= False, action='store_true', help='save output results')


args = parser.parse_args()
args.data_test = args.data_test.split('+')

with torch.no_grad():
    for dataset in args.data_test:
        image_list = glob.glob(args.dir_data + "/test_images_mat/{}_mat/*.*".format(dataset))
        avg_psnr = 0.0
        avg_ssim = 0.0
        sum_time = 0.0
      
        for image_name in image_list:
            image = sio.loadmat(image_name)['im_gt_y']
            image = image.astype(float)

            im_input = image / 255.
            im_input = torch.from_numpy(im_input).float().view(1, -1, im_input.shape[0], im_input.shape[1])
            if not args.cpu:
                im_input = im_input.cuda()
            
            module = importlib.import_module("tdcn.{}".format(args.model))
            net = module.TDCNet(base_filter= args.sampling_point)
            state_dict = torch.load(model_path, map_location='cuda:0')
    
            net.load_state_dict(state_dict['model'])
    
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
            net = net.to(device)

            start = time()
            im_output = net(im_input).squeeze()
         
            image_res = utility.normalize_255(im_output)
            end = time()
            sum_time += end - start

            psnr = utility.calc_psnr_255(image, image_res)
            ssim = utility.calc_ssim(image, image_res)
            avg_psnr += psnr
            avg_ssim += ssim

            if args.save_results:
                path = os.path.join(args.dir, args.sampling_rate)
                utility.save_image(image_res, psnr, ssim, path, dataset, image_name)

        avg_psnr = avg_psnr / len(image_list)
        avg_ssim = avg_ssim / len(image_list)
        print('[{}]\tPSNR: {:.2f}\tSSIM: {:.4f}'.format(dataset, avg_psnr, avg_ssim))
        print("Run time is %.4f" % (sum_time))
 