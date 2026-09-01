import argparse
import os
import torch
import json
from ignite.contrib import metrics
from models import fastflow
import core.praser as Praser
import data.dataset as Dataset
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve
import core.util as Util

class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def train_one_epoch(dataloader, model, optimizer, epoch, args):
    model.train()
    loss_meter = AverageMeter()
    for step, ret in enumerate(dataloader):
        # forward
        data = ret['img'].cuda()
        ret = model(data)
        loss = ret["loss"]
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # log
        loss_meter.update(loss.item())
        if (step + 1) % args['train']['log_interval'] == 0 or (step + 1) == len(dataloader):
            print(
                "Epoch {} - Step {}: loss = {:.3f}({:.3f})".format(
                    epoch, step + 1, loss_meter.val, loss_meter.avg
                )
            )

def eval_once(dataloader, model):
    model.eval()
    
    pixel_auroc = metrics.ROC_AUC()  
    image_auroc = metrics.ROC_AUC()   
    
    with torch.no_grad():
        for ret in tqdm(dataloader, desc="Evaluating"):
            input_images = ret['img'].cuda()
            outputs = model(input_images)

            pix_preds = outputs["anomaly_map"].flatten().cpu()
            pix_targets = ret['gt'].flatten().cpu().int()
            pixel_auroc.update((pix_preds, pix_targets))

            img_preds = outputs["anomaly_score"].cpu()
            img_targets = ret['gt_label'].cpu().int()
            image_auroc.update((img_preds, img_targets))
    
    pixel_score = pixel_auroc.compute()
    image_score = image_auroc.compute()
        
    print(f"Pixel-level AUROC: {pixel_score:.3f}")
    print(f"Image-level AUROC: {image_score:.3f}")
    
    return pixel_score, image_score


def eval_with_threshold(dataloader, model):
    model.eval()
    
    all_pix_preds = []
    all_pix_targets = []
    
    with torch.no_grad():
        for ret in tqdm(dataloader, desc="Evaluating"):
            input_images = ret['img'].cuda()
            outputs = model(input_images)
            
            pix_preds = outputs["anomaly_map"].flatten().cpu().numpy()
            pix_targets = ret['gt'].flatten().cpu().numpy().astype(int)
    
            all_pix_preds.extend(pix_preds)
            all_pix_targets.extend(pix_targets)
    
    all_pix_preds = np.array(all_pix_preds)
    all_pix_targets = np.array(all_pix_targets)

    pixel_auroc = roc_auc_score(all_pix_targets, all_pix_preds)
    fpr_pix, tpr_pix, thresholds_pix = roc_curve(all_pix_targets, all_pix_preds)
    youden_pix = tpr_pix - fpr_pix
    best_idx_pix = np.argmax(youden_pix)
    best_threshold_pix = thresholds_pix[best_idx_pix]
    
    pix_min = all_pix_preds.min()
    pix_max = all_pix_preds.max()
    
    return {
        'pixel_auroc': float(pixel_auroc),
        'pixel_threshold': float(best_threshold_pix),
        'pixel_min': float(pix_min),
        'pixel_max': float(pix_max)
    }


def train(args):
    train_dataset = Dataset.ImageDataset(
        data_root = args['datasets']['train']['dataset']['data_root'],
        data_len = args['datasets']['train']['dataset']['data_len'],
        image_size = args['datasets']['train']['dataset']['image_size'],
        transforms_mean = args['model']['transforms_mean'], 
        transforms_std = args['model']['transforms_std'],
        data_aug = args['datasets']['train']['dataset']['data_aug'],
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = args['datasets']['train']['dataloader']['batch_size'],
        shuffle = args['datasets']['train']['dataloader']['shuffle'],
        pin_memory = args['datasets']['train']['dataloader']['pin_memory'],
        num_workers = args['datasets']['train']['dataloader']['num_workers'],
        drop_last = args['datasets']['train']['dataloader']['drop_last'],
    )
    
    val_dataset = Dataset.AnomalyDataset(
        imgs_root = args['datasets']['val']['dataset']['imgs_root'],
        gts_root = args['datasets']['val']['dataset']['gts_root'],
        data_len = args['datasets']['val']['dataset']['data_len'],
        image_size = args['datasets']['val']['dataset']['image_size'],
        target_size = args['datasets']['val']['dataset']['target_size'],
        transforms_mean = args['model']['transforms_mean'], 
        transforms_std = args['model']['transforms_std'],
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size = args['datasets']['val']['dataloader']['batch_size'],
        shuffle = args['datasets']['val']['dataloader']['shuffle'],
        pin_memory = args['datasets']['val']['dataloader']['pin_memory'],
        num_workers = args['datasets']['val']['dataloader']['num_workers'],
        drop_last = args['datasets']['val']['dataloader']['drop_last'],
    )
    
    print('len data', len(train_dataloader), len(val_dataloader))

    model = fastflow.build_model(args['model'], args['model']['backbone_pretrained'])
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args['train']['lr'], weight_decay=args['train']['weight_decay'])
    
    start_epoch = 0
    pixel_auroc_list = []; image_auroc_list = []; epoch_list = []
    if args['path']['resume_path'] is not None:
        checkpoint = torch.load(args['path']['resume_path'])
        model.load_state_dict(checkpoint['model_state_dict'])
        print('--------------------load resume model:[{:s}]'.format(args['path']['resume_path']))
    if args['path']['resume_state'] is not None:
        checkpoint = torch.load(args['path']['resume_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        epoch_list = checkpoint['epoch_list']
        pixel_auroc_list = checkpoint['pixel_auroc_list']
        image_auroc_list = checkpoint['image_auroc_list']
        print('--------------------load resume state:[{:s}]'.format(args['path']['resume_state']))
    
    for epoch in range(start_epoch+1, args['train']['num_epochs']+1):
        train_one_epoch(train_dataloader, model, optimizer, epoch, args)
        if epoch % args['train']['eval_interval'] == 0:
            pixel_auroc, image_auroc = eval_once(val_dataloader, model)
            
            pixel_auroc_list.append(pixel_auroc)
            image_auroc_list.append(image_auroc)
            epoch_list.append(epoch)
            
            best_pixel_auroc = max(pixel_auroc_list) if pixel_auroc_list else 0.0
            best_image_auroc = max(image_auroc_list) if image_auroc_list else 0.0
            best_pixel_epoch = epoch_list[pixel_auroc_list.index(best_pixel_auroc)] if pixel_auroc_list else 0
            best_image_epoch = epoch_list[image_auroc_list.index(best_image_auroc)] if image_auroc_list else 0
            plt.figure()
            plt.plot(epoch_list, pixel_auroc_list, 'b-', linewidth=2, label='Pixel AUROC')
            plt.plot(epoch_list, image_auroc_list, 'g-', linewidth=2, label='Image AUROC')
            plt.plot(best_pixel_epoch, best_pixel_auroc, 'b*', label=f'Best Pixel: {best_pixel_auroc:.3f}')
            plt.plot(best_image_epoch, best_image_auroc, 'g*', label=f'Best Image: {best_image_auroc:.3f}')
            plt.xlabel('Epoch')
            plt.ylabel('AUROC')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(args['path']['experiments_root'], "results", "auroc.png"))
            plt.close()
            np.save(os.path.join(args['path']['experiments_root'], "results", "pixel_auroc.npy"), pixel_auroc_list)
            np.save(os.path.join(args['path']['experiments_root'], "results", "image_auroc_list.npy"), image_auroc_list)
            np.save(os.path.join(args['path']['experiments_root'], "results", "epoch_list.npy"), epoch_list)
            
        if epoch % args['train']['checkpoint_interval'] == 0:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                },
                os.path.join(args['path']['experiments_root'], "checkpoint", "%d_model.pt" % epoch),
            )
            torch.save(
                {
                    "epoch": epoch,
                    "epoch_list": epoch_list,
                    "pixel_auroc_list": pixel_auroc_list,
                    "image_auroc_list": image_auroc_list,
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                os.path.join(args['path']['experiments_root'], "checkpoint", "%d_train_state.pt" % epoch),
            )


def evaluate(args, opt):    
    val_dataset = Dataset.AnomalyDataset(
        imgs_root = opt['datasets']['val']['dataset']['imgs_root'],
        gts_root = opt['datasets']['val']['dataset']['gts_root'],
        data_len = opt['datasets']['val']['dataset']['data_len'],
        image_size = opt['datasets']['val']['dataset']['image_size'],
        transforms_mean = opt['model']['transforms_mean'], 
        transforms_std = opt['model']['transforms_std'],
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size = opt['datasets']['val']['dataloader']['batch_size'],
        shuffle = opt['datasets']['val']['dataloader']['shuffle'],
        pin_memory = opt['datasets']['val']['dataloader']['pin_memory'],
        num_workers = opt['datasets']['val']['dataloader']['num_workers'],
        drop_last = opt['datasets']['val']['dataloader']['drop_last'],
    )
    
    model = fastflow.build_model(opt['model'], opt['model']['backbone_pretrained'])
    model.cuda()
    checkpoint = torch.load(opt['evaluate']['resume_path'])
    model.load_state_dict(checkpoint["model_state_dict"])
    
    result = eval_with_threshold(val_dataloader, model)
    result["resume_path"] = opt['evaluate']['resume_path'] 
    with open(args.eval_dir, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

def parse_args():
    parser = argparse.ArgumentParser(description="Train FastFlows")
    parser.add_argument("-c", "--config", type=str, default='./config/config_fastflow.json', help="path to config file")
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'test'], help='Run train or test', default='train')
    parser.add_argument('-eval', '--eval_dir', type=str, default='./config/config_fastflow_evaluate.json', help="path to eval file")
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-b', '--batch', type=int, default=None, help='Batch size in every gpu')
    parser.add_argument('-d', '--debug', action='store_true', help='Change the experiment folder name, see line 119 of the core.praser.py')
    parser.add_argument('-save', '--save_exp_log', default=True, type=bool, help='Create a experiment folder, see line 126 of the core.praser.py')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    opt = Praser.parse(args)
    
    ''' cuda devices, multi-GPU not supported '''
    gpu_str = ','.join(str(x) for x in opt['gpu_ids'])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
    print('export CUDA_VISIBLE_DEVICES={}'.format(gpu_str))

    Util.set_seed(opt['seed'])
    print('set seed={}'.format(opt['seed']))

    if args.phase == 'test':
        evaluate(args, opt)
    else:
        train(opt)
