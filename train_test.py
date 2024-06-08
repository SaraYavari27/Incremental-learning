from __future__ import absolute_import, print_function
from copy import deepcopy
import argparse
import torch.utils.data
from torch.backends import cudnn
from torch.autograd import Variable
import torchvision.models as models
import torch
import losses
from utils import RandomIdentitySampler, mkdir_if_missing, logging
from torch.optim.lr_scheduler import StepLR
from ImageFolder import *
import torch.nn.functional as F
import torch.nn as nn
import collections
import random
import glob
import SimpleITK as sitk
import time
from torchvision import transforms as T
from torch.utils.data import TensorDataset
import numpy as np
import errno

cudnn.benchmark = True


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.resnet18 = models.resnet18(pretrained=False)
        self.resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def forward(self, x):
        x = self.resnet18(x)
        return x


class DeepInversionFeatureHook():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]

        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)

        # forcing mean and variance to match between two distributions
        r_feature = torch.norm(module.running_var.data.type(var.type()) - var, 2) + torch.norm(
            module.running_mean.data.type(var.type()) - mean, 2)

        self.r_feature = r_feature
        # must have no output

    def close(self):
        self.hook.remove()


# This function performs both Re-slicing and cropping (written by Moslem)
def volume_resample_crop(image, spacing, crop_size, image_name):
    # image: input simpleitk image list
    # spacing: desired(output) spacing
    # crop_size: desired(output) image size
    # image_name: could be one of "T2w", "Adc", "Hbv", "Lesion" or "Prostate"

    orig_size = np.array(image.GetSize())
    new_spacing = np.array(spacing)
    orig_spacing = image.GetSpacing()
    new_size = orig_size * (orig_spacing / new_spacing)
    new_size = np.floor(new_size)
    new_size = [int(s) for s in new_size]
    # Create the ResampleImageFilter
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(list(new_spacing))
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)
    if image_name == "Lesion" or image_name == "Prostate":
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resampler.SetInterpolator(sitk.sitkBSpline)
    # Execute the filter
    resample_image = resampler.Execute(image)
    # Get the size of the MRI volume
    size = resampler.GetSize()
    # Get the center voxel index of the MRI volume
    center = [int(size[0] / 2), int(size[1] / 2), int(size[2] / 2)]
    # Calculate the start and end indices for each dimension
    start_index = [center[i] - int(crop_size[i] / 2) for i in range(3)]
    end_index = [start_index[i] + crop_size[i] for i in range(3)]
    cropper = sitk.CropImageFilter()
    # Set the crop boundaries
    cropper.SetLowerBoundaryCropSize(start_index)
    cropper.SetUpperBoundaryCropSize([size[i] - end_index[i] for i in range(3)])
    # Crop the volume
    resample_cropped_volume = cropper.Execute(resample_image)

    return resample_cropped_volume


# This function performs slicing on each MRI volume in z axis (written by Moslem)
def slice_data(image, image_name, fold_number, new_spacing, crop_size):
    # image: input simpleitk image list
    # image_name: could be one of "T2w", "Adc", "Hbv", "Lesion" or "Prostate"
    # fold_number: could be one of 0, 1, 2, 3, 4
    # new_spacing: desired(output) spacing
    # crop_size: desired(output) image size

    if image_name == "T2w":
        var_name = "image_T2w_list_class_" + str(fold_number)
        dict_vars = {}
        dict_vars[var_name] = []
        for i in range(len(image)):
            resample_image_T2w = volume_resample_crop(image[i], new_spacing, crop_size, image_name)
            for z in range(resample_image_T2w.GetSize()[2]):
                z_slice = sitk.Extract(resample_image_T2w, [resample_image_T2w.GetSize()[0],
                                                            resample_image_T2w.GetSize()[1], 0],
                                       [0, 0, z])
                dict_vars[var_name].append(z_slice)
    elif image_name == "Adc":
        var_name = "image_Adc_list_class_" + str(fold_number)
        dict_vars = {}
        dict_vars[var_name] = []
        for i in range(len(image)):
            resample_image_Adc = volume_resample_crop(image[i], new_spacing, crop_size, image_name)
            for z in range(resample_image_Adc.GetSize()[2]):
                z_slice = sitk.Extract(resample_image_Adc, [resample_image_Adc.GetSize()[0],
                                                            resample_image_Adc.GetSize()[1], 0],
                                       [0, 0, z])
                dict_vars[var_name].append(z_slice)
    elif image_name == "Hbv":
        var_name = "image_Hbv_list_class_" + str(fold_number)
        dict_vars = {}
        dict_vars[var_name] = []
        for i in range(len(image)):
            resample_image_Hbv = volume_resample_crop(image[i], new_spacing, crop_size, image_name)
            for z in range(resample_image_Hbv.GetSize()[2]):
                z_slice = sitk.Extract(resample_image_Hbv, [resample_image_Hbv.GetSize()[0],
                                                            resample_image_Hbv.GetSize()[1], 0],
                                       [0, 0, z])
                dict_vars[var_name].append(z_slice)
    elif image_name == "Lesion":
        var_name = "image_Lesion_list_class_" + str(fold_number)
        dict_vars = {}
        dict_vars[var_name] = []
        for i in range(len(image)):
            resample_image_Lesion = volume_resample_crop(image[i], new_spacing, crop_size, image_name)
            for z in range(resample_image_Lesion.GetSize()[2]):
                z_slice = sitk.Extract(resample_image_Lesion, [resample_image_Lesion.GetSize()[0],
                                                               resample_image_Lesion.GetSize()[1], 0],
                                       [0, 0, z])
                dict_vars[var_name].append(z_slice)
    elif image_name == "Prostate":
        var_name = "image_Prostate_list_Fold_" + str(fold_number)
        dict_vars = {}
        dict_vars[var_name] = []
        for i in range(len(image)):
            resample_image_Prostate = volume_resample_crop(image[i], new_spacing, crop_size, image_name)
            for z in range(resample_image_Prostate.GetSize()[2]):
                z_slice = sitk.Extract(resample_image_Prostate, [resample_image_Prostate.GetSize()[0],
                                                                 resample_image_Prostate.GetSize()[1], 0],
                                       [0, 0, z])
                dict_vars[var_name].append(z_slice)

    return dict_vars[var_name]


def augment_data(img_list, data_mean, data_std, data_type):
    # img_list: input numpy image list
    # data_mean: mean of data to be normalized
    # data_std: std of data to be normalized
    # data_type: could be one of "data" or "target"

    if data_type == "data":
        transform_tr = T.Compose([
            T.RandomHorizontalFlip(0.1),
            T.Normalize(data_mean, data_std)
        ])
        transformed_imgs = []
        for img in img_list:
            float_tensor = T.ToTensor()(Image.fromarray(img)).float()
            img_tensor = transform_tr(float_tensor)
            transformed_imgs.append(img_tensor)
    elif data_type == "target":
        transformed_imgs = torch.tensor(img_list.astype(np.float32))
    return transformed_imgs

def augment_data_test(img_list, data_mean, data_std, data_type):
    if data_type == "data":
        transform_tr = T.Compose([
            T.Normalize(data_mean, data_std)
        ])
        transformed_imgs = []
        for img in img_list:
            float_tensor = T.ToTensor()(Image.fromarray(img)).float()
            img_tensor = transform_tr(float_tensor)
            transformed_imgs.append(img_tensor)

    elif data_type == "target":
        transformed_imgs = torch.tensor(img_list.astype(np.float32))
    return transformed_imgs

def get_image_prior_losses(inputs_jit):
    # COMPUTE total variation regularization loss
    diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
    diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
    diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
    diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]

    loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
    loss_var_l1 = (diff1.abs() / 255.0).mean() + (diff2.abs() / 255.0).mean() + (
            diff3.abs() / 255.0).mean() + (diff4.abs() / 255.0).mean()
    loss_var_l1 = loss_var_l1 * 255.0
    return loss_var_l1, loss_var_l2


def mkdir_if_missing(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return model


def initial_train_fun(args, trainloader, num_class, dictlist):
    model = models.resnet18(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_class)
    model = model.cuda()
    best_model_wts = deepcopy(model.state_dict())
    log_dir = os.path.join('checkpoints', args.log_dir)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_0, momentum=args.momentum_0,
                                weight_decay=args.weight_decay_0)
    exp_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.lr_step)
    mkdir_if_missing(log_dir)

    for epoch in range(args.epochs_0):
        print(f'Epoch {epoch}/{args.epochs_0 - 1}')
        print('-' * 50)

        model.train()  # Set model to training mode
        dataloaders = deepcopy(trainloader)
        # dataset_sizes = dataset_sizes_train
        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for inputs, labels in dataloaders:

            inputs = inputs.repeat(1, 3, 1, 1)
            labels = labels
            labels_np = labels.numpy()

            for ii in range(len(labels_np)):
                labels_np[ii] = dictlist[labels_np[ii]]

            labels = labels.type(torch.LongTensor)
            inputs = Variable(inputs.cuda())
            labels = Variable(labels).cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # track history if only in train
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                exp_lr_scheduler.step()

    best_model_wts = deepcopy(model.state_dict())
    model.load_state_dict(best_model_wts)
    # model = nn.Sequential(*list(model.children())[:-1])

    state = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state,
               os.path.join(log_dir, args.method + '_task_' + str(0) + '_%d_model.pt' % args.epochs))


def train_fun(args, train_loader, current_task, old_labels):
    log_dir = os.path.join('checkpoints', args.log_dir)
    mkdir_if_missing(log_dir)
    sys.stdout = logging.Logger(os.path.join(log_dir, 'log.txt'))
    num_class = 2

    if current_task > 0:
        model = models.resnet18(pretrained=False)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_class)
        state1 = torch.load(os.path.join(log_dir, args.method +
                                         '_task_' + str(current_task - 1) + '_%d_model.pt' % int(args.epochs)))
        model.load_state_dict(state1['state_dict'])

        model_old = deepcopy(model)
        model_old = nn.Sequential(*list(model_old.children())[:-1])
        model_old.eval()
        model_old = freeze_model(model_old)
        model_old = model_old.cuda()
        model_gen = deepcopy(model)
        model_gen.eval()

    model = model.cuda()
    model.train()

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=200, gamma=0.1)

    if current_task > 0:
        loss_r_feature_layers = []

        for module in model_gen.modules():
            if isinstance(module, nn.BatchNorm2d):
                loss_r_feature_layers.append(DeepInversionFeatureHook(module))
    all_gen_input = []
    all_gen_labels = []

    for epoch in range(args.start, args.epochs + 1):
        print(f'Epoch {epoch}/{args.epochs - 1}')
        print('-' * 50)

        running_loss = 0.0

        if epoch == 0 and current_task > 0:
            print(50 * '#')
            print("Synthetic data generating...")

            for ii in range(len(train_loader)):

                data_type = torch.float
                inputs_d = torch.randn((args.batch_size_gen, 1, args.resolution, args.resolution),
                                       requires_grad=True,
                                       device='cuda', dtype=data_type)
                optimizer_gen = torch.optim.Adam([inputs_d], lr=args.lr_gen)
                optimizer_gen.state = collections.defaultdict(dict)
                lim_0, lim_1 = 6, 6
                prev_label = old_labels[current_task - 1]
                prev_label = list(set(prev_label))

                targets = torch.LongTensor([prev_label[0]] * int(args.batch_size_gen / 2)
                                           + [prev_label[1]] * int(args.batch_size_gen / 2)).to('cuda')

                # # Record the start time
                # start_time = time.time()
                for epoch2 in range(args.epoch_gen):
                    off1 = random.randint(-lim_0, lim_0)
                    off2 = random.randint(-lim_1, lim_1)
                    inputs_jit = torch.roll(inputs_d, shifts=(off1, off2), dims=(2, 3))
                    inputs_jit = inputs_jit.repeat(1, 3, 1, 1)

                    # foward with jit images
                    optimizer_gen.zero_grad()
                    model_gen.zero_grad()
                    model_gen = model_gen.cuda()
                    model_gen.eval()

                    embed_feat_gen = model_gen(inputs_jit)
                    # embed_feat_gen = torch.squeeze(embed_feat_gen)
                    # embed_feat_normal_gen = F.normalize(embed_feat_gen, p=2, dim=1)

                    loss_gen = criterion(embed_feat_gen, targets)

                    # R_prior losses
                    loss_var_l1, loss_var_l2 = get_image_prior_losses(inputs_jit)

                    # R_feature loss
                    rescale = [args.first_bn_mul] + [1. for _ in range(len(loss_r_feature_layers) - 1)]
                    loss_r_feature = sum(
                        [mod.r_feature * rescale[idx] for (idx, mod) in enumerate(loss_r_feature_layers)])

                    # l2 loss on images
                    loss_l2 = torch.norm(inputs_jit.view(args.BatchSize, -1), dim=1).mean()

                    # combining losses
                    loss_aux = args.tv_l2 * loss_var_l2 + \
                               args.tv_l1 * loss_var_l1 + \
                               args.bn_reg_scale * loss_r_feature + \
                               args.l2 * loss_l2

                    loss_gen = args.main_mul * loss_gen + loss_aux

                    loss_gen.backward()
                    optimizer_gen.step()

                # # Record the end time
                # end_time = time.time()
                # # Calculate the running time
                # running_time = end_time - start_time
                # print(f"The running time is {running_time:.4f} seconds.")
                all_gen_input.append(inputs_d)
                all_gen_labels.append(targets)

            ge_input = torch.cat(all_gen_input)
            ge_label = torch.cat(all_gen_labels)
            trainset = torch.utils.data.TensorDataset(ge_input, ge_label)
            train_loader_synth = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size_gen, drop_last=True,
                                                             num_workers=args.nThreads)

        for jj, (img, img1) in enumerate(zip(train_loader, train_loader_synth), 0):

            inputs, labels = img
            labels = labels.type(torch.LongTensor)
            inputs_synth, labels_synth = img1
            inputs_synth = Variable(inputs_synth.cuda())
            inputs = inputs.repeat(1, 3, 1, 1)
            inputs_synth = inputs_synth.repeat(1, 3, 1, 1)

            # wrap them in Variable
            inputs = Variable(inputs.cuda())
            labels = Variable(labels).cuda()
            optimizer.zero_grad()

            if current_task == 0:
                loss_aug = 0
            elif current_task > 0:
                if args.method == 'Fine_tuning':
                    loss_aug = 0

                elif args.method == 'Ours':
                    loss_aug = 0
                    embed_feat_old = model_old(inputs_synth)
                    embed_feat_old = torch.squeeze(embed_feat_old)

                    model_kd = deepcopy(model)
                    model_kd = nn.Sequential(*list(model_kd.children())[:-1])
                    embed_feat_synth = model_kd(inputs_synth)
                    embed_feat_synth = torch.squeeze(embed_feat_synth)

                    cov_loss = F.mse_loss(embed_feat_synth, embed_feat_old)

                    loss_aug += args.lambda_cov * cov_loss


            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss += loss_aug
                loss.backward()
                optimizer.step()

            running_loss += loss.data
            if epoch == 0 and jj == 0:
                print(50 * '#')
                print('Training...')

        model_save = deepcopy(model)
        # model_save = nn.Sequential(*list(model_save.children())[:-1])

        if epoch % args.save_step == 0:
            state = {
                'state_dict': model_save.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            torch.save(state,
                       os.path.join(log_dir, args.method + '_task_' + str(current_task) + '_%d_model.pt' % epoch))


def test1(args, test_data, current_task):

    log_dir = os.path.join('checkpoints', args.log_dir)
    model = models.resnet18(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    state1 = torch.load(os.path.join(log_dir, args.method +
                                     '_task_' + str(current_task - 1) + '_%d_model.pt' % int(args.epochs)))
    model.load_state_dict(state1['state_dict'])
    model.cuda()
    model.eval()

    # Test the model on the test data and calculate the accuracy
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_data:
            images = images.repeat(1, 3, 1, 1)
            images = images.cuda()
            labels = labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total

    return accuracy

def test2(args, test_data, current_task):
    # Load the pre-trained ResNet18 model
    log_dir = os.path.join('checkpoints', args.log_dir)
    model = models.resnet18(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    state1 = torch.load(os.path.join(log_dir, args.method +
                                     '_task_' + str(current_task - 1) + '_%d_model.pt' % int(args.epochs)))
    model.load_state_dict(state1['state_dict'])
    model.cuda()
    model.eval()

    # Test the model on the test data and calculate the accuracy
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_data:
            images = images.repeat(1, 3, 1, 1)
            images = images.cuda()
            labels = labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total

    return accuracy

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='CL Medical Training')

    # hyper-parameters
    parser.add_argument('-lr', type=float, default=1e-5, help="learning rate")
    parser.add_argument('-lambda_cov', type=float, default=0.8, help="MSE loss Coefficient")
    parser.add_argument('-margin', type=float, default=0.0, help="margin for metric loss")
    parser.add_argument('-BatchSize', '-b', default=32, type=int, metavar='N', help='mini-batch size Default: 64')
    parser.add_argument('-num_instances', default=2, type=int, metavar='n',
                        help=' number of samples from one class in mini-batch')
    parser.add_argument('-dim', default=512, type=int, metavar='n', help='dimension of embedding space')

    # generator hyper-parameters
    parser.add_argument('--jitter', default=30, type=int, help='jittering factor')
    parser.add_argument('--bn_reg_scale', type=float, default=0.05,
                        help='coefficient for feature distribution regularization')
    parser.add_argument('--first_bn_mul', type=float, default=10.0,
                        help='additional multiplier on first bn layer of R_feature')
    parser.add_argument('--tv_l1', type=float, default=0.0, help='coefficient for total variation L1 loss')
    parser.add_argument('--tv_l2', type=float, default=0.001, help='coefficient for total variation L2 loss')
    parser.add_argument('--lr_gen', type=float, default=0.1, help='learning rate for optimization')
    parser.add_argument('--l2', type=float, default=0.00001, help='l2 loss on the image')
    parser.add_argument('--main_mul', type=float, default=1.0, help='coefficient for the main loss in optimization')
    parser.add_argument('--resolution', type=int, default=224, help='resolution of image')
    parser.add_argument('--batch_size_gen', type=int, default=16, metavar='N', help='generator batch size')
    parser.add_argument('--epoch_gen', type=int, default=100, help='epochs for generating synthetic images')
    parser.add_argument('-num_instances_gen', default=8, type=int, metavar='n',
                        help=' number of samples from one class in generated mini-batch')

    # data & network
    parser.add_argument('-data', default='PICAI', help='path to Data Set')
    parser.add_argument('-loss', default='triplet', help='loss for training network')
    parser.add_argument('-epochs', default=50, type=int, metavar='N', help='epochs for training process')
    parser.add_argument('-seed', default=2022, type=int, metavar='N', help='seeds for training process')
    parser.add_argument('-save_step', default=50, type=int, metavar='N', help='number of epochs to save model')
    parser.add_argument('-lr_step', default=200, type=int, metavar='N', help='scheduler step')
    parser.add_argument('-start', default=0, type=int, help='resume epoch')

    # basic parameter
    parser.add_argument('-log_dir', default='PICAI', help='path that the trained models save')
    parser.add_argument('--nThreads', '-j', default=0, type=int, metavar='N',
                        help='number of data loading threads (default: 0)')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=2e-4)
    parser.add_argument("-gpu", type=str, default='0', help='which gpu to choose')
    parser.add_argument("-method", type=str, default='Ours')
    parser.add_argument('-task', default=2, type=int, help='number of tasks')
    parser.add_argument('-base', default=2, type=int, help='number of classes in non_incremental_state')

    # Non-incremental train parameters
    parser.add_argument('--momentum_0', type=float, default=0.9)
    parser.add_argument('--weight-decay_0', type=float, default=5e-4)
    parser.add_argument('-lr_0', type=float, default=0.001,
                        help="learning rate of non_incremental_state")
    parser.add_argument('-BatchSize_0', default=64, type=int, metavar='N',
                        help='mini-batch size Default: 256')
    parser.add_argument('-epochs_0', default=50, type=int, metavar='N', help='epochs for non_incremental_state'
                                                                            'training process')

    args = parser.parse_args()

    if args.data == "PICAI":
        output_spacing = [0.5, 0.5, 3.0]
        output_size = [224, 224, 16]
        mean_T2w = 209.71
        std_T2w = 134.86
        root = 'DataSet' + '/PICAI'
        traindir1 = glob.glob(os.path.join(root, "train/case_ISUP0/*t2w.mha"))
        traindir2 = glob.glob(os.path.join(root, "train/case_ISUP1/*t2w.mha"))
        traindir3 = glob.glob(os.path.join(root, "train/case_ISUP2/*t2w.mha"))
        traindir4 = glob.glob(os.path.join(root, "train/case_ISUP3/*t2w.mha"))

        testdir1 = glob.glob(os.path.join(root, "test/case_ISUP0/*t2w.mha"))
        testdir2 = glob.glob(os.path.join(root, "test/case_ISUP1/*t2w.mha"))
        testdir3 = glob.glob(os.path.join(root, "test/case_ISUP2/*t2w.mha"))
        testdir4 = glob.glob(os.path.join(root, "test/case_ISUP3/*t2w.mha"))
        num_classes = 4
        label_map = list(range(0, num_classes))

    num_task = args.task
    num_class_per_task = int((num_classes - args.base) / (num_task - 1))

    np.random.seed(args.seed)
    random_perm = np.random.permutation(num_classes)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    class_indexes = []
    for i in range(num_task):
        if i == 0:
            class_index = [0, 1]
            num_class = 2
            Images_T2w_clsss_0 = [sitk.ReadImage(filename) for filename in traindir1]
            Images_T2w_clss_1 = [sitk.ReadImage(filename) for filename in traindir2]
            image_T2w_clss_0_list = slice_data(Images_T2w_clsss_0, "T2w", 0, output_spacing, output_size)
            labels_c1 = np.array(["0"] * len(image_T2w_clss_0_list))
            del Images_T2w_clsss_0
            image_T2w_clss_1_list = slice_data(Images_T2w_clss_1, "T2w", 1, output_spacing, output_size)
            labels_c2 = np.array(["1"] * len(image_T2w_clss_1_list))
            del Images_T2w_clss_1

            training_Tw2 = image_T2w_clss_0_list + image_T2w_clss_1_list
            labels = np.concatenate((labels_c1, labels_c2))
            training_Tw2_numpy = []
            for sitk_image in training_Tw2:
                np_image = sitk.GetArrayFromImage(sitk_image)
                training_Tw2_numpy.append(np_image)
            del training_Tw2

            transformed_training_imgs = augment_data(training_Tw2_numpy, mean_T2w, std_T2w, "data")
            del training_Tw2_numpy
            transformed_training_labels = augment_data(labels, mean_T2w, std_T2w, "target")

            train_dataset_Tw2 = TensorDataset(torch.stack(transformed_training_imgs), transformed_training_labels)
            train_loader_0 = torch.utils.data.DataLoader(train_dataset_Tw2, batch_size=args.BatchSize_0, shuffle=True,
                                                         drop_last=True, num_workers=args.nThreads)
            dictlist = dict(zip(class_index, label_map))
            print(50 * '#')
            print("Start of Non-incremental Task")
            print("base classes number = {}".format(num_class))
            print("Training-set size = " + str(len(train_loader_0.dataset)))
            initial_train_fun(args, train_loader_0, num_class, dictlist)

            classes_number = 2
            class_indexes.append(class_index)
        else:
            class_index = [2, 3]
            Images_T2w_clsss_2 = [sitk.ReadImage(filename) for filename in traindir3]
            Images_T2w_clss_3 = [sitk.ReadImage(filename) for filename in traindir4]
            image_T2w_clss_2_list = slice_data(Images_T2w_clsss_2, "T2w", 0, output_spacing, output_size)
            labels_c3 = np.array(["0"] * len(image_T2w_clss_2_list))
            del Images_T2w_clsss_2
            image_T2w_clss_3_list = slice_data(Images_T2w_clss_3, "T2w", 1, output_spacing, output_size)
            labels_c4 = np.array(["1"] * len(image_T2w_clss_3_list))
            del Images_T2w_clss_3

            training_Tw2 = image_T2w_clss_2_list + image_T2w_clss_3_list
            labels = np.concatenate((labels_c3, labels_c4))
            training_Tw2_numpy = []
            for sitk_image in training_Tw2:
                np_image = sitk.GetArrayFromImage(sitk_image)
                training_Tw2_numpy.append(np_image)
            del training_Tw2

            transformed_training_imgs = augment_data(training_Tw2_numpy, mean_T2w, std_T2w, "data")
            del training_Tw2_numpy
            transformed_training_labels = augment_data(labels, mean_T2w, std_T2w, "target")

            train_dataset_Tw2 = TensorDataset(torch.stack(transformed_training_imgs), transformed_training_labels)

            train_loader = torch.utils.data.DataLoader(train_dataset_Tw2, batch_size=args.BatchSize, shuffle=True,
                                                       drop_last=True, num_workers=args.nThreads)

            classes_number = 2
            class_indexes.append(class_index)

            print("Start of task: {}".format(i))
            print("new classes number = {}".format(num_class_per_task))
            print("Training-set size = " + str(len(train_loader.dataset)))
            train_fun(args, train_loader, i, class_indexes)

            # Load the test data and apply the transformations
            Images_T2w_clsss_2 = [sitk.ReadImage(filename) for filename in testdir3]
            Images_T2w_clss_3 = [sitk.ReadImage(filename) for filename in testdir4]
            image_T2w_clss_2_list = slice_data(Images_T2w_clsss_2, "T2w", 0, output_spacing, output_size)
            labels_c3 = np.array(["0"] * len(image_T2w_clss_2_list))
            del Images_T2w_clsss_2
            image_T2w_clss_3_list = slice_data(Images_T2w_clss_3, "T2w", 1, output_spacing, output_size)
            labels_c4 = np.array(["1"] * len(image_T2w_clss_3_list))
            del Images_T2w_clss_3

            test_Tw2 = image_T2w_clss_2_list + image_T2w_clss_3_list
            labels = np.concatenate((labels_c3, labels_c4))
            test_Tw2_numpy = []
            for sitk_image in test_Tw2:
                np_image = sitk.GetArrayFromImage(sitk_image)
                test_Tw2_numpy.append(np_image)
            del test_Tw2

            transformed_test_imgs = augment_data_test(test_Tw2_numpy, mean_T2w, std_T2w, "data")
            del test_Tw2_numpy
            transformed_test_labels = augment_data_test(labels, mean_T2w, std_T2w, "target")

            test_dataset_Tw2 = TensorDataset(torch.stack(transformed_test_imgs), transformed_test_labels)

            test_loader = torch.utils.data.DataLoader(test_dataset_Tw2, batch_size=64, shuffle=False)
            print(50 * '#')
            print("Start of Evaluating")
            acc1 = test1(args, test_loader, 1)
            acc2 = test2(args, test_loader, 2)

            Images_T2w_clsss_0 = [sitk.ReadImage(filename) for filename in testdir1]
            Images_T2w_clss_1 = [sitk.ReadImage(filename) for filename in testdir2]
            image_T2w_clss_0_list = slice_data(Images_T2w_clsss_0, "T2w", 0, output_spacing, output_size)
            labels_c1 = np.array(["0"] * len(image_T2w_clss_0_list))
            del Images_T2w_clsss_0
            image_T2w_clss_1_list = slice_data(Images_T2w_clss_1, "T2w", 1, output_spacing, output_size)
            labels_c2 = np.array(["1"] * len(image_T2w_clss_1_list))
            del Images_T2w_clss_1

            test_Tw2 = image_T2w_clss_0_list + image_T2w_clss_1_list
            labels = np.concatenate((labels_c1, labels_c2))
            test_Tw2_numpy = []
            for sitk_image in test_Tw2:
                np_image = sitk.GetArrayFromImage(sitk_image)
                test_Tw2_numpy.append(np_image)
            del test_Tw2

            transformed_test_imgs = augment_data_test(test_Tw2_numpy, mean_T2w, std_T2w, "data")
            del test_Tw2_numpy
            transformed_test_labels = augment_data_test(labels, mean_T2w, std_T2w, "target")

            test_dataset_Tw2 = TensorDataset(torch.stack(transformed_test_imgs), transformed_test_labels)
            test_loader = torch.utils.data.DataLoader(test_dataset_Tw2, batch_size=64, shuffle=False)
            acc3 = test1(args, test_loader, 1)
            acc4 = test2(args, test_loader, 2)

