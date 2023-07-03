import copy
import csv
import os
import warnings
from argparse import ArgumentParser

import cv2
import numpy
import torch
import tqdm
from torch.utils import data
from torchvision import transforms

from nets import nn
from utils import util
from utils.dataset import Dataset

warnings.filterwarnings("ignore")

data_dir = os.path.join('..', 'Dataset', 'Metal')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def load_backbone():
    backbone = nn.resnet18()
    state_dict = torch.load('./weights/resnet18.pth')['state_dict']
    backbone.load_state_dict(state_dict)
    for p in backbone.parameters():
        p.requires_grad_ = False
    return backbone


def train(args):
    util.setup_seed()
    util.setup_multi_processes()
    # Model
    model = load_backbone()
    device = torch.device('cuda:0')
    filters = 128 * model.fn.expansion + 256 * model.fn.expansion

    model = model.to(device)
    model_d = nn.Discriminator(filters, filters).to(device)
    model_g = nn.Generator(args, filters, filters).to(device)

    optimizer_d = torch.optim.Adam(model_d.parameters(), 2E-4, weight_decay=1E-5)
    optimizer_g = torch.optim.AdamW(model_g.parameters(), 1E-4)

    dataset = Dataset(os.path.join(data_dir, 'train'),
                      transforms.Compose([transforms.Resize(size=args.input_size),
                                          transforms.CenterCrop(args.input_size),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomVerticalFlip(),
                                          transforms.ToTensor(),
                                          normalize]))

    loader = data.DataLoader(dataset, args.batch_size, True,
                             num_workers=4, pin_memory=True)
    criterion = util.ComputeLoss(device)

    with open('weights/step.csv', 'w') as f:
        best = 0
        writer = csv.DictWriter(f, fieldnames=['epoch',
                                               'train_loss',
                                               'roc_auc', 'auc',
                                               'f1', 'acc'])
        writer.writeheader()
        for epoch in range(args.epochs):
            _ = model.eval()

            model_d.train()
            model_g.train()

            print(('\n' + '%10s' * 5) % ('epoch', 'memory', 'true', 'fake', 'loss'))
            p_bar = tqdm.tqdm(loader, total=len(loader))
            m_loss = util.AverageMeter()
            for samples, _, _ in p_bar:
                optimizer_d.zero_grad()
                optimizer_g.zero_grad()

                samples = samples.to(torch.float).to(device)
                with torch.no_grad():
                    features = model(samples)
                loss, true, fake = criterion(features, model_g, model_d)

                loss.backward()

                optimizer_d.step()
                optimizer_g.step()

                true = true.cpu().item()
                fake = fake.cpu().item()
                loss = loss.detach().cpu().item()

                m_loss.update(loss, samples.size(0))
                gpu = f'{torch.cuda.memory_reserved() / 1E9:.3g}G'  # (GB)
                s = ('%10s' * 2 + '%10.4g' * 3) % (f'{epoch + 1}/{args.epochs}', gpu, true, fake, m_loss.avg)
                p_bar.set_description(s)

            save_g = copy.deepcopy(model_g)
            save_d = copy.deepcopy(model_d)
            last = test(args, device, save_g, save_d)
            writer.writerow({'epoch': str(epoch + 1).zfill(3),
                             'roc_auc': str(f'{last[0]:.3f}'),
                             'auc': str(f'{last[1]:.3f}'),
                             'f1': str(f'{last[2]:.3f}'),
                             'acc': str(f'{last[3]:.3f}'),
                             'train_loss': str(f'{m_loss.avg:.5f}')})
            f.flush()

            state = {'model_g': copy.deepcopy(model_g),
                     'model_d': copy.deepcopy(model_d)}
            torch.save(state, 'weights/last.pt')
            last = util.fitness(numpy.array(last))
            if last > best:
                torch.save(state, 'weights/best.pt')
                best = last
            del state

    torch.cuda.empty_cache()


@torch.no_grad()
def test(args, device=None, model_g=None, model_d=None):
    if device is None:
        device = torch.device('cuda:0')
        model_g = torch.load('./weights/best.pt', device)['model_g'].float()
        model_d = torch.load('./weights/best.pt', device)['model_d'].float()

        model_g.args = args

    model = load_backbone()
    model = model.to(device)

    model_g.eval()
    model_d.eval()
    model.eval()

    dataset = Dataset(os.path.join(data_dir, 'test'),
                      transforms.Compose([transforms.Resize(size=args.input_size),
                                          transforms.CenterCrop(args.input_size),
                                          transforms.ToTensor(),
                                          normalize]))

    scores = []
    labels = []
    loader = data.DataLoader(dataset, args.batch_size // 2,
                             num_workers=2, pin_memory=True)

    desc = ('%10s' * 5) % ('', 'f1', 'acc', 'roc_auc', 'auc')
    for samples, targets, filenames in tqdm.tqdm(loader, desc):
        samples = samples.to(torch.float).to(device)
        labels.extend(targets.numpy().tolist())

        shape = samples.shape[0]
        with torch.no_grad():
            features = model(samples)
            features = model_g(features)

            patch_scores = image_scores = -model_d(features)
            patch_scores = patch_scores.cpu().numpy()
            image_scores = image_scores.cpu().numpy()

            image_scores = image_scores.reshape(shape, -1, *image_scores.shape[1:])
            image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
            was_numpy = False
            if isinstance(image_scores, numpy.ndarray):
                was_numpy = True
                image_scores = torch.from_numpy(image_scores)
            while image_scores.ndim > 2:
                image_scores = torch.max(image_scores, dim=-1).values
            if image_scores.ndim == 2:
                image_scores = torch.max(image_scores, dim=1).values
            if was_numpy:
                image_scores = image_scores.numpy()
            if args.test:
                size = args.input_size // 8
                mean = numpy.array([0.485, 0.456, 0.406])  # mean
                std = numpy.array([0.229, 0.224, 0.225])  # standard deviation
                patch_scores = patch_scores.reshape(shape, -1, *patch_scores.shape[1:])
                patch_scores = patch_scores.reshape(shape, size, size)
                masks = util.score_to_mask(args, patch_scores, device)
                for i, mask in enumerate(masks):
                    filename = os.path.basename(filenames[i])
                    image = samples.cpu().numpy()[i].transpose(1, 2, 0)
                    image = (image * std) + mean
                    image = image * 255
                    mask = mask * 255
                    _, mask = cv2.threshold(mask, 225, 255, cv2.THRESH_BINARY)
                    cv2.imwrite(f'./weights/{filename[:-4]}_mask.png', mask)
                    cv2.imwrite(f'./weights/{filename}', image)

            scores.extend(list(image_scores))
    scores = numpy.squeeze(numpy.array(scores))
    min_scores = numpy.min(scores, -1)
    max_scores = numpy.max(scores, -1)
    scores = (scores - min_scores) / (max_scores - min_scores)
    roc_auc, auc, f1, accuracy = util.compute_metrics(scores, labels)
    print(("%10s" + '%10.3g' * 4) % ("", f1, accuracy, roc_auc, auc))
    return roc_auc, auc, f1, accuracy


def main():
    parser = ArgumentParser()
    parser.add_argument('--input-size', default=288, type=int)
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')

    args = parser.parse_args()

    if not os.path.exists('weights'):
        os.makedirs('weights')

    if args.train:
        train(args)
    if args.test:
        test(args)


if __name__ == "__main__":
    main()
