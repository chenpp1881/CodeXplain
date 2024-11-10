import logging
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from IFmodule import InformationFusionBlock

logger = logging.getLogger(__name__)


def all_metrics(y_true, y_pred, is_training=False):
    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)

    epsilon = 1e-7

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    acc = (tp + tn) / (tp + tn + fp + fn + epsilon)

    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    f1.requires_grad = is_training

    return f1.item(), precision.item(), recall.item(), acc.item(), tp.item(), tn.item(), fp.item(), fn.item()


class Trainer():
    def __init__(self, args):
        self.args = args
        self.start_epoch = 0
        self.initial_f1 = 0.0

        self.text_seq_len = args.max_length
        IF = InformationFusionBlock(
            hidden=768,
            dropout=0.2,
            args=args,
        )
        self.optimizer = optim.AdamW(IF.parameters(), lr=args.lr_IF)
        self.criterion = torch.nn.CrossEntropyLoss().to(args.device)

        IF = torch.nn.DataParallel(IF, device_ids=[0, 1])
        self.model = IF.to(args.device)

        self.results_data = []
        if args.resume_file:
            assert os.path.exists(args.resume_file), 'checkpoint not found!'
            logger.info('loading model checkpoint from %s..' % args.resume_file)
            checkpoint = torch.load(args.resume_file)
            IF.load_state_dict(checkpoint['state_dict'], strict=False)
            # self.start_epoch = checkpoint['k'] + 1

    def savemodel(self, k):
        if not os.path.exists(os.path.join(self.args.savepath, self.args.dataset)):
            os.mkdir(os.path.join(self.args.savepath, self.args.dataset))
        torch.save({'state_dict': self.model.state_dict(),
                    'k': k,
                    'optimizer': self.optimizer.state_dict()},
                    os.path.join(self.args.savepath, self.args.dataset,
                                f'model_{k}.pth'))
        logger.info(f'save:{k}.pth')

    def train_classicication(self, dataset):
        logging.info(f'Start classicition training!')
        train_loader = DataLoader(dataset[0], batch_size=self.args.batch_size, shuffle=True, drop_last=True)
        validation_loader = DataLoader(dataset[1], batch_size=self.args.batch_size, shuffle=True, drop_last=True)
        for epoch in range(self.start_epoch, self.args.epoch + self.start_epoch):
            self.train_cla_epoch(epoch, train_loader)
            logging.info('Epoch %d finished' % epoch)
            f1 = self.eval_epoch(validation_loader)
            if f1 >= self.initial_f1:
                self.savemodel(epoch)
                self.initial_f1 = f1
        result_df = pd.DataFrame(self.results_data, columns=['f1', 'precision', 'recall', 'acc'])
        save_path = self.args.savepath + '/result_record_val_' + self.args.dataset + '.csv'
        result_df.to_csv(save_path, mode='a', index=False, header=True)

    def train_cla_epoch(self, epoch, train_loader):
        self.model.train()

        loss_num = 0.0
        all_labels = []
        all_preds = []

        logger.info(f"epoch {epoch} training star!")
        pbar = tqdm(train_loader, total=len(train_loader))
        for i, (code_tokens, desc_tokens, label) in enumerate(pbar):
            outputs = self.model(code_tokens, desc_tokens)
            label = label.to(self.args.device)
            loss = self.criterion(outputs, label)

            _, predicted = torch.max(outputs.data, dim=1)
            all_preds.extend(predicted)
            all_labels.extend(label)
            self.optimizer.zero_grad()
            loss.sum().backward()
            self.optimizer.step()

            loss_num += loss.sum().item()

            pbar.set_description(f'epoch: {epoch}')
            # loss and step
            pbar.set_postfix(index=i, loss=loss.sum().item())

    def eval_epoch(self, dev_loader):
        self.model.eval()

        all_labels = []
        all_preds = []
        with torch.no_grad():
            for code_tokens, desc_tokens, label in tqdm(dev_loader):
                outputs = self.model(code_tokens, desc_tokens)
                _, predicted = torch.max(outputs.data, dim=1)
                all_preds.extend(predicted)
                all_labels.extend(label)

            tensor_labels, tensor_preds = torch.tensor(all_labels), torch.tensor(all_preds)
            f1, precision, recall, acc, tp, tn, fp, fn = all_metrics(tensor_labels, tensor_preds)
            self.results_data.append([f1, precision, recall, acc])
            logger.info(
                'Valid set -f1: {:.4f}, precision: {:.4f}, recall: {:.4f}, acc: {:.4f}'
                    .format(f1, precision, recall, acc))
            logger.info('Valid set -tp: {:.4f}, tn: {:.4f}, fp: {:.4f}, fn: {:.4f}'.format(tp, tn, fp, fn))
            return f1

    def test_cla(self, dataset):
        logging.info(f'Start classicition testing!')
        test_loader = DataLoader(dataset, batch_size=self.args.batch_size_IF, shuffle=True, drop_last=True)
        self.eval_epoch(test_loader)
