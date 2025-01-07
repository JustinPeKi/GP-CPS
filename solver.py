import numpy as np
import torch
import torch.nn.functional as F
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from loss_function import loss_fn_semi, loss_fn_latent, loss_fn_dice, loss_fn_bce, acc_fn


class Solver:
    def __init__(self, args, optim=torch.optim.Adam):
        self.args = args
        self.optim = optim
        self.NumClass = self.args.num_class
        self.lr = args.lr
        self.loss_semi = loss_fn_semi
        self.loss_latent = loss_fn_latent
        self.loss_bce = loss_fn_bce
        self.loss_dice = loss_fn_dice
        self.acc_fn = acc_fn


    def create_exp_directory(self, exp_id):
        if not os.path.exists('models/' + str(exp_id)):
            os.makedirs('models/' + str(exp_id))

        csv = str(exp_id)+'_results'+'.csv'
        with open(os.path.join(self.args.save, csv), 'w') as f:
            f.write('epoch, acc \n')
        txt = str(exp_id) + '.txt'
        with open(os.path.join(self.args.save, txt), 'w') as f:
            f.write(self.args.tip)

    def train(self, model1, model2, train_labeled_loader, train_unlabeled_loader, val_loader, exp_id, num_epochs=10):
        torch.cuda.set_device(self.args.cuda_id)
        optimizer1 = self.optim(model1.parameters(), lr=self.lr)
        optimizer2 = self.optim(model2.parameters(), lr=self.lr)

        print("Start Training")

        self.create_exp_directory(exp_id)

        loss_list = []
        loss_dice_list = []
        loss_dice_vali_list = []
        loss_ce_list = []
        loss_ce_vali_list = []
        loss_semi_list = []
        loss_latent_list = []
        grad_edge_list = []
        grad_inside_list = []
        grad_outside_list = []
        loss_mix_list = []
        acc_list = []

        best_p = 0  # 记录最好acc值
        best_epo = 0

        if self.args.test_only:
            self.vali_epoch(model1, val_loader)
        else:
            pbar1 = tqdm(total=num_epochs, position=0, desc="Epoch", leave=False, colour='green', unit='epoch')
            pbar1.set_postfix_str(' Tot:unknown' +
                                  ' Sup:unknown' +
                                  ' Semi:unknown' +
                                  ' Lat:unknown' +
                                  ' dice:unknown' +
                                  ' dice_p:unknown' +
                                  ' bce:unknown' +
                                  ' bce_p:unknown'+
                                  ' acc:unknown')

            for epoch in range(num_epochs):
                model1.train()
                model2.train()

                i_batch = 0

                pbar2 = tqdm(total=len(train_labeled_loader), position=1, desc="Iter", leave=False, colour='green',
                             unit='iter')
                if epoch == 0:
                    pbar2.set_postfix_str(' Tot:unknown' +
                                          ' Sup:unknown' +
                                          ' Semi:unknown' +
                                          ' Lat:unknown' +
                                          ' dice1:unknown' +
                                          ' dice2:unknown' +
                                          ' bce:unknown')
                for sample_labeled, sample_unlabeled in zip(train_labeled_loader, train_unlabeled_loader):
                    i_batch += 1
                    image_labeled = sample_labeled[0].float().cuda()
                    label = sample_labeled[1].unsqueeze(1).float().cuda()
                    unlabeled = sample_unlabeled[0].float().cuda()
                    eps = torch.rand(1).cuda()
                    # eps = torch.tensor(0.2).cuda()

                    len_label = len(label)
                    len_unlabel = len(unlabeled)
                    len_i2 = int(len_unlabel / len_label)




                    for i2 in range(len_i2):
                        out3, _ = model1(unlabeled[i2 * len_label:(i2 + 1) * len_label])
                        out4, _ = model2(unlabeled[i2 * len_label:(i2 + 1) * len_label])
                        mix_lu = eps * image_labeled + (1 - eps) * unlabeled[i2 * len_label:(i2 + 1) * len_label]
                        mix_lu.requires_grad_()
                        image_labeled.requires_grad_()


                        out_mix1, _ = model1(mix_lu)
                        label_mix1 = eps * label + (1 - eps) * out4
                        label_mix1_detach = label_mix1.detach()
                        out_mix2, _ = model2(mix_lu)
                        label_mix2 = eps * label + (1 - eps) * out3
                        label_mix2_detach = label_mix2.detach()

                        mix_lu.grad = None
                        out_mix_mean1 = torch.sum(out_mix1)
                        out_mix_mean2 = torch.sum(out_mix2)
                        (out_mix_mean1+out_mix_mean2).backward(retain_graph=True)

                        # image_labeled.grad = None
                        # out_mix_mean1 = torch.sum(out1)
                        # out_mix_mean1.backward(retain_graph=True)
                        # out_mix_mean2 = torch.sum(out2)
                        # out_mix_mean2.backward(retain_graph=True)

                        if self.args.lam_lip == 0:
                            mix_grad = mix_lu.grad / 2
                        else:
                            mix_grad = torch.autograd.grad(
                                outputs=(torch.sum(out_mix1) + torch.sum(out_mix2)) / 2,
                                inputs=mix_lu,
                                create_graph=True,
                                retain_graph=True)[0]
                        del mix_lu

                        label_edge = self.find_edge(label)
                        # label_edge = torch.bitwise_and(label.int(), label_edge.int())
                        fake_label = torch.where((out3[:len(image_labeled)] > 0.5), torch.tensor(1.0).cuda(), torch.tensor(0.0).cuda())
                        mix_label = torch.bitwise_or(fake_label.int(), label.int())
                        mix_label = label
                        label_inside = torch.bitwise_and(mix_label.int(), (1-label_edge).int())
                        label_outside = torch.bitwise_and((1 - label_inside).int(), (1 - label_edge).int())

                        label_expand_edge = label_edge.repeat(1, 3, 1, 1)
                        label_expand_edge.requires_grad = False
                        # grad_edge = torch.sum(torch.abs(mix_grad) * label_expand_edge) / label_expand_edge.sum()
                        # loss_lip_edge = torch.sum(torch.abs(torch.abs(mix_grad) - self.args.grad_edge) * label_expand_edge) / label_expand_edge.sum()
                        label_expand_inside = label_inside.repeat(1, 3, 1, 1)
                        label_expand_inside.requires_grad = False
                        # grad_inside = torch.sum(torch.abs(mix_grad) * label_expand_inside) / label_expand_inside.sum()
                        # loss_lip_inside = torch.sum(torch.abs(torch.abs(mix_grad) - self.args.grad_inside) * label_expand_inside) / label_expand_inside.sum()
                        label_expand_outside = label_outside.repeat(1, 3, 1, 1)
                        label_expand_outside.requires_grad = False
                        grad_outside = torch.sum(torch.abs(mix_grad) * label_expand_outside) / label_expand_outside.sum()
                        loss_lip_outside = torch.sum(torch.abs(torch.abs(mix_grad) - self.args.grad_outside) * label_expand_outside) / label_expand_outside.sum()

                        # loss_lip = loss_lip_edge / 3 + loss_lip_inside / 3 + loss_lip_outside / 3
                        loss_lip = loss_lip_outside
                        loss_mix = self.loss_dice(out_mix1, label_mix1_detach) + self.loss_dice(out_mix2, label_mix2_detach)
                        optimizer1.zero_grad()
                        optimizer2.zero_grad()
                        ((loss_lip * self.args.lam_lip + loss_mix) / len_i2).backward()
                        optimizer1.step()
                        optimizer2.step()
                        del out3, out4, out_mix1, out_mix2, out_mix_mean1, out_mix_mean2, loss_lip, loss_mix, loss_lip_outside

                    optimizer1.zero_grad()
                    optimizer2.zero_grad()

                    out1, latent1 = model1(image_labeled)
                    out2, latent2 = model2(image_labeled)
                    loss_dice1 = self.loss_dice(out1, label)
                    loss_dice2 = self.loss_dice(out2, label)
                    loss_ce1 = self.loss_bce(out1, label)
                    loss_ce2 = self.loss_bce(out2, label)
                    loss_ce = loss_ce1 + loss_ce2
                    loss_supervision = ((loss_dice1 + loss_dice2) * self.args.lam_dice +
                                        loss_ce * self.args.lam_ce)


                    # with amp.scale_loss(loss, optimizer) as scale_loss:
                    #     scale_loss.backward()
                    loss_supervision.backward()

                    optimizer1.step()
                    optimizer2.step()



                    pbar2.set_postfix_str(' Sup:%.3f' % (loss_supervision.item()))
                    pbar2.update(1)


                dice_p, ce_p, acc = self.vali_epoch(model1, val_loader)
                acc_list.append(acc.item())
                if best_p < acc:
                    best_p = acc
                    best_epo = epoch
                    torch.save(model1.state_dict(), 'models/' + str(exp_id) + '/best_model1.pth')
                    torch.save(model2.state_dict(), 'models/' + str(exp_id) + '/best_model2.pth')
                if (epoch + 1) % self.args.save_per_epochs == 0:
                    torch.save(model1.state_dict(), 'models/' + str(exp_id) + '/' + str(epoch + 1) + '_model1.pth')
                    torch.save(model2.state_dict(), 'models/' + str(exp_id) + '/' + str(epoch + 1) + '_model2.pth')
                pbar1.set_postfix_str(' acc:%.3f' % (acc.item()))
                pbar1.update(1)

            csv = str(exp_id) + '_results' + '.csv'
            with open(os.path.join(self.args.save, csv), 'a') as f:
                f.write('%03d,%0.6f \n' % (
                    best_epo,
                    best_p
                ))
            # writer.close()
            self.draw(loss_list, loss_semi_list, loss_latent_list, loss_dice_list, loss_dice_vali_list, loss_ce_list,
                      loss_ce_vali_list, acc_list, grad_edge_list, grad_inside_list, grad_outside_list, loss_mix_list)
            print('FINISH.')

    def vali_epoch(self, model, loader) -> torch.tensor:
        model.eval()
        dice_vali_sum = 0
        ce_vali_sum = 0
        acc_sum = 0
        with torch.no_grad():
            for j_batch, vali_data in enumerate(loader):
                image_vali = vali_data[0].float().cuda()
                label_vali = vali_data[1].unsqueeze(1).float().cuda()

                output_vali, _ = model(image_vali)
                dice_vali = self.loss_dice(output_vali, label_vali)
                ce_vali = self.loss_bce(output_vali, label_vali)
                acc = self.acc_fn(output_vali, label_vali)
                dice_vali_sum += dice_vali
                ce_vali_sum += ce_vali
                acc_sum += acc
            dice_vali_mean = dice_vali_sum / len(loader)
            ce_vali_mean = ce_vali_sum / len(loader)
            acc_mean = acc_sum / len(loader)

        return dice_vali_mean, ce_vali_mean, acc_mean

    def find_edge(self, label):
        kernel = torch.ones(1, 1, 3, 3).cuda()
        edges = F.conv_transpose2d(label, kernel, padding=1, bias=None)
        mask = (edges > 0) & (edges < 9)
        return mask.float()


    def draw(self, loss_list, loss_semi_list, loss_latent_list, loss_dice_list, loss_dice_vali_list, loss_ce_list,
             loss_ce_vali_list, acc_list, grad_edge_list, grad_inside_list, grad_outside_list, loss_mix_list):
        # 检查保存路径是否存在，如果不存在则创建
        if not os.path.exists(self.args.save):
            os.makedirs(self.args.save)

        plt.figure(figsize=(8, 6))
        plt.plot(loss_list, color='c', label='Loss')
        plt.grid(True)
        plt.title('Loss over epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        min_loss = min(loss_list)
        min_epoch = loss_list.index(min_loss)
        plt.plot(min_epoch, min_loss, 'o', color='c', markersize=10)
        plt.axvline(x=min_epoch, linestyle='--', linewidth=0.5)
        plt.text(min_epoch, min_loss, f'{min_loss:.4f}')
        plt.savefig(os.path.join(self.args.save, str(self.args.exp_id) + '_loss.png'))
        plt.close()

        plt.figure(figsize=(8, 6))
        plt.plot(loss_semi_list, color='c', label='Loss-semi')
        plt.grid(True)
        plt.title('Loss-semi over epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss-semi')
        plt.legend()
        min_loss = min(loss_semi_list)
        min_epoch = loss_semi_list.index(min_loss)
        plt.plot(min_epoch, min_loss, 'o', color='c', markersize=10)
        plt.axvline(x=min_epoch, linestyle='--', linewidth=0.5)
        plt.text(min_epoch, min_loss, f'{min_loss:.4f}')
        plt.savefig(os.path.join(self.args.save, str(self.args.exp_id) + '_loss_semi.png'))
        plt.close()

        plt.figure(figsize=(8, 6))
        plt.plot(loss_latent_list, color='c', label='loss-latent')
        plt.grid(True)
        plt.title('Loss-latent over epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss-latent')
        plt.legend()
        min_loss = min(loss_latent_list)
        min_epoch = loss_latent_list.index(min_loss)
        plt.plot(min_epoch, min_loss, 'o', color='c', markersize=10)
        plt.axvline(x=min_epoch, linestyle='--', linewidth=0.5)
        plt.text(min_epoch, min_loss, f'{min_loss:.4f}')
        plt.savefig(os.path.join(self.args.save, str(self.args.exp_id) + '_loss_latent.png'))
        plt.close()

        plt.figure(figsize=(8, 6))
        plt.plot(acc_list, color='c', label='acc')
        plt.grid(True)
        plt.title('Acc over epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Acc')
        plt.legend()
        min_loss = max(acc_list)
        min_epoch = acc_list.index(min_loss)
        plt.plot(min_epoch, min_loss, 'o', color='c', markersize=10)
        plt.axvline(x=min_epoch, linestyle='--', linewidth=0.5)
        plt.text(min_epoch, min_loss, f'{min_loss:.4f}')
        plt.savefig(os.path.join(self.args.save, str(self.args.exp_id) + '_acc.png'))
        plt.close()

        plt.figure(figsize=(8, 6))
        plt.plot(grad_edge_list, color='c', label='grad_edge')
        plt.grid(True)
        plt.title('Grad_edge over epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Grad_edge')
        plt.legend()
        min_loss = min(grad_edge_list)
        min_epoch = grad_edge_list.index(min_loss)
        plt.plot(min_epoch, min_loss, 'o', color='c', markersize=10)
        plt.axvline(x=min_epoch, linestyle='--', linewidth=0.5)
        plt.text(min_epoch, min_loss, f'{min_loss:.4f}')
        plt.savefig(os.path.join(self.args.save, str(self.args.exp_id) + '_grad_edge.png'))
        plt.close()

        plt.figure(figsize=(8, 6))
        plt.plot(grad_inside_list, color='c', label='grad_inside')
        plt.grid(True)
        plt.title('Grad_inside over epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Grad_inside')
        plt.legend()
        min_loss = min(grad_inside_list)
        min_epoch = grad_inside_list.index(min_loss)
        plt.plot(min_epoch, min_loss, 'o', color='c', markersize=10)
        plt.axvline(x=min_epoch, linestyle='--', linewidth=0.5)
        plt.text(min_epoch, min_loss, f'{min_loss:.4f}')
        plt.savefig(os.path.join(self.args.save, str(self.args.exp_id) + '_grad_inside.png'))
        plt.close()

        plt.figure(figsize=(8, 6))
        plt.plot(grad_outside_list, color='c', label='grad_outside')
        plt.grid(True)
        plt.title('Grad_outside over epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Grad_outside')
        plt.legend()
        min_loss = min(grad_outside_list)
        min_epoch = grad_outside_list.index(min_loss)
        plt.plot(min_epoch, min_loss, 'o', color='c', markersize=10)
        plt.axvline(x=min_epoch, linestyle='--', linewidth=0.5)
        plt.text(min_epoch, min_loss, f'{min_loss:.4f}')
        plt.savefig(os.path.join(self.args.save, str(self.args.exp_id) + '_grad_outside.png'))
        plt.close()

        plt.figure(figsize=(8, 6))
        plt.plot(loss_mix_list, color='c', label='mix')
        plt.grid(True)
        plt.title('Mix over epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Mix')
        plt.legend()
        min_loss = min(loss_mix_list)
        min_epoch = loss_mix_list.index(min_loss)
        plt.plot(min_epoch, min_loss, 'o', color='c', markersize=10)
        plt.axvline(x=min_epoch, linestyle='--', linewidth=0.5)
        plt.text(min_epoch, min_loss, f'{min_loss:.4f}')
        plt.savefig(os.path.join(self.args.save, str(self.args.exp_id) + '_mix.png'))
        plt.close()

        plt.figure(figsize=(8, 6))
        plt.plot(loss_dice_list, color='c', label='Loss Dice (Train)')
        plt.plot(loss_dice_vali_list, color='r', label='Loss Dice (Validation)')
        plt.grid(True)
        plt.title('Loss Dice over epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss Dice')
        plt.legend()
        min_loss = min(loss_dice_list)
        min_epoch = loss_dice_list.index(min_loss)
        plt.plot(min_epoch, min_loss, 'o', color='c', markersize=10)
        plt.axvline(x=min_epoch, linestyle='--', linewidth=0.5)
        plt.text(min_epoch, min_loss, f'{min_loss:.4f}')
        min_loss = min(loss_dice_vali_list)
        min_epoch = loss_dice_vali_list.index(min_loss)
        plt.plot(min_epoch, min_loss, 'o', color='r', markersize=10)
        plt.axvline(x=min_epoch, linestyle='--', linewidth=0.5)
        plt.text(min_epoch, min_loss, f'{min_loss:.4f}')
        plt.savefig(os.path.join(self.args.save, str(self.args.exp_id) + '_loss_dice.png'))
        plt.close()

        plt.figure(figsize=(8, 6))
        plt.plot(loss_ce_list, color='c', label='Loss ce (Train)')
        plt.plot(loss_ce_vali_list, color='r', label='Loss ce (Validation)')
        plt.grid(True)
        plt.title('Loss ce over epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss ce')
        plt.legend()
        min_loss = min(loss_ce_list)
        min_epoch = loss_ce_list.index(min_loss)
        plt.plot(min_epoch, min_loss, 'o', color='c', markersize=10)
        plt.axvline(x=min_epoch, linestyle='--', linewidth=0.5)
        plt.text(min_epoch, min_loss, f'{min_loss:.4f}')
        min_loss = min(loss_ce_vali_list)
        min_epoch = loss_ce_vali_list.index(min_loss)
        plt.plot(min_epoch, min_loss, 'o', color='r', markersize=10)
        plt.axvline(x=min_epoch, linestyle='--', linewidth=0.5)
        plt.text(min_epoch, min_loss, f'{min_loss:.4f}')
        plt.savefig(os.path.join(self.args.save, str(self.args.exp_id) + '_loss_ce.png'))
        plt.close()
