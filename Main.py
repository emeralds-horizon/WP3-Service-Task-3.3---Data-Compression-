from data_prepration import read_dataset, data_loader_creator
from model_config import set_backbones
from model import Optimizer
from config import parser
import time
import torch
import numpy as np
import output_handler

args = parser.parse_args()

if __name__ == "__main__":

    print("Reading data...")
    data = read_dataset(args.data_filename, args.num_of_weeks, args.num_of_days, args.num_of_hours,
                        args.points_per_hour)
    # test set (ground truth)
    true_value = data['test']['target']

    # set data loader
    train_loader = data_loader_creator(args.device, 'train', data, args.batch_size, True)
    val_loader = data_loader_creator(args.device, 'val', data, args.batch_size, False)
    test_loader = data_loader_creator(args.device, 'test', data, args.batch_size, False)

    # get model's structure. Based on : week, day, hour
    sub_net_name, all_backbones = set_backbones(args.adj_filename, args.num_of_vertices, args.num_of_weeks,
                                                args.num_of_days, args.num_of_hours, args.k)
    engine = Optimizer(sub_net_name, all_backbones)

    # Train:
    print("start train...", flush=True)

    his_loss = []
    train_time = []
    val_time = []

    for i in range(1, args.epochs + 1):
        train_loss = []
        train_mape = []
        train_rmse = []
        t1 = time.time()

        # shuffle train data:
        train_loader = data_loader_creator(args.device, 'train', data, args.batch_size, True)  # 260

        for iter, (week, day, recent, target) in enumerate(train_loader):
            train_w = torch.Tensor(week).to(args.device)
            train_d = torch.Tensor(day).to(args.device)
            train_r = torch.Tensor(recent).to(args.device)
            train_t = torch.Tensor(target).to(args.device)

            if len(train_w) < args.batch_size:  # last batch with shorter size
                break  # do better

            metrics = engine.train([train_w, train_d, train_r], train_t)

            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])

            log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
            print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]), flush=True)


        t2 = time.time()
        train_time.append(t2 - t1)

        # validation
        valid_loss = []
        s1 = time.time()

        for iter, (week, day, recent, target) in enumerate(val_loader):
            val_w = torch.Tensor(week).to(args.device)
            val_d = torch.Tensor(day).to(args.device)
            val_r = torch.Tensor(recent).to(args.device)
            val_t = torch.Tensor(target).to(args.device)

            if len(val_w) < args.batch_size:  # last batch with shorter size
                break  # do better

            metrics = engine.eval([val_w, val_d, val_r], val_t)
            valid_loss.append(metrics)


        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(i, (s2 - s1)))
        val_time.append(s2 - s1)

        mtrain_loss = np.mean(train_loss)
        mvalid_loss = np.mean(valid_loss)

        his_loss.append(mvalid_loss)

        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Valid Loss: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(i, mtrain_loss, mvalid_loss, (t2 - t1)), flush=True)
        print('**************************************************************')
        pp = args.save + "_epoch_" + str(i) + "_" + str(round(mvalid_loss, 2)) + ".pth"
        torch.save(engine.model.state_dict(), pp)

    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))
    print("Training finished")

    # test
    outputs = []
    grand_truth = []

    # best epoch
    bestid = np.argmin(his_loss)  # best model in epochs
    pp = args.save + "_epoch_" + str(bestid + 1) + "_" + str(round(his_loss[bestid], 2)) + ".pth"
    engine.model.load_state_dict(torch.load(pp))

    for iter, (week, day, recent, target) in enumerate(test_loader):
        tst_w = torch.Tensor(week).to(args.device)
        tst_d = torch.Tensor(day).to(args.device)
        tst_r = torch.Tensor(recent).to(args.device)
        tst_t = torch.Tensor(target).to(args.device)

        if len(tst_w) < args.batch_size:  # last batch with shorter size
            break  # do better

        with torch.no_grad():  # no back-propagation
            engine.model.is_test = True
            preds = engine.model([tst_w, tst_d, tst_r])  # GCN model input: data

        outputs.extend([arr.reshape(-1) for arr in preds])  # temp: when i want to pred current time only
        grand_truth.extend(tst_t)

    # save errors:
    output_handler.segments_error(outputs, grand_truth, args.save_errors)
    print("The valid loss on best model is", str(round(his_loss[bestid], 3)))

    exit()
    log = 'Evaluate best model on test data fold {}, Test MKLR: {:.4f}, Test JSD: {:.4f}, Test Earth Distance: {:.4f}, Test MAE: {:.4f}, Test RMSE: {:.4f}'
    print(log.format(fold, mklr, jsd, wass, h_metrics[0], h_metrics[1]))
    mklrs.append(mklr)
    jsds.append(jsd)
    earths.append(wass)
    maes.append(h_metrics[0])
    rmses.append(h_metrics[1])






