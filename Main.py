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
    data, roads_names = read_dataset(args.data_filename, args.num_of_weeks, args.num_of_days, args.num_of_hours,
                        args.data_per_hour, args.artificial_missing_ratio)
    
    # set data loader
    train_loader = data_loader_creator(args.device, 'train', data, args.batch_size, True)
    val_loader = data_loader_creator(args.device, 'val', data, args.batch_size, False)
    test_loader = data_loader_creator(args.device, 'test', data, args.batch_size, False)

    # get model's structure. Based on : week, day, hour
    sub_net_name, all_backbones = set_backbones(args.adj_filename, args.num_of_vertices, args.num_of_weeks,
                                                args.num_of_days, args.num_of_hours, args.k, roads_names)
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

        for iter, (week, day, recent, current, real, week_mask, day_mask, recent_mask, mask) in enumerate(train_loader):
            train_w = torch.Tensor(week).to(args.device)
            train_d = torch.Tensor(day).to(args.device)
            train_r = torch.Tensor(recent).to(args.device)
            train_c = torch.Tensor(current).to(args.device)
            train_real = torch.Tensor(real).to(args.device)
            train_mask_w = torch.Tensor(week_mask).to(args.device)
            train_mask_d = torch.Tensor(day_mask).to(args.device)
            train_mask_r = torch.Tensor(recent_mask).to(args.device)
            train_mask = torch.Tensor(mask).to(args.device)

            if len(train_w) < args.batch_size:  # last batch with shorter size
                break  # do better
            metrics = engine.train([train_w, train_d, train_r], train_c, train_real,
                                   [train_mask_w, train_mask_d, train_mask_r], train_mask)

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

        for iter, (week, day, recent, current, real, week_mask, day_mask, recent_mask, mask) in enumerate(val_loader):
            val_w = torch.Tensor(week).to(args.device)
            val_d = torch.Tensor(day).to(args.device)
            val_r = torch.Tensor(recent).to(args.device)
            val_c = torch.Tensor(current).to(args.device)
            val_real = torch.Tensor(real).to(args.device)
            val_mask_w = torch.Tensor(week_mask).to(args.device)
            val_mask_d = torch.Tensor(day_mask).to(args.device)
            val_mask_r = torch.Tensor(recent_mask).to(args.device)
            val_mask = torch.Tensor(mask).to(args.device)

            if len(val_w) < args.batch_size:  # last batch with shorter size
                break  # do better

            metrics = engine.eval([val_w, val_d, val_r], val_c, val_real,
                                  [val_mask_w, val_mask_d, val_mask_r], val_mask)

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
        pp = args.save + "/_epoch_" + str(i) + "_" + str(round(mvalid_loss, 2)) + ".pth"
        torch.save(engine.model.state_dict(), pp)

    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))
    print("Training finished")

    # test
    outputs = []
    ground_truth = []
    mask_ground_truth = []

    # best epoch
    bestid = np.argmin(his_loss)  # best model in epochs
    pp = args.save + "/_epoch_" + str(bestid + 1) + "_" + str(round(his_loss[bestid], 2)) + ".pth"
    engine.model.load_state_dict(torch.load(pp))

    for iter, (week, day, recent, current, real, week_mask, day_mask, recent_mask, mask) in enumerate(test_loader):

        tst_w = torch.Tensor(week).to(args.device)
        tst_d = torch.Tensor(day).to(args.device)
        tst_r = torch.Tensor(recent).to(args.device)
        tst_c = torch.Tensor(current).to(args.device)
        tst_real = torch.Tensor(real).to(args.device)
        tst_mask_w = torch.Tensor(week_mask).to(args.device)
        tst_mask_d = torch.Tensor(day_mask).to(args.device)
        tst_mask_r = torch.Tensor(recent_mask).to(args.device)
        tst_mask = torch.Tensor(mask).to(args.device)

        if len(tst_w) < args.batch_size:  # last batch with shorter size
            break  # do better

        with torch.no_grad():  # no back-propagation
            engine.model.is_test = True
            # ests = engine.model([tst_w, tst_d, tst_r], tst_c, tst_real, [tst_mask_w, tst_mask_d, tst_mask_r], tst_mask)
            tst_data = [tst_w, tst_d, tst_r]
            tst_c = tst_c.unsqueeze(-1)
            tst_data = [torch.cat([d, tst_c], dim=2) for d in tst_data] 
            ests = engine.model(tst_data)  # GCN forward pass

        outputs.extend(ests)
        ground_truth.extend(tst_real)
        mask_ground_truth.extend(tst_mask)

    # save errors: csv file, containing value of each road segment per data sample
    output_handler.segments_error(outputs, ground_truth, mask_ground_truth, args.save_outputs)

    # print best validation
    print("The valid loss on best model is", str(round(his_loss[bestid], 3)))

    # print best test
    # log = 'Evaluate best model on test data fold {}, Test MAE: {:.4f}, Test RMSE: {:.4f}'
    # print(log.format(fold, mklr, jsd))
