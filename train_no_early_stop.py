import torch
import numpy as np
import os
import logging
import datetime
import random

from config import DATA, TRAINING
from model.mukara import Mukara
from model.dataloader import load_gt, load_json, get_lsoa_vector

import model.utils as utils

class MukaraTrainer:
    def __init__(self):
        # Set random seeds
        np.random.seed(TRAINING['seed'])
        torch.manual_seed(TRAINING['seed'])
        random.seed(TRAINING['seed'])
        
        self.device = torch.device(TRAINING['device'])

        # Logging
        logging.basicConfig(
            filename='eval/logs/training_log.log',
            level=logging.INFO,
            format='%(asctime)s:%(levelname)s:%(message)s',
            filemode='w'
        )

        # ------ preload node features once (optionally PCA) -------
        lsoa_json = load_json("data/node_features/lsoa21_features_normalized.json")
        node_to_lsoa = load_json("data/node_features/node_to_lsoa.json")

        # build a deterministic ordering of LSOAs
        lsoa_codes = sorted(lsoa_json.keys())
        # stack raw feature matrix
        feat_rows = []
        for code in lsoa_codes:
            v = get_lsoa_vector(lsoa_json[code])  # torch tensor 1D
            feat_rows.append(v.cpu().numpy())
        X = np.vstack(feat_rows).astype(np.float32)  # (N_lsoa, F_raw)

        feature_bank = {}    # {lsoa_code: torch.tensor(feature_dim,)}

        if TRAINING.get("pca", False):
            k = int(TRAINING.get("pca_components", 32))
            Xp, mean, comps = utils.pca_project(X, k)
            # store projected vectors
            for i, code in enumerate(lsoa_codes):
                feature_bank[code] = torch.from_numpy(Xp[i])  # CPU tensor; model moves to device
            # (optional) persist PCA model for reuse/repro
            os.makedirs("data/node_features", exist_ok=True)
            np.savez("data/node_features/pca_model_lsoa21.npz",
                     mean=mean, components=comps, codes=np.array(lsoa_codes, dtype=object))
            logging.info(f"PCA enabled: raw_dim={X.shape[1]}, k={k}")
            print(f"[PCA] raw_dim={X.shape[1]} -> k={k}; saved pca_model_lsoa21.npz")
        else:
            # keep original feature vectors
            for i, code in enumerate(lsoa_codes):
                feature_bank[code] = torch.from_numpy(X[i])


        # Initialize model and optimizer
        self.model = Mukara(feature_bank=feature_bank, node_to_lsoa=node_to_lsoa).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=TRAINING['lr'])
        
        # Load checkpoint if provided
        ckpt_path = TRAINING.get("checkpoint")
        if ckpt_path and os.path.isfile(ckpt_path):
            logging.info(f"Loading model weights from {ckpt_path}")
            print(f"Loading model weights from {ckpt_path}")
            state_dict = torch.load(ckpt_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)

        # Load GT and split
        self.edge_to_gt, self.scaler = load_gt()            
        self.all_edge_ids = list(self.edge_to_gt.keys())
        self.train_ids, self.val_ids = utils.train_test_sampler(self.all_edge_ids)

    def compute_loss(self, gt, pred, loss_function):
        return getattr(utils, loss_function)(gt, pred, self.scaler)

    def train_model(self):
        counter = 0
        for epoch in range(TRAINING['epoch']):
            logging.info(f"Epoch {epoch} started.")
            random.shuffle(self.train_ids)

            for step, edge_id in enumerate(self.train_ids):
                counter += 1
                self.model.train()
                gt = self.edge_to_gt[edge_id]
                pred = self.model(edge_id)

                gt_tensor = torch.tensor([gt], dtype=torch.float32, device=self.device)
                pred = pred.to(self.device)

                loss = self.compute_loss(gt_tensor, pred, TRAINING['loss_function'])
                self.optimizer.zero_grad()
                loss.backward()
                if TRAINING['clip_gradient']:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), TRAINING['clip_gradient'])
                self.optimizer.step()

                # mse = utils.MSE(gt_tensor, pred, self.scaler).item()
                # mae = utils.MAE(gt_tensor, pred, self.scaler).item()
                # geh = utils.MGEH(gt_tensor, pred, self.scaler).item()

                # real_gt = self.scaler.inverse_transform(gt_tensor).item() if self.scaler else gt_tensor.item()
                # real_pred = self.scaler.inverse_transform(pred).item() if self.scaler else pred.item()
                # logging.info(
                #     f"Epoch {epoch} Step {step}: Train Edge {edge_id}, MSE: {mse:.6f}, MAE: {mae:.2f}, MGEH: {geh:.2f}, "
                #     f"GT: {real_gt:.2f}, Pred: {real_pred:.2f}"
                # )

                # Evaluation interval
                if counter % TRAINING['eval_interval'] == 0:
                    self.evaluate_random_sample(epoch, step)

            # Final evaluation at end of epoch
            # self.evaluate_random_sample(epoch, 'end')
            self.save_model(epoch)

    def evaluate_random_sample(self, epoch, step):
        self.model.eval()
        with torch.no_grad():
            sample_train = random.sample(self.train_ids, min(TRAINING['eval_sample_train'], len(self.train_ids)))
            sample_val = random.sample(self.val_ids, min(TRAINING['eval_sample_eval'], len(self.val_ids)))

            for split, sample_ids in [('Train', sample_train), ('Validation', sample_val)]:
                preds, gts = [], []
                for edge_id in sample_ids:
                    gt = self.edge_to_gt[edge_id]
                    pred = self.model(edge_id)
                    if pred is None or pred.numel() == 0:
                        continue
                    preds.append(pred.unsqueeze(0))
                    gts.append(torch.tensor([gt], dtype=torch.float32, device=self.device))

                preds = torch.cat(preds)
                gts = torch.cat(gts)

                mse = utils.MSE(gts, preds, self.scaler).item()
                mae = utils.MAE(gts, preds, self.scaler).item()
                geh = utils.MGEH(gts, preds, self.scaler).item()
                mape = utils.MAPE(gts, preds, self.scaler).item()
                r2 = utils.R_square(gts, preds, self.scaler).item()

                logging.info(
                    f"Epoch {epoch} Step {step} {split} Eval: MSE: {mse:.6f}, MAE: {mae:.6f}, MGEH: {geh:.6f}, MAPE: {mape:.6f}, R2: {r2:.6f}"
                )

    def save_model(self, epoch):
        formatted_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        os.makedirs("param", exist_ok=True)
        os.makedirs(f"param/temp", exist_ok=True)
        model_filename = f"param/temp/epoch{epoch}_{formatted_time}.pt"
        torch.save(self.model.state_dict(), model_filename)
        print("Model saved successfully.")

if __name__ == '__main__':
    print("Initiating model...")
    trainer = MukaraTrainer()

    print("Training started...")
    trainer.train_model()

    for handler in logging.root.handlers[:]:
        handler.close()
        logging.root.removeHandler(handler)
