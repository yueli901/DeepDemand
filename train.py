import torch
import numpy as np
import os
import logging
import datetime
import random

from config import DATA, TRAINING
from model.deepdemand import DeepDemand
from model.dataloader import load_gt, load_json, get_lsoa_vector
import model.utils as utils


class DeepDemandTrainer:
    """
    Adds:
      • Staged LR schedule: [1e-3, 1e-4, 1e-5]
      • Early stopping per LR stage on Validation MGEH
      • "Minimum improvement" = 0.1 (absolute decrease in MGEH)
      • Save best weights-only checkpoint for EACH LR stage
        right before we drop LR (and also for the final 1e-5 stage).
      • Patience counted in *evaluation steps*: 10
    """

    def __init__(self):
        # Set random seeds
        np.random.seed(TRAINING['seed'])
        torch.manual_seed(TRAINING['seed'])
        random.seed(TRAINING['seed'])

        self.device = torch.device(TRAINING['device'])

        # Logging
        os.makedirs('eval/logs', exist_ok=True)
        logging.basicConfig(
            filename=f'eval/logs/{TRAINING["name"]}.log',
            level=logging.INFO,
            format='%(asctime)s:%(levelname)s:%(message)s',
            filemode='w'
        )
        # -------- run folder (unique per training run) --------
        self.run_dir = f"param/{TRAINING['name']}"
        os.makedirs(self.run_dir, exist_ok=True)
        logging.info(f"Run directory: {self.run_dir}")
        print(f"[Run dir] {self.run_dir}")

        # ------ preload features once (optionally PCA) -------
        lsoa_json = load_json(DATA["lsoa_json"])
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
        self.model = DeepDemand(feature_bank=feature_bank, node_to_lsoa=node_to_lsoa).to(self.device)

        # LR schedule (staged)
        self.lr_stages = TRAINING['lr']
        self.current_stage_idx = 0
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr_stages[self.current_stage_idx], weight_decay = 1e-4)

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
        self.train_ids, self.val_ids = utils.get_cv_split(
            self.all_edge_ids,
            k=TRAINING["cv_k"],
            fold_idx=TRAINING["cv_fold"],
            seed=TRAINING["seed"],
        )
        logging.info(f"CV split: k={TRAINING['cv_k']}, fold={TRAINING['cv_fold']} "
                    f"(train={len(self.train_ids)}, val={len(self.val_ids)})")
        
        # Early stopping config (on Validation MGEH)
        self.patience_steps = TRAINING["patience"]                       # evaluations without sufficient improvement
        self.min_improve = 0.1                         # absolute MGEH improvement required
        self.eval_interval = TRAINING['eval_interval'] # evaluations cadence (in training steps)

        # Metric tracking (per stage)
        self.best_val_geh = float('inf')
        self.best_state_dict = None
        self.steps_since_improve = 0
        self.total_eval_calls_in_stage = 0

        logging.info(f"LR schedule: {self.lr_stages}")

    # ------------------------- helpers -------------------------

    def _set_lr(self, lr: float):
        for g in self.optimizer.param_groups:
            g['lr'] = lr
        logging.info(f"LR set to {lr:.6g}")
        print(f"[LR] set to {lr:.6g}")

    def _save_best_for_stage(self):
        if self.best_state_dict is None:
            return
        lr = self.lr_stages[self.current_stage_idx]
        path = os.path.join(self.run_dir, f"best_stage_{self.current_stage_idx+1}_lr{lr:.0e}.pt")
        torch.save(self.best_state_dict, path)
        logging.info(f"Saved BEST model for stage {self.current_stage_idx+1} (lr={lr:.0e}) to {path}")
        print(f"[BEST] Saved best for stage {self.current_stage_idx+1} (lr={lr:.0e}) -> {path}")

    def _reset_stage_trackers(self):
        self.best_val_geh = float('inf')
        self.best_state_dict = None
        self.steps_since_improve = 0
        self.total_eval_calls_in_stage = 0

    def compute_loss(self, gt, pred, loss_function):
        return getattr(utils, loss_function)(gt, pred, self.scaler)

    # ------------------------- core -------------------------

    def train_model(self):
        counter = 0
        # ensure starting LR
        self._set_lr(self.lr_stages[self.current_stage_idx])

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

                # Evaluation interval
                if counter % self.eval_interval == 0:
                    metrics = self.evaluate_random_sample(epoch, step)
                    if metrics is None:
                        continue
                    val_geh = metrics['val_geh']
                    self.total_eval_calls_in_stage += 1

                    # Early stopping criterion on MGEH (lower is better):
                    # consider it an improvement only if MGEH decreased by at least self.min_improve
                    if (self.best_val_geh - val_geh) >= self.min_improve:
                        self.best_val_geh = val_geh
                        # store CPU state_dict for a clean, weights-only save
                        self.best_state_dict = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
                        self.steps_since_improve = 0
                        logging.info(f"[IMPROVED] Stage {self.current_stage_idx+1} best_val_MGEH={self.best_val_geh:.6f}")
                    else:
                        self.steps_since_improve += 1
                        logging.info(
                            f"[NO-IMPROVE] {self.steps_since_improve}/{self.patience_steps} "
                            f"(Delta={self.best_val_geh - val_geh:.6f}, min_improve={self.min_improve})"
                        )

                    # Early stopping for this LR stage
                    if self.steps_since_improve >= self.patience_steps:
                        if self.best_state_dict is not None:
                            self.model.load_state_dict(self.best_state_dict)
                            print(f"[STAGE {self.current_stage_idx+1}] Restored best weights before dropping LR.")
                            logging.info(f"Restored best weights for stage {self.current_stage_idx+1} before LR drop.")

                        # Save best for this stage before dropping LR
                        self._save_best_for_stage()

                        # Move to next LR stage; if done with last stage, exit
                        if self.current_stage_idx + 1 >= len(self.lr_stages):
                            print("[EARLY STOP] Final stage reached and patience exhausted. Training stops.")
                            logging.info("Final LR stage patience exhausted. Stopping training.")
                            return
                        else:
                            # advance stage
                            self.current_stage_idx += 1
                            self._set_lr(self.lr_stages[self.current_stage_idx])
                            # reset trackers for new stage
                            self._reset_stage_trackers()
                            logging.info(f"Entering LR stage {self.current_stage_idx+1}/{len(self.lr_stages)}.")

        # Finished all epochs; also save best of final stage if any
        self._save_best_for_stage()
        logging.info("Training finished (max epochs reached).")

    def evaluate_random_sample(self, epoch, step):
        self.model.eval()
        with torch.no_grad():
            sample_train = random.sample(self.train_ids, min(TRAINING['eval_sample_train'], len(self.train_ids)))
            sample_val = random.sample(self.val_ids, min(TRAINING['eval_sample_eval'], len(self.val_ids)))

            # ----- Training subset (for logging only) -----
            preds, gts = [], []
            for edge_id in sample_train:
                gt = self.edge_to_gt[edge_id]
                pred = self.model(edge_id)
                if pred is None or pred.numel() == 0:
                    continue
                preds.append(pred.unsqueeze(0))
                gts.append(torch.tensor([gt], dtype=torch.float32, device=self.device))
            if preds and gts:
                preds_t = torch.cat(preds)
                gts_t = torch.cat(gts)
                train_mae = utils.MAE(gts_t, preds_t, self.scaler).item()
                train_geh = utils.MGEH(gts_t, preds_t, self.scaler).item()
                train_rmse = utils.RMSE(gts_t, preds_t, self.scaler).item()
                train_r2 = utils.R_square(gts_t, preds_t, self.scaler).item()
                logging.info(
                    f"Epoch {epoch} Step {step} Train Eval: "
                    f"RMSE: {train_rmse:.6f}, MAE: {train_mae:.6f}, MGEH: {train_geh:.6f}, "
                    f"R2: {train_r2:.6f}"
                )

            # ----- Validation subset (used for early stopping) -----
            preds, gts = [], []
            for edge_id in sample_val:
                gt = self.edge_to_gt[edge_id]
                pred = self.model(edge_id)
                if pred is None or pred.numel() == 0:
                    continue
                preds.append(pred.unsqueeze(0))
                gts.append(torch.tensor([gt], dtype=torch.float32, device=self.device))

            if not preds or not gts:
                logging.warning("Empty predictions during evaluation; skipping metrics.")
                return None

            preds_t = torch.cat(preds)
            gts_t = torch.cat(gts)

            val_rmse = utils.RMSE(gts_t, preds_t, self.scaler).item()
            val_mae = utils.MAE(gts_t, preds_t, self.scaler).item()
            val_geh = utils.MGEH(gts_t, preds_t, self.scaler).item()
            val_r2 = utils.R_square(gts_t, preds_t, self.scaler).item()

            logging.info(
                f"Epoch {epoch} Step {step} Validation Eval: "
                f"RMSE: {val_rmse:.6f}, MAE: {val_mae:.6f}, MGEH: {val_geh:.6f}, "
                f"R2: {val_r2:.6f} "
                f"(stage {self.current_stage_idx+1}/{len(self.lr_stages)}, lr={self.lr_stages[self.current_stage_idx]:.0e})"
            )

            return {
                'val_mse': val_rmse,
                'val_mae': val_mae,
                'val_geh': val_geh,
                'val_r2': val_r2
            }

    def save_model(self, epoch):
        formatted_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        path = os.path.join(self.run_dir, f"epoch{epoch}_{formatted_time}.pt")
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")


if __name__ == '__main__':
    print("Initiating model...")
    trainer = DeepDemandTrainer()

    print("Training started...")
    trainer.train_model()

    for handler in logging.root.handlers[:]:
        handler.close()
        logging.root.removeHandler(handler)