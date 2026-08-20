import torch
import numpy as np
import os
import logging
import datetime
import random

from config import DATA, TRAINING, MODEL_NAME
from model.deepdemand import DeepDemand
from model.dataloader import load_gt, load_json, get_lsoa_vector
import model.utils as utils


ModelCls = DeepDemand
# MODEL_VARIANT swaps in a DeepDemand subclass so an ablation reuses the whole training
# pipeline. 'full' (or unset) is the reported model.
_variant = os.environ.get("MODEL_VARIANT")
if _variant and _variant != "full":
    from model.ablation_models import (
        AblationSharedEncoder, AblationConstantDeterrence, AblationLinearAggregation)
    _vmap = {"shared_encoder": AblationSharedEncoder,
             "constant_deterrence": AblationConstantDeterrence,
             "linear_aggregation": AblationLinearAggregation}
    ModelCls = _vmap[_variant]
    print(f"[variant] MODEL_VARIANT={_variant} -> {ModelCls.__name__}")


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

        # Logging (nested by MODEL_NAME so different models don't overwrite each other).
        # RUN_NAME env overrides the log/checkpoint name so each CV fold is separate.
        run_name = os.environ.get("RUN_NAME", TRAINING["name"])
        log_dir = f'eval/logs/{MODEL_NAME}'
        os.makedirs(log_dir, exist_ok=True)
        logging.basicConfig(
            filename=f"{log_dir}/{run_name}{'_posthoc' if os.environ.get('POSTHOC_CKPT') else ''}.log",
            level=logging.INFO,
            format='%(asctime)s:%(levelname)s:%(message)s',
            filemode='w'
        )
        # -------- run folder (unique per training run, nested by MODEL_NAME) --------
        self.run_dir = f"param/{MODEL_NAME}/{run_name}"
        os.makedirs(self.run_dir, exist_ok=True)
        logging.info(f"Run directory: {self.run_dir}")
        print(f"[Run dir] {self.run_dir}")

        # ------ preload features once (optionally PCA), or attach the OD-index cache ------
        node_to_lsoa = load_json("data/node_features/node_to_lsoa.json")
        use_cache = TRAINING.get("use_od_cache", False) or os.environ.get("USE_OD_CACHE") == "1"

        if use_cache:
            # P0 fast path: prebuilt OD-index cache (no per-step feather reads) +
            # prebuilt post-PCA feature matrix F. feature_bank is derived from F so the
            # feather fallback (if any edge is missing) stays consistent.
            from model.od_cache import OdIndexCache
            cache = OdIndexCache()  # per-edge .npz files
            feat_npz = os.environ.get("FEATURE_NPZ")  # 4.2 vintage-feature sensitivity: swap F (64-dim, same LSOA order, same OD cache)
            if feat_npz:
                _d = np.load(feat_npz, allow_pickle=True)
                Fnp = _d["F"].astype(np.float32)
                assert [str(c) for c in _d["codes"]] == list(cache.lsoa_codes), "FEATURE_NPZ codes != cache lsoa_codes"
                print(f"[FEATURE_NPZ] vintage F from {feat_npz}: {Fnp.shape}")
            else:
                Fnp = cache.feature_matrix().astype(np.float32)
            F_t = torch.from_numpy(Fnp)
            feature_bank = {c: F_t[i] for i, c in enumerate(cache.lsoa_codes)}
            print(f"[Model] {MODEL_NAME} -> {ModelCls.__name__} | OD cache ON "
                  f"(per-edge, dense_nodes={len(cache.dense_to_row)}, F={tuple(F_t.shape)})")
            logging.info(f"OD cache enabled; F={tuple(F_t.shape)}")
            self.model = ModelCls(feature_bank=feature_bank, node_to_lsoa=node_to_lsoa,
                                  od_cache=cache, feature_matrix=F_t).to(self.device)
        else:
            lsoa_json = load_json(DATA["lsoa_json"])
            lsoa_codes = sorted(lsoa_json.keys())
            feat_rows = [get_lsoa_vector(lsoa_json[code]).cpu().numpy() for code in lsoa_codes]
            X = np.vstack(feat_rows).astype(np.float32)

            feature_bank = {}
            if TRAINING.get("pca", False):
                k = int(TRAINING.get("pca_components", 32))
                Xp, mean, comps = utils.pca_project(X, k)
                for i, code in enumerate(lsoa_codes):
                    feature_bank[code] = torch.from_numpy(Xp[i])
                os.makedirs("data/node_features", exist_ok=True)
                np.savez("data/node_features/pca_model_lsoa21.npz",
                         mean=mean, components=comps, codes=np.array(lsoa_codes, dtype=object))
                logging.info(f"PCA enabled: raw_dim={X.shape[1]}, k={k}")
                print(f"[PCA] raw_dim={X.shape[1]} -> k={k}; saved pca_model_lsoa21.npz")
            else:
                for i, code in enumerate(lsoa_codes):
                    feature_bank[code] = torch.from_numpy(X[i])

            print(f"[Model] Using: {MODEL_NAME} -> {ModelCls.__name__}")
            logging.info(f"Model class: {MODEL_NAME} -> {ModelCls.__name__}")
            self.model = ModelCls(feature_bank=feature_bank, node_to_lsoa=node_to_lsoa).to(self.device)

        # LR schedule (staged)
        _lr_env = os.environ.get("LR")
        self.lr_stages = [float(x) for x in _lr_env.split(",")] if _lr_env else TRAINING['lr']
        self.current_stage_idx = 0
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr_stages[self.current_stage_idx], weight_decay = 1e-4)
        # Gradient accumulation over N edges per optimizer step (env GRAD_ACCUM overrides config).
        self.grad_accum = int(os.environ.get("GRAD_ACCUM", TRAINING.get("grad_accum", 1)))
        if self.grad_accum > 1:
            print(f"[train] gradient accumulation: {self.grad_accum} steps / optimizer update")
            logging.info(f"grad_accum={self.grad_accum}")
        self.batch_edges = int(os.environ.get("BATCH_EDGES", TRAINING.get("batch_edges", 1)))
        if self.batch_edges > 1:
            print(f"[train] batched forward: {self.batch_edges} edges / step")
            logging.info(f"batch_edges={self.batch_edges}")

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
        # CV_TYPE ('random'|'spatial') + CV_FOLD select protocol/fold. The held-out fold is
        # the TEST set. THREE_WAY=1 carves a VAL subset out of the training pool for
        # early-stopping ONLY (test stays untouched -> unbiased); else 2-way (val == test,
        # the submitted protocol).
        cv_type = os.environ.get("CV_TYPE", TRAINING.get("cv_type", "random"))
        cv_fold = int(os.environ.get("CV_FOLD", TRAINING["cv_fold"]))
        three_way = os.environ.get("THREE_WAY", "0") == "1"
        val_frac = float(os.environ.get("VAL_FRAC", "0.125"))
        self.train_ids, self.val_ids, self.test_ids = utils.make_split(
            self.all_edge_ids, cv_type=cv_type, cv_fold=cv_fold, three_way=three_way,
            val_frac=val_frac, seed=TRAINING["seed"], cv_k=TRAINING["cv_k"])
        logging.info(f"{cv_type} CV fold={cv_fold} three_way={three_way}: "
                     f"train={len(self.train_ids)} val={len(self.val_ids)} test={len(self.test_ids)}")
        
        # Early stopping config (on Validation MGEH)
        self.patience_steps = int(os.environ.get("PATIENCE", TRAINING["patience"]))  # evals without sufficient improvement (env PATIENCE overrides)
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

    def _predict_batch(self, batch):
        """Predictions for a list of edges, as a [len(batch)] tensor. Uses the batched
        segment-sum forward when batch_edges>1 and the OD cache is attached;
        otherwise per-edge forward() (preserves the single-edge path exactly)."""
        if self.batch_edges > 1 and getattr(self.model, "od_cache", None) is not None:
            return self.model.forward_batch(batch)
        return torch.cat([self.model(e).reshape(1) for e in batch]).to(self.device)

    def _eval_and_maybe_stop(self, epoch, step) -> bool:
        """Validation eval + early-stopping + staged-LR control. Returns True to stop."""
        metrics = self.evaluate_random_sample(epoch, step)
        if metrics is None:
            return False
        val_geh = metrics['val_geh']
        self.total_eval_calls_in_stage += 1

        # improvement = MGEH dropped by at least self.min_improve
        if (self.best_val_geh - val_geh) >= self.min_improve:
            self.best_val_geh = val_geh
            self.best_state_dict = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
            self.steps_since_improve = 0
            logging.info(f"[IMPROVED] Stage {self.current_stage_idx+1} best_val_MGEH={self.best_val_geh:.6f}")
        else:
            self.steps_since_improve += 1
            logging.info(f"[NO-IMPROVE] {self.steps_since_improve}/{self.patience_steps} "
                         f"(Delta={self.best_val_geh - val_geh:.6f}, min_improve={self.min_improve})")

        if self.steps_since_improve >= self.patience_steps:
            if self.best_state_dict is not None:
                self.model.load_state_dict(self.best_state_dict)
                print(f"[STAGE {self.current_stage_idx+1}] Restored best weights before dropping LR.")
                logging.info(f"Restored best weights for stage {self.current_stage_idx+1} before LR drop.")
            self._save_best_for_stage()
            if self.current_stage_idx + 1 >= len(self.lr_stages):
                print("[EARLY STOP] Final stage reached and patience exhausted. Training stops.")
                logging.info("Final LR stage patience exhausted. Stopping training.")
                return True
            self.current_stage_idx += 1
            self._set_lr(self.lr_stages[self.current_stage_idx])
            self._reset_stage_trackers()
            logging.info(f"Entering LR stage {self.current_stage_idx+1}/{len(self.lr_stages)}.")
        return False

    def train_model(self):
        counter = 0
        self._set_lr(self.lr_stages[self.current_stage_idx])
        B = self.batch_edges   # edges per step; B=1 is single-edge SGD

        for epoch in range(TRAINING['epoch']):
            logging.info(f"Epoch {epoch} started.")
            random.shuffle(self.train_ids)
            last_eval, batch_idx = 0, 0

            for start in range(0, len(self.train_ids), B):
                batch = self.train_ids[start:start + B]
                counter += len(batch)
                batch_idx += 1
                self.model.train()

                preds = self._predict_batch(batch)
                gts = torch.tensor([self.edge_to_gt[e] for e in batch],
                                   dtype=torch.float32, device=self.device)
                loss = self.compute_loss(gts, preds, TRAINING['loss_function']) / self.grad_accum
                loss.backward()
                if batch_idx % self.grad_accum == 0:
                    if TRAINING['clip_gradient']:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), TRAINING['clip_gradient'])
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                # evaluate roughly every eval_interval EDGES (counter steps by len(batch))
                if counter - last_eval >= self.eval_interval:
                    last_eval = counter
                    if self._eval_and_maybe_stop(epoch, start):
                        return

        self._save_best_for_stage()
        logging.info("Training finished (max epochs reached).")

    def _predict(self, ids):
        """Forward all edges in `ids` (eval mode, no grad). Returns (kept_edge_ids, gt_z, pred_z)
        tensors in scaler space, or None if empty."""
        self.model.eval()
        with torch.no_grad():
            kept, preds, gts = [], [], []
            for edge_id in ids:
                p = self.model(edge_id)
                if p is None or p.numel() == 0:
                    continue
                kept.append(edge_id); preds.append(p.reshape(1))
                gts.append(torch.tensor([self.edge_to_gt[edge_id]], dtype=torch.float32, device=self.device))
            if not preds:
                return None
            return kept, torch.cat(gts), torch.cat(preds)

    def evaluate_random_sample(self, epoch, step):
        # Early-stopping signal: full validation set, MGEH only -- the ONLY metric used on the fly.
        # Every other metric is computed post-hoc from the saved per-edge predictions (save_outputs).
        r = self._predict(self.val_ids)
        if r is None:
            logging.warning("Empty val predictions during evaluation; skipping.")
            return None
        _, G, P = r
        val_geh = utils.MGEH(G, P, self.scaler).item()
        logging.info(f"Epoch {epoch} Step {step} Validation MGEH: {val_geh:.6f} "
                     f"(stage {self.current_stage_idx+1}/{len(self.lr_stages)}, "
                     f"lr={self.lr_stages[self.current_stage_idx]:.0e})")
        return {'val_geh': val_geh}

    def save_outputs(self):
        """Restore best weights, then save per-edge predicted AADT (de-scaled) for train/val/test
        to {run_dir}/pred.csv. All metrics are computed post-hoc from this file
        (eval/posthoc_metrics.py) -- training produces nothing else."""
        import csv
        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)
        out = os.path.join(self.run_dir, "pred.csv")
        n = 0
        with open(out, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["edge_id", "set", "gt", "pred"])
            for tag, ids in (("train", self.train_ids), ("val", self.val_ids), ("test", self.test_ids)):
                r = self._predict(ids)
                if r is None:
                    continue
                kept, G, P = r
                gt = (self.scaler.inverse_transform(G) if self.scaler else G).cpu().numpy()
                pr = (self.scaler.inverse_transform(P) if self.scaler else P).cpu().numpy()
                for e, g, p in zip(kept, gt, pr):
                    w.writerow([e, tag, float(g), float(p)]); n += 1
        print(f"[saved] {n} per-edge predictions -> {out}")
        logging.info(f"Saved {n} per-edge predictions -> {out}")

    def save_model(self, epoch):
        formatted_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        path = os.path.join(self.run_dir, f"epoch{epoch}_{formatted_time}.pt")
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")


if __name__ == '__main__':
    print("Initiating model...")
    trainer = DeepDemandTrainer()

    ckpt = os.environ.get("POSTHOC_CKPT")
    if ckpt:
        # post-hoc: load a saved checkpoint, regenerate + save per-edge predictions, no training
        print(f"[posthoc] loading {ckpt}")
        trainer.model.load_state_dict(torch.load(ckpt, map_location=trainer.device, weights_only=True))
        trainer.best_state_dict = None
        trainer.save_outputs()
    else:
        print("Training started...")
        trainer.train_model()
        trainer.save_outputs()

    for handler in logging.root.handlers[:]:
        handler.close()
        logging.root.removeHandler(handler)