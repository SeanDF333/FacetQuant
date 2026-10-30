import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import CrystalDataset, collate, split_paths
from .model import load_model


def _rle(mask, mask_utils):
    return mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))


@torch.inference_mode()
def evaluate(args):
    try:
        from pycocotools import mask as mask_utils
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:
        raise RuntimeError("Install evaluation dependencies: pip install -e '.[eval]'") from exc

    sources = tuple(args.sources.split(","))
    paths = split_paths(Path(args.data).resolve(), sources, "validation")
    dataset = CrystalDataset(paths, args.max_size)
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate, num_workers=args.workers)
    model = load_model(args.weights, args.device, args.max_size)
    gt = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "crystal"}], "info": {}}
    detections = []
    count_errors = []
    area_errors = []
    confident_count = 0
    annotation_id = 1
    for index, (images, targets) in enumerate(loader):
        image, target = images[0], targets[0]
        height, width = image.shape[1:]
        gt["images"].append({"id": index, "width": width, "height": height, "file_name": str(paths[index])})
        true_masks = target["masks"].numpy().astype(bool)
        true_rles = []
        for mask in true_masks:
            rle = _rle(mask, mask_utils)
            true_rles.append(rle)
            gt["annotations"].append({"id": annotation_id, "image_id": index, "category_id": 1, "segmentation": rle, "bbox": mask_utils.toBbox(rle).tolist(), "area": int(mask.sum()), "iscrowd": 0})
            annotation_id += 1
        output = model([image.to(args.device)])[0]
        scores = output["scores"].cpu().numpy()
        masks = output["masks"][:, 0].cpu().numpy() >= args.mask_threshold
        pred_rles = []
        for score, mask in zip(scores, masks):
            if score < 0.05 or not mask.any():
                continue
            rle = _rle(mask, mask_utils)
            pred_rles.append((float(score), rle, int(mask.sum())))
            detections.append({"image_id": index, "category_id": 1, "segmentation": rle, "bbox": mask_utils.toBbox(rle).tolist(), "score": float(score)})
        confident = [(score, rle, area) for score, rle, area in pred_rles if score >= args.score_threshold]
        confident_count += len(confident)
        count_errors.append(abs(len(confident) - len(true_rles)))
        if confident and true_rles:
            ious = mask_utils.iou([item[1] for item in confident], true_rles, [0] * len(true_rles))
            used = set()
            for p, (_, _, area) in enumerate(confident):
                candidates = [(ious[p, j], j) for j in range(len(true_rles)) if j not in used]
                if not candidates:
                    break
                best_iou, j = max(candidates)
                if best_iou >= 0.5:
                    used.add(j)
                    true_area = int(true_masks[j].sum())
                    area_errors.append(abs(area - true_area) / true_area)
        if (index + 1) % 10 == 0 or index + 1 == len(dataset):
            print(f"evaluated {index + 1}/{len(dataset)}", flush=True)

    result = {"images": len(paths), "gt_instances": len(gt["annotations"]), "pred_instances_above_threshold": confident_count, "count_mae": float(np.mean(count_errors)), "matched_area_median_ape": float(np.median(area_errors)) if area_errors else None, "matched_pairs": len(area_errors), "score_threshold": args.score_threshold, "mask_threshold": args.mask_threshold}
    if detections:
        coco_gt = COCO()
        coco_gt.dataset = gt
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(detections)
        for kind in ("segm", "bbox"):
            evaluator = COCOeval(coco_gt, coco_dt, kind)
            evaluator.evaluate()
            evaluator.accumulate()
            evaluator.summarize()
            result[kind] = {name: float(evaluator.stats[i]) for i, name in enumerate(("AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large", "AR_1", "AR_10", "AR_100", "AR_small", "AR_medium", "AR_large"))}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


def add_evaluate_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--data", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--sources", default="vmxi,xchem")
    parser.add_argument("--out", default="runs/validation_metrics.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-size", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--score-threshold", type=float, default=0.6)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
