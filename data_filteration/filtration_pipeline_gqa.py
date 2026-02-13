import os
import json
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import cv2
from tqdm import tqdm
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


class WritableSurfacePipeline:
    """
    Image-only pipeline with 2 criteria:
    1) At least one writable object/region exists
       - Writable = NOT in non_writable list (complement)
    2) If text is written on it, it will be visible
       - Heuristics: region area ratio, bbox width/height, and not too thin/elongated

    Also saves a segmentation debug image with:
    - Overlay segmentation colors on the image
    - Legend listing top classes by area + whether they are writable/non-writable
    - Highlighted best writable region bbox
    """

    def __init__(
        self,
        images_folder: str,
        output_folder: str = "writable_results",
        use_gpu: bool = True,
        image_extensions=(".jpg", ".jpeg", ".png", ".bmp", ".webp"),
        save_debug: bool = True,
        debug_alpha: float = 0.55,
        debug_max_legend_items: int = 25,
    ):
        self.images_folder = Path(images_folder)
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)

        self.image_extensions = tuple([e.lower() for e in image_extensions])
        self.save_debug = save_debug
        self.debug_alpha = float(debug_alpha)
        self.debug_max_legend_items = int(debug_max_legend_items)

        # --- Visibility / "good region for text" heuristics (tunable) ---
        # At least one writable region must satisfy these to pass.
        self.min_region_area_ratio = 0.06   # >= 6% of the whole image area
        self.min_region_width = 180         # pixels
        self.min_region_height = 80         # pixels
        self.aspect_ratio_min = 0.3         # width/height
        self.aspect_ratio_max = 6.0

        # Optional: avoid scenes that are overwhelmingly non-writable
        self.max_non_writable_ratio = 0.90  # set high; only blocks extreme cases

        # Device
        self.device = "cuda" if (torch.cuda.is_available() and use_gpu) else "cpu"
        print(f"Device: {self.device}")
        if self.device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")

        # SegFormer
        model_name = "nvidia/segformer-b4-finetuned-ade-512-512"
        self.seg_processor = SegformerImageProcessor.from_pretrained(model_name)
        self.seg_model = SegformerForSemanticSegmentation.from_pretrained(model_name).eval().to(self.device)
        self.id2label = self.seg_model.config.id2label  # ADE20K labels

        # Non-writable classes (your list)
        # Writable = everything else (complement)
        self.non_writable_classes = {
            1: "building",          # large structure; usually not a target surface
            2: "sky",
            4: "tree",
            9: "grass",
            12: "person",
            13: "earth",
            16: "mountain",
            17: "plant",
            18: "curtain",
            21: "water",
            26: "sea",
            29: "field",
            34: "rock",
            36: "lamp",             # light source
            39: "cushion",
            46: "sand",
            57: "pillow",
            60: "river",
            63: "blind",            # slats, non-uniform
            66: "flower",
            68: "hill",
            72: "palm",
            81: "towel",
            82: "light",            # light source
            87: "streetlight",      # light source
            92: "apparel",
            94: "land",
            98: "bottle",           # small curved
            104: "fountain",
            108: "plaything",       # small/irregular
            109: "swimming pool",
            113: "waterfall",
            114: "tent",            # fabric
            115: "bag",             # soft/curved
            119: "ball",            # curved
            120: "food",
            125: "pot",             # curved
            126: "animal",
            128: "lake",
            131: "blanket",
            132: "sculpture",       # irregular
            135: "vase",            # curved/small
            139: "fan",             # moving blades
            145: "shower",          # glass/wet area
            147: "glass",           # reflective/transparent
            149: "flag",             # fabric/moving
            3: "floor",                 # if you allow text on ground planes (chalk/paint). Otherwise move to non_writable.
            5: "ceiling",
            6: "road",                  # if you allow markings/paint. Otherwise move to non_writable.
            7: "bed ",                  # blanket/sheets not here; but bed can be messy—optional
            8: "windowpane",            # optional: can write on glass; if you don't want it, move to non_writable
            11: "sidewalk",             # optional
            22: "painting",             # text overlay might look odd; optional
            23: "sofa",                 # (moved to non_writable above)
            27: "mirror",               # reflective; many people mark mirrors—optional
            25: "house",
            28: "rug",                  # fabric; often non-writable; optional
            30: "armchair",             # fabric; optional
            31: "seat",
            43: "signboard",            # often already has text; but is writable surface
            48: "skyscraper",
            52: "path",                 # optional
            53: "stairs",               # optional
            54: "runway",               # markings; optional
            86: "awning",               # fabric-ish; optional
            97: "ottoman",              # often fabric; optional
            136: "traffic light",       # typically not writable; you may move to non_writable
            84: "tower",
            85: "chandelier",
        }

        self.writable_classes = {
            0: "wall",
            10: "cabinet",
            14: "door",
            15: "table",
            19: "chair",
            20: "car",
            24: "shelf",
            32: "fence",
            33: "desk",
            35: "wardrobe",
            37: "bathtub",
            38: "railing",
            40: "base",
            41: "box",
            42: "column",
            44: "chest of drawers",
            45: "counter",
            47: "sink",
            49: "fireplace",
            50: "refrigerator",
            51: "grandstand",
            55: "case",
            56: "pool table",
            58: "screen door",
            59: "stairway",
            61: "bridge",
            62: "bookcase",
            64: "coffee table",
            65: "toilet",
            67: "book",                 # cover is writable-ish
            69: "bench",
            70: "countertop",
            71: "stove",
            73: "kitchen island",
            74: "computer",
            75: "swivel chair",
            76: "boat",
            77: "bar",
            78: "arcade machine",
            79: "hovel",
            80: "bus",
            83: "truck",
            88: "booth",
            89: "television receiver",
            90: "airplane",
            91: "dirt track",           # optional
            93: "pole",
            95: "bannister",
            96: "escalator",
            99: "buffet",
            100: "poster",              # very writable
            101: "stage",
            102: "van",
            103: "ship",
            105: "conveyer belt",
            106: "canopy",              # fabric-ish; optional
            107: "washer",
            110: "stool",
            111: "barrel",
            112: "basket",              # woven; optional
            116: "minibike",
            117: "cradle",
            118: "oven",
            121: "step",
            122: "tank",
            123: "trade name",          # already text-like; treat separately in OCR pipeline
            124: "microwave",
            127: "bicycle",
            129: "dishwasher",
            130: "screen",              # great for visible text
            133: "hood",
            134: "sconce",
            137: "tray",
            138: "ashcan",
            140: "pier",
            141: "crt screen",
            142: "plate",               # small/curved; optional
            143: "monitor",
            144: "bulletin board",      # great writable
            146: "radiator",
            148: "clock"                # face is writable-ish; optional
            }




        self.non_writable_class_ids = list(self.non_writable_classes.keys())

        # Output subfolders
        self.debug_dir = self.output_folder / "seg_debug"
        self.debug_dir.mkdir(parents=True, exist_ok=True)

        print("Pipeline ready.")

    # ---------------------------
    # Utilities
    # ---------------------------

    def _get_color_map(self, num_classes=150, seed=123):
        rng = np.random.RandomState(seed)
        colors = rng.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)
        colors[0] = np.array([0, 0, 0], dtype=np.uint8)
        return colors

    def _is_class_writable(self, class_id: int) -> bool:
        return int(class_id) not in self.non_writable_class_ids

    def _find_regions(self, mask_bool: np.ndarray):
        """Connected components regions on a boolean mask."""
        mask_uint8 = (mask_bool.astype(np.uint8) * 255)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)

        total_pixels = mask_bool.size
        regions = []
        for i in range(1, num_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            w = int(stats[i, cv2.CC_STAT_WIDTH])
            h = int(stats[i, cv2.CC_STAT_HEIGHT])
            aspect = (w / h) if h > 0 else 0.0
            area_ratio = area / total_pixels

            regions.append({
                "region_id": i,
                "area": area,
                "area_ratio": float(area_ratio),
                "bbox": [x, y, w, h],
                "width": w,
                "height": h,
                "aspect_ratio": float(aspect),
            })

        regions.sort(key=lambda r: r["area"], reverse=True)
        return regions

    def _is_visible_text_region(self, region: dict) -> bool:
        """Heuristic: region must be big enough + not too skinny."""
        if region["area_ratio"] < self.min_region_area_ratio:
            return False
        if region["width"] < self.min_region_width or region["height"] < self.min_region_height:
            return False
        if not (self.aspect_ratio_min <= region["aspect_ratio"] <= self.aspect_ratio_max):
            return False
        return True

    def _save_debug_image(self, image_pil: Image.Image, segmentation: np.ndarray, best_region: dict, out_path: Path):
        """Save [overlay | legend] PNG; draws bbox for best_region if provided."""
        img_rgb = np.array(image_pil.convert("RGB"))
        h, w = img_rgb.shape[:2]

        num_classes = int(max(150, int(segmentation.max()) + 1))
        cmap = self._get_color_map(num_classes=num_classes)

        seg_color = cmap[segmentation]  # RGB
        overlay_rgb = (img_rgb * (1 - self.debug_alpha) + seg_color * self.debug_alpha).astype(np.uint8)

        # Draw best region bbox on overlay (in RGB then convert to BGR later)
        if best_region is not None:
            x, y, bw, bh = best_region["bbox"]
            cv2.rectangle(overlay_rgb, (x, y), (x + bw, y + bh), (255, 255, 255), 3)  # white bbox
            cv2.putText(
                overlay_rgb,
                f"best writable region: {best_region['area_ratio']:.3f}",
                (x, max(15, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # Legend stats
        total_pixels = segmentation.size
        present_ids, counts = np.unique(segmentation, return_counts=True)
        order = np.argsort(-counts)
        present_ids = present_ids[order]
        counts = counts[order]

        legend_w = int(max(380, 0.40 * w))
        panel = np.full((h, legend_w, 3), 255, dtype=np.uint8)

        y0 = 26
        cv2.putText(panel, "Classes present (by area)", (10, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
        y0 += 18
        cv2.putText(panel, "id | label | writable? | ratio", (10, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1, cv2.LINE_AA)
        y0 += 22

        shown = 0
        y = y0
        for cid, cnt in zip(present_ids, counts):
            if shown >= self.debug_max_legend_items:
                break

            ratio = float(cnt) / float(total_pixels)
            label = self.id2label.get(int(cid), f"class_{int(cid)}")
            writable = self._is_class_writable(int(cid))
            tag = "WRITABLE" if writable else "NON-WRITABLE"
            tag_color = (0, 140, 0) if writable else (0, 0, 200)  # BGR

            # color box
            box_color_bgr = tuple(int(x) for x in cmap[int(cid)][::-1])
            cv2.rectangle(panel, (10, y - 12), (30, y + 8), box_color_bgr, -1)

            text = f"{int(cid):3d} | {label[:28]:28s} | {tag:12s} | {ratio:.3f}"
            cv2.putText(panel, text, (40, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, tag_color, 1, cv2.LINE_AA)

            y += 20
            shown += 1

        overlay_bgr = overlay_rgb[:, :, ::-1]
        combined = np.concatenate([overlay_bgr, panel], axis=1)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), combined)

    # ---------------------------
    # Core processing
    # ---------------------------

    def process_image(self, image_path: Path) -> dict:
        """
        Returns:
          - passes: bool
          - writable_ratio / non_writable_ratio
          - best_writable_region (largest region that meets visibility heuristics)
          - writable_classes_present / non_writable_classes_present distributions
          - debug_image_path (if enabled)
        """
        image = Image.open(image_path).convert("RGB")

        inputs = self.seg_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.seg_model(**inputs)
            logits = outputs.logits

        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=image.size[::-1],
            mode="bilinear",
            align_corners=False,
        )
        segmentation = upsampled_logits.argmax(dim=1)[0].cpu().numpy()

        total_pixels = segmentation.size

        non_writable_mask = np.isin(segmentation, self.non_writable_class_ids)
        non_writable_pixels = int(non_writable_mask.sum())
        non_writable_ratio = non_writable_pixels / total_pixels

        writable_mask = ~non_writable_mask
        writable_pixels = int(writable_mask.sum())
        writable_ratio = writable_pixels / total_pixels

        # Criterion 1: at least one writable pixel/region
        has_writable_pixels = writable_pixels > 0

        # Find regions on writable mask and choose best region that meets visibility heuristics
        regions = self._find_regions(writable_mask)
        good_regions = [r for r in regions if self._is_visible_text_region(r)]
        best_region = good_regions[0] if good_regions else None

        # Criterion 2: "text will be visible" => at least one good region
        has_visible_region = best_region is not None

        # Optional guard against "almost everything non-writable"
        not_extremely_non_writable = non_writable_ratio <= self.max_non_writable_ratio

        passes = bool(has_writable_pixels and has_visible_region and not_extremely_non_writable)

        # Distributions present (for reporting)
        present_ids, counts = np.unique(segmentation, return_counts=True)
        writable_dist = {}
        non_writable_dist = {}
        for cid, cnt in zip(present_ids, counts):
            ratio = float(cnt) / float(total_pixels)
            label = self.id2label.get(int(cid), f"class_{int(cid)}")
            if self._is_class_writable(int(cid)):
                writable_dist[label] = round(ratio, 3)
            else:
                non_writable_dist[label] = round(ratio, 3)

        writable_dist = dict(sorted(writable_dist.items(), key=lambda x: x[1], reverse=True))
        non_writable_dist = dict(sorted(non_writable_dist.items(), key=lambda x: x[1], reverse=True))

        debug_path = None
        if self.save_debug:
            debug_path = self.debug_dir / f"{image_path.stem}_seg.png"
            self._save_debug_image(image, segmentation, best_region, debug_path)

        return {
            "image_name": image_path.name,
            "image_path": str(image_path),
            "passes": passes,
            "criteria": {
                "has_writable_pixels": has_writable_pixels,
                "has_visible_writable_region": has_visible_region,
                "not_extremely_non_writable": not_extremely_non_writable,
            },
            "ratios": {
                "writable_ratio": round(float(writable_ratio), 3),
                "non_writable_ratio": round(float(non_writable_ratio), 3),
            },
            "best_writable_region": best_region,  # includes area_ratio + bbox
            "num_writable_regions": len(regions),
            "num_good_writable_regions": len(good_regions),
            "classes_present": {
                "writable": writable_dist,
                "non_writable": non_writable_dist,
            },
            "debug_image_path": str(debug_path) if debug_path else None,
        }

    def run(self):
        """Process all images in images_folder."""
        image_paths = []
        for p in self.images_folder.rglob("*"):
            if p.is_file() and p.suffix.lower() in self.image_extensions:
                image_paths.append(p)
        image_paths.sort()

        print(f"Found {len(image_paths)} images in {self.images_folder}")

        all_results = []
        passed = []
        failed = []

        for p in tqdm(image_paths, desc="Processing"):
            res = self.process_image(p)
            all_results.append(res)
            (passed if res["passes"] else failed).append(res)

        summary = {
            "total": len(all_results),
            "passed": len(passed),
            "failed": len(failed),
            "pass_rate": round(100.0 * len(passed) / max(1, len(all_results)), 2),
            "thresholds": {
                "min_region_area_ratio": self.min_region_area_ratio,
                "min_region_width": self.min_region_width,
                "min_region_height": self.min_region_height,
                "aspect_ratio_min": self.aspect_ratio_min,
                "aspect_ratio_max": self.aspect_ratio_max,
                "max_non_writable_ratio": self.max_non_writable_ratio,
            },
            "debug_outputs": {
                "enabled": self.save_debug,
                "folder": str(self.debug_dir.resolve()),
            },
        }

        self._save_json(all_results, "all_results.json")
        self._save_json(passed, "passed.json")
        self._save_json(failed, "failed.json")
        self._save_json(summary, "summary.json")

        print("\n=== SUMMARY ===")
        print(json.dumps(summary, indent=2))
        return {"all_results": all_results, "passed": passed, "failed": failed, "summary": summary}

    def _save_json(self, data, filename):
        out_path = self.output_folder / filename
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2, cls=NumpyEncoder)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    pipeline = WritableSurfacePipeline(
        images_folder="./filtered_images_gqa/",
        output_folder="writable_results",
        use_gpu=True,
        save_debug=True,            # saves overlay + legend + best region bbox
        debug_alpha=0.55,
        debug_max_legend_items=25,
    )
    pipeline.run()
