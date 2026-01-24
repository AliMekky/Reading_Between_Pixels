import base64
import os
import time
import cv2
import numpy as np
import torch
import json
from datetime import datetime
import random

from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel

from utils.get_rectangle_by_mask import largest_inscribed_rectangle
# from utils.som import SoM
from utils.completion_request import CompletionRequest
from utils.text_diffuser import TextDiffuser



class PlanTextAnalysis(BaseModel):
    text_position_number: int
    short_caption: str


def pil_to_base64(pil_image):
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def format_instance_json(instance):
    # Get the attribute names from the class definition
    attributes = instance.__class__.__annotations__.keys()

    # Retrieve the values from the instance
    values = {attr: getattr(instance, attr) for attr in attributes}

    return values

def load_cache(image_path):
    """Load cached data for an image if it exists."""
    image_name = os.path.basename(image_path)
    image_name_base = os.path.splitext(image_name)[0]
    cache_file = f"./cache/{image_name_base}_cache.json"
    
    if os.path.exists(cache_file):
        with open(cache_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


# def find_text_region(text, left, top, right, bottom, font_path='./fonts/arial.ttf', font_size=20, aspect_ratio_threshold=0.1):
#     # Load the font (you may need to provide the correct font path)
#     font = ImageFont.truetype(font_path, font_size)

#     # Calculate the width and height of the original region
#     w = right - left
#     h = bottom - top

#     # Get the text size (width and height)
#     text_width, text_height = font.getsize(text)

#     # Calculate text aspect ratio
#     text_aspect_ratio = text_height / text_width

#     # Calculate the region aspect ratio
#     region_aspect_ratio = h / w

#     # Compare the two aspect ratios
#     aspect_ratio_difference = abs(region_aspect_ratio - text_aspect_ratio)

#     if aspect_ratio_difference > aspect_ratio_threshold:
#         # If the aspect ratios differ too much, adjust the region
#         if text_aspect_ratio > region_aspect_ratio:
#             # Text is taller relative to the region aspect ratio, adjust height
#             scaled_height = h
#             scaled_width = scaled_height / text_aspect_ratio
#         else:
#             # Text is wider relative to the region aspect ratio, adjust width
#             scaled_width = w
#             scaled_height = scaled_width * text_aspect_ratio

#         # Center the found region within the original [left, top, right, bottom]
#         find_left = left + (w - scaled_width) / 2
#         find_top = top + (h - scaled_height) / 2
#         find_right = find_left + scaled_width
#         find_bottom = find_top + scaled_height

#         return int(find_left), int(find_top), int(find_right), int(find_bottom)

#     # If aspect ratio is close enough, return the original region
#     return int(left), int(top), int(right), int(bottom)
def find_text_region(text, left, top, right, bottom,
                     font_path='./fonts/arial.ttf', font_size=20,
                     aspect_ratio_threshold=0.1, debug=False):
    font = ImageFont.truetype(font_path, font_size)

    w = right - left
    h = bottom - top

    # Use getbbox (more reliable than getsize in newer Pillow)
    try:
        x0, y0, x1, y1 = font.getbbox(text)
        text_width, text_height = (x1 - x0), (y1 - y0)
    except Exception:
        text_width, text_height = font.getsize(text)

    if debug:
        print("\n[find_text_region]")
        print("  IN box:", (left, top, right, bottom), "w,h:", (w, h))
        print("  text:", repr(text), "text_w,h:", (text_width, text_height))

    # Guard against zero/negative sizes
    if w <= 0 or h <= 0 or text_width <= 0 or text_height <= 0:
        if debug:
            print("  ⚠️ invalid dims -> returning input")
        return int(left), int(top), int(right), int(bottom)

    text_ar = text_height / text_width     # (h/w)
    region_ar = h / w
    diff = abs(region_ar - text_ar)

    if debug:
        print("  region_ar:", region_ar, "text_ar:", text_ar, "diff:", diff)

    if diff > aspect_ratio_threshold:
        if text_ar > region_ar:
            scaled_height = h
            scaled_width = scaled_height / text_ar
        else:
            scaled_width = w
            scaled_height = scaled_width * text_ar

        find_left = left + (w - scaled_width) / 2
        find_top = top + (h - scaled_height) / 2
        find_right = find_left + scaled_width
        find_bottom = find_top + scaled_height

        if debug:
            print("  OUT box:", (find_left, find_top, find_right, find_bottom))
            # Invariants: OUT should be within IN (up to float rounding)
            print("  within-in?",
                  find_left >= left - 1e-6,
                  find_top >= top - 1e-6,
                  find_right <= right + 1e-6,
                  find_bottom <= bottom + 1e-6)

        return int(find_left), int(find_top), int(find_right), int(find_bottom)

    if debug:
        print("  OUT box (unchanged):", (left, top, right, bottom))

    return int(left), int(top), int(right), int(bottom)





class TypoAttackPlanner:
    def __init__(self, som_image_folder=None, temperature=0.2, max_tokens=4095, top_p=0.1):
        """
        Initialize the TypoAttackPlanner class.
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p

        self.som_image_folder = som_image_folder

        self.diffuser = TextDiffuser()
        self.irrelevant_rng = np.random.default_rng()


        # system instruction
        with open('prompt/position_caption_prompt.txt',
                  'r') as file:
            self.instruction = file.read()

        with open("./utils/irrelevant_words_short.json", 'r', encoding='utf-8') as f:
            self.random_words = json.load(f)
        

    def generate_variants(self, image_path, question, correct_answer, options, model="gpt-4o-2024-08-06", FIXED_POSITION_CACHE=None):
            """
            Generate all three text variants using the SAME text_position_number.
            Returns supporting, misleading, and irrelevant plans with identical positions.
            """
            # Load the image
            image = Image.open(image_path).convert("RGB")

            # Load som image and mask
            image_name = image_path.split("/")[-1]

            seg_image = Image.open(os.path.join(self.som_image_folder, image_name)).convert("RGB")
            mask = np.load(os.path.join(self.som_image_folder, image_name.replace(".jpg", ".npy")), allow_pickle=True)

            # get typo attack plan from chatgpt
            base64_image = pil_to_base64(image)
            base64_image_som = pil_to_base64(seg_image)

            # ============================================
            # STEP 1: Get the best region and caption from the model
            # ============================================

            completion_request = CompletionRequest(
                model=model, 
                top_p=self.top_p,
                response_format=PlanTextAnalysis
            )
            completion_request.set_system_instruction(self.instruction)

            # Format options for the prompt
            user_text = f"Image 0 is the original image, Image 1 is the corresponding segmentation map. Observe the image and the corresponding segmentation map carefully. Please provide the segmentation number and the caption"

            completion_request.add_user_message(text=user_text, base64_image=[base64_image, base64_image_som], image_first=True)

            # Adjust misleading plan
            max_retries = 3

            for attempt in range(1, max_retries + 1):
                try:
                    completion = completion_request.get_completion_payload()

                    # Safely extract parsed result
                    if (completion and getattr(completion, "choices", None)
                            and len(completion.choices) > 0):
                        plan = completion.choices[0].message
                        parsed = getattr(plan, "parsed", None)
                        if parsed and getattr(parsed, "text_position_number", None) and getattr(parsed, "short_caption", None):
                            print("✅ Text position and caption extracted successfully.")
                            break  # success → exit retry loop
                        else:
                            print("⚠️  No data found in response.")
                    else:
                        print("⚠️  Invalid or empty completion object.")

                except Exception as e:
                    print(f"❌ Exception on attempt {attempt}: {e}")

                # Optional small delay before retry
                import time
                time.sleep(1)

            # Fallback after retries
            if plan is None:
                print("❗ Failed to get adjusted plan after 3 tries, using original plan.")

            # THE POSITION IS NOW FIXED - all other variants must use this position
            if FIXED_POSITION_CACHE is not None:
                FIXED_POSITION = FIXED_POSITION_CACHE
                print(f"\n*** USING CACHED FIXED POSITION FOR ALL VARIANTS: {FIXED_POSITION} ***\n")
            else:
                FIXED_POSITION = parsed.text_position_number
                CAPTION = parsed.short_caption
            print("✅ Final Fixed Position Number:", FIXED_POSITION)
            print("✅ Caption for Text Diffusion:", CAPTION)


            # ============================================
            # STEP 2: Calculate Text Region (SHARED - based on fixed position)
            # ============================================

            if int(FIXED_POSITION) <= len(mask):
                target_mask = mask[int(FIXED_POSITION) - 1]['segmentation']
            else:
                target_mask = mask[0]['segmentation']

            label = True
            x, y, w, h = largest_inscribed_rectangle(target_mask, label)

            # Change coordinate system
            mask_width, mask_height = target_mask.T.shape
            left, top, right, bottom = (
                x / mask_width * image.width, 
                y / mask_height * image.height, 
                (x + w) / mask_width * image.width, 
                (y + h) / mask_height * image.height
            )


            # Store the base coordinates
            base_left, base_top, base_right, base_bottom = left, top, right, bottom
            base_bbox_xyxy = [int(base_left), int(base_top), int(base_right), int(base_bottom)]
            
            # ============================================
            # STEP 3: Get the text to be overlaid for each variant
            # ============================================
            # misleading answer
            options_list = [options['A'], options['B'], options['C'], options['D']]
            incorrect_options = [opt for opt in options_list if opt != options[correct_answer]]
            MISLEADING_TEXT = random.choice(incorrect_options)

            # Correct Answer (no change)
            CORRECT_TEXT = options[correct_answer]
            

            # irrelevant answer
            self.irrelevant_rng = np.random.default_rng()
            IRRELEVANT_TEXT = self.irrelevant_rng.choice(self.random_words)
            # IRRELEVANT_TEXT = random.choice(self.random_words)


            # ============================================
            # STEP 4: Generate Diffusion Images for Each Variant
            # ============================================


            results = {}
            positive_prompt = (
                "clear, printed signage text, high contrast, wide letter spacing, no merged letters, accurate letters, natural, realistic"
            )

            # ---  MISLEADING TEXT DIFFUSION ---


            # RESIZE BY SCALE - adjust rectangle to fit the text aspect ratio
            ml_left, ml_top, ml_right, ml_bottom = find_text_region(
                MISLEADING_TEXT,  # The text to fit
                base_left, base_top, base_right, base_bottom,  # Base coordinates
                font_path="./fonts/arial.ttf",
                font_size=20, 
                aspect_ratio_threshold=0.1
            )
            misleading_bbox_xyxy = [int(ml_left), int(ml_top), int(ml_right), int(ml_bottom)] ## to store in cache

            # dbg = image.copy()
            # d = ImageDraw.Draw(dbg)
            # d.rectangle(
            #     [int(base_left), int(base_top), int(base_right), int(base_bottom)],
            #     outline="red", width=4
            # )

            # # draw final text bbox
            # d.rectangle(
            #     [int(ml_left), int(ml_top), int(ml_right), int(ml_bottom)],
            #     outline="green", width=4
            # )

            # dbg.save("./debug_global_bbox.png")
            # print("Saved debug_global_bbox.png")

            # Create two-point positions for diffusion
            # point_positions = [(int(left), int(top)), (int(right), int(bottom))]
            # print("IMAGE SIZE:", image.width, image.height)
            # print("RAW BOX:", ml_left, ml_top, ml_right, ml_bottom)

            point_positions = [
                (int(ml_left),  int(ml_top)),     # top-left
                (int(ml_right), int(ml_top)),     # top-right
                (int(ml_right), int(ml_bottom)),  # bottom-right
                (int(ml_left),  int(ml_bottom)),  # bottom-left
            ]


            # Run diffusion
            diffusion_result = self.diffuser.generate(
                point_positions, 
                image_path, 
                MISLEADING_TEXT,
                CAPTION, 
                # radio="Two Points",
                radio="Four Points",
                # positive_prompt = positive_prompt,
                scale_factor=3, 
                regional_diffusion=True
            )

            # Resize diffusion images to match original image size
            misleading_diffusion_images = diffusion_result[0]
            misleading_diffusion_images = [img.resize((image.width, image.height)) for img in misleading_diffusion_images]

            results['misleading'] = {
                'text': MISLEADING_TEXT,
                'diffusion_images': misleading_diffusion_images,
                'coordinates': (ml_left, ml_top, ml_right, ml_bottom)
            }

            # --- IRRELEVANT TEXT DIFFUSION ---

            # RESIZE BY SCALE - adjust rectangle to fit the irrelevant text aspect ratio
            ir_left, ir_top, ir_right, ir_bottom = find_text_region(
                IRRELEVANT_TEXT,  # Different text content
                base_left, base_top, base_right, base_bottom,  # SAME base coordinates
                font_path="./fonts/arial.ttf",
                font_size=20, 
                aspect_ratio_threshold=0.1
            )
            irrelevant_bbox_xyxy = [int(ir_left), int(ir_top), int(ir_right), int(ir_bottom)]

            point_positions = [
                (int(ir_left), int(ir_top)),           # Top-left
                (int(ir_right), int(ir_top)),          # Top-right
                (int(ir_right), int(ir_bottom)),       # Bottom-right
                (int(ir_left), int(ir_bottom))         # Bottom-left
            ]

            diffusion_result = self.diffuser.generate(
                point_positions,
                image_path,
                IRRELEVANT_TEXT,
                CAPTION,
                # radio="Two Points",
                radio="Four Points",
                # positive_prompt = positive_prompt,
                scale_factor=3,             # 2–3 is more stable than 4
                regional_diffusion=True,
            )

            irrelevant_diffusion_images = diffusion_result[0]
            irrelevant_diffusion_images = [img.resize((image.width, image.height)) for img in irrelevant_diffusion_images]

            results['irrelevant'] = {
                'text': IRRELEVANT_TEXT,
                'diffusion_images': irrelevant_diffusion_images,
                'coordinates': (ir_left, ir_top, ir_right, ir_bottom)
            }


            # --- CORRECT TEXT DIFFUSION ---
            # RESIZE BY SCALE - adjust rectangle to fit the text aspect ratio
            cr_left, cr_top, cr_right, cr_bottom = find_text_region(
                CORRECT_TEXT,  # The text to fit
                base_left, base_top, base_right, base_bottom,  # Base coordinates
                font_path="./fonts/arial.ttf",
                font_size=20, 
                aspect_ratio_threshold=0.1
            )
            # Create two-point positions for diffusion
            # point_positions = [(int(left), int(top)), (int(right), int(bottom))]
            point_positions = [
                (int(cr_left),  int(cr_top)),     # top-left
                (int(cr_right), int(cr_top)),     # top-right
                (int(cr_right), int(cr_bottom)),  # bottom-right
                (int(cr_left),  int(cr_bottom)),  # bottom-left
            ]
            correct_bbox_xyxy = [int(cr_left), int(cr_top), int(cr_right), int(cr_bottom)]


            # Run diffusion
            diffusion_result = self.diffuser.generate(
                point_positions, 
                image_path, 
                CORRECT_TEXT,
                CAPTION, 
                # radio="Two Points",
                radio="Four Points",
                # positive_prompt = positive_prompt,
                scale_factor=3, 
                regional_diffusion=True
            )

            # Resize diffusion images to match original image size
            correct_diffusion_images = diffusion_result[0]
            correct_diffusion_images = [img.resize((image.width, image.height)) for img in correct_diffusion_images]

            results['correct'] = {
                "text": CORRECT_TEXT,
                'diffusion_images': correct_diffusion_images,
                'coordinates': (cr_left, cr_top, cr_right, cr_bottom)
            }

            results['fixed_position'] = FIXED_POSITION
            results['seg_image'] = seg_image
            results['mask'] = mask

            # ============================================
            # STEP 6: Create/Update Cache File
            # ============================================
            
            # Define cache directory and file path
            cache_dir = "./cache"
            os.makedirs(cache_dir, exist_ok=True)
            
            # Create a unique cache filename based on image name
            image_name_base = os.path.splitext(image_name)[0]
            cache_file = os.path.join(cache_dir, f"{image_name_base}_cache.json")
            
            # Prepare cache data
            cache_data = {
                "image_name": image_name,
                "fixed_position": FIXED_POSITION,
                "caption": CAPTION,
                "base_bbox_xyxy": base_bbox_xyxy,   # <-- PRE (shared anchor)
                "variants": {
                    "misleading": {
                        "text": MISLEADING_TEXT,
                        "text_bbox_xyxy": misleading_bbox_xyxy,   # <-- POST (exact intended text box)
                    },
                    "irrelevant": {
                        "text": IRRELEVANT_TEXT,
                        "text_bbox_xyxy": irrelevant_bbox_xyxy,
                    },
                    "correct": {
                        "text": CORRECT_TEXT,
                        "text_bbox_xyxy": correct_bbox_xyxy,
                    }
                },
                "question": question,
                "correct_answer": CORRECT_TEXT,
                "timestamp": datetime.now().isoformat()
            }

            
            # Write cache file
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            
            print(f"\n✅ Cache file created: {cache_file}")

            return results

