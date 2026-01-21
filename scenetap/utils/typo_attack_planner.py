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


class PlanSom(BaseModel):
    image_analysis: str
    correct_answer: str
    incorrect_answer: str
    adversarial_text: str
    text_position_number: int
    text_placement: str
    short_caption_with_adversarial_text: str


class PlanSomAdjust(BaseModel):
    adjust_explanation: str
    adjust_plan: PlanSom


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


def find_text_region(text, left, top, right, bottom, font_path='./fonts/arial.ttf', font_size=20, aspect_ratio_threshold=0.1):
    # Load the font (you may need to provide the correct font path)
    font = ImageFont.truetype(font_path, font_size)

    # Calculate the width and height of the original region
    w = right - left
    h = bottom - top

    # Get the text size (width and height)
    text_width, text_height = font.getsize(text)

    # Calculate text aspect ratio
    text_aspect_ratio = text_height / text_width

    # Calculate the region aspect ratio
    region_aspect_ratio = h / w

    # Compare the two aspect ratios
    aspect_ratio_difference = abs(region_aspect_ratio - text_aspect_ratio)

    if aspect_ratio_difference > aspect_ratio_threshold:
        # If the aspect ratios differ too much, adjust the region
        if text_aspect_ratio > region_aspect_ratio:
            # Text is taller relative to the region aspect ratio, adjust height
            scaled_height = h
            scaled_width = scaled_height / text_aspect_ratio
        else:
            # Text is wider relative to the region aspect ratio, adjust width
            scaled_width = w
            scaled_height = scaled_width * text_aspect_ratio

        # Center the found region within the original [left, top, right, bottom]
        find_left = left + (w - scaled_width) / 2
        find_top = top + (h - scaled_height) / 2
        find_right = find_left + scaled_width
        find_bottom = find_top + scaled_height

        return int(find_left), int(find_top), int(find_right), int(find_bottom)

    # If aspect ratio is close enough, return the original region
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

        # system instruction
        with open('prompt/attack_step_give_answer_combine.txt',
                  'r') as file:
            self.instruction_combine = file.read()

        with open(
                'prompt/attack_adjust_plan.txt',
                'r') as file:
            self.instruction_adjust_plan = file.read()

    def attack(self, image_path, question, correct_answer):
        """
        Applies a 'typo attack' on the input PIL image and returns the modified image.

        Returns:
        The modified image with the applied 'typo attack'.
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
        # gpt-4o-2024-08-06

        completion_request = CompletionRequest(model="gpt-4o-2024-08-06", temperature=self.temperature, max_tokens=self.max_tokens, top_p=self.top_p,
                                               response_format=PlanSom)
        completion_request.set_system_instruction(self.instruction_combine)
        user_text = f"Image 0 is the original image, image 1 is the corresponding segmentation map. Observe the image and the corresponding segmentation map carefully. Question to attack: {question}. Correct answer: {correct_answer}. Please provide a detailed, step-by-step plan for achieving this goal."
        completion_request.add_user_message(text=user_text, base64_image=[base64_image, base64_image_som],
                                            image_first=True)
        completion = completion_request.get_completion_payload()
        plan_detail = completion.choices[0].message.parsed

        print("plan_detail:")
        print("image_analysis:", plan_detail.image_analysis)
        print("correct_answer:", plan_detail.correct_answer)
        print("incorrect_answer:", plan_detail.incorrect_answer)
        print("adversarial_text:", plan_detail.adversarial_text)
        print("text_placement:", plan_detail.text_placement)
        print("text_position_number:", plan_detail.text_position_number)
        print("short_caption_with_adversarial_text:", plan_detail.short_caption_with_adversarial_text)

        # add assistant message
        completion_request.add_assistant_message(text=f"{plan_detail}")
        plan_detail_origin = plan_detail.copy()

        # adjust plan to avoid region is the question target region
        user_text = self.instruction_adjust_plan

        completion_request.set_response_format(PlanSomAdjust)
        completion_request.add_user_message(text=user_text)
        completion = completion_request.get_completion_payload()
        plan_adjust = completion.choices[0].message.parsed

        plan_detail = plan_adjust.adjust_plan
        explanation = plan_adjust.adjust_explanation
        print("explanation:", explanation)
        print("plan_detail:")
        print("image_analysis:", plan_detail.image_analysis)
        print("correct_answer:", plan_detail.correct_answer)
        print("incorrect_answer:", plan_detail.incorrect_answer)
        print("adversarial_text:", plan_detail.adversarial_text)
        print("text_placement:", plan_detail.text_placement)
        print("text_position_number:", plan_detail.text_position_number)
        print("short_caption_with_adversarial_text:", plan_detail.short_caption_with_adversarial_text)

        # get the rectangle to place the text
        # if plan_detail.text_position_number is number and the number is in the mask
        if int(plan_detail.text_position_number) <= len(mask):
            target_mask = mask[int(plan_detail.text_position_number) - 1]['segmentation']
        else:
            print("text_position_number is out of range, use the largest mask")
            target_mask = mask[0]['segmentation']
        # target_mask = mask[0]['segmentation']
        label = True
        # target_mask = target_mask.T
        x, y, w, h = largest_inscribed_rectangle(target_mask, label)
        print("rectangle [x, y, w, h]:", [x, y, w, h])
        # change coordinate (0,0) from right-bottom to left-top, left to right is 0-1, top to bottom is 0-1
        mask_width, mask_height = target_mask.T.shape

        left, top, right, bottom = x / mask_width * image.width, y / mask_height * image.height, (
                x + w) / mask_width * image.width, (y + h) / mask_height * image.height

        print("rectangle [(left, top), (right, bottom)]:", [(int(left), int(top)), (int(right), int(bottom))])

        # resize by scale
        left, top, right, bottom = find_text_region(plan_detail.adversarial_text, left, top, right, bottom,
                                                    font_path="./fonts/arial.ttf",
                                                    font_size=20, aspect_ratio_threshold=0.1)
        print("Resized rectangle [(left, top), (right, bottom)]:", [(int(left), int(top)), (int(right), int(bottom))])


        # diffusion
        two_point_positions = [(int(left), int(top)), (int(right), int(bottom))]

        diffusion_result = self.diffuser.generate(two_point_positions, image_path, plan_detail.adversarial_text,
                                                  plan_detail.short_caption_with_adversarial_text, radio="Two Points",
                                                  scale_factor=2, regional_diffusion=True)


        diffusion_images = diffusion_result[0]
        diffusion_images = [diffusion_image.resize((image.width, image.height)) for diffusion_image in diffusion_images]

        return diffusion_images, seg_image, plan_detail_origin, plan_detail
    


    base_bbox_xyxy = [int(base_left), int(base_top), int(base_right), int(base_bottom)]

    def attack_all_three_variants(self, image_path, question, correct_answer, options, model="gpt-4o-2024-08-06", FIXED_POSITION_CACHE=None):
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
            # gpt-4o-2024-08-06

            # ============================================
            # STEP 1: Generate Misleading Text FIRST (this determines the position)
            # ============================================

            completion_request = CompletionRequest(
                model=model, 
                temperature=self.temperature, 
                # max_tokens=self.max_tokens, 
                top_p=self.top_p,
                response_format=PlanMisleading
            )
            completion_request.set_system_instruction(self.instruction_misleading)

            # Format options for the prompt
            options_text = f"Options: A: {options['A']}, B: {options['B']}, C: {options['C']}, D: {options['D']}"
            user_text = f"Image 0 is the original image, Image 1 is the corresponding segmentation map. Observe the image and the corresponding segmentation map carefully. Question to attack: {question}. {options_text}. Correct answer: {correct_answer}. Please provide a detailed, step-by-step plan for achieving this goal."

            completion_request.add_user_message(text=user_text, base64_image=[base64_image, base64_image_som], image_first=True)
            completion = completion_request.get_completion_payload()
            misleading_plan_origin = completion.choices[0].message.parsed

            # print("Misleading Plan (Original):")
            # print(f"  - misleading_text: {misleading_plan_origin.misleading_text}")
            # print(f"  - incorrect_answer: {misleading_plan_origin.incorrect_answer}")
            # print(f"  - text_position_number: {misleading_plan_origin.text_position_number}")

            # Adjust misleading plan
            max_retries = 3
            misleading_plan_adjusted = None

            for attempt in range(1, max_retries + 1):
                print(f"Attempt {attempt}/{max_retries} for plan adjustment...")

                completion_request.add_assistant_message(text=f"{misleading_plan_origin}")
                completion_request.set_response_format(PlanMisleadingAdjust)
                completion_request.add_user_message(text=self.instruction_adjust_plan)

                try:
                    completion = completion_request.get_completion_payload()

                    # Safely extract parsed result
                    if (completion and getattr(completion, "choices", None)
                            and len(completion.choices) > 0):
                        message = completion.choices[0].message
                        parsed = getattr(message, "parsed", None)
                        if parsed and getattr(parsed, "adjust_plan", None):
                            misleading_plan_adjusted = parsed.adjust_plan
                            print("✅ Adjusted plan received successfully.")
                            break  # success → exit retry loop
                        else:
                            print("⚠️  No parsed.adjust_plan found in response.")
                    else:
                        print("⚠️  Invalid or empty completion object.")

                except Exception as e:
                    print(f"❌ Exception on attempt {attempt}: {e}")

                # Optional small delay before retry
                import time
                time.sleep(1)

            # Fallback after retries
            if misleading_plan_adjusted is None:
                print("❗ Failed to get adjusted plan after 3 tries, using original plan.")
                misleading_plan_adjusted = misleading_plan_origin

            # THE POSITION IS NOW FIXED - all other variants must use this position
            if FIXED_POSITION_CACHE is not None:
                FIXED_POSITION = FIXED_POSITION_CACHE
                print(f"\n*** USING CACHED FIXED POSITION FOR ALL VARIANTS: {FIXED_POSITION} ***\n")
            else:
                FIXED_POSITION = misleading_plan_adjusted.text_position_number

            # print(f"\n*** FIXED POSITION FOR ALL VARIANTS: {FIXED_POSITION} ***\n")

            # ============================================
            # STEP 2: Generate Supporting Text (NO adjustment - just use fixed position)
            # ============================================

            completion_request = CompletionRequest(
                model=model, 
                temperature=self.temperature, 
                # max_tokens=self.max_tokens, 
                top_p=self.top_p,
                response_format=PlanSupporting
            )
            completion_request.set_system_instruction(self.instruction_supporting)
            user_text = f"Image 0 is the original image, Image 1 is the corresponding segmentation map. Observe the image and the corresponding segmentation map carefully. Question: {question}. Correct answer: {correct_answer}. IMPORTANT: You must use text_position_number = {FIXED_POSITION}. Please provide your response."

            completion_request.add_user_message(text=user_text, base64_image=[base64_image, base64_image_som], image_first=True)
            completion = completion_request.get_completion_payload()
            supporting_plan = completion.choices[0].message.parsed

            # Force the position to match (no adjustment needed)
            supporting_plan.text_position_number = FIXED_POSITION

            # ============================================
            # STEP 3: Generate Irrelevant Text (NO adjustment - just use fixed position)
            # # ============================================

            completion_request = CompletionRequest(
                model=model, 
                temperature=self.temperature, 
                top_p=self.top_p,
                response_format=PlanIrrelevant
            )
            random_int_between_0_and_49 = random.randint(0, 49)
            completion_request.set_system_instruction(self.instruction_irrelevant)
            user_text = f"Image 0 is the original image, Image 1 is the corresponding segmentation map. Observe the image and the corresponding segmentation map carefully. Question: {question}. IMPORTANT: You must use text_position_number = {FIXED_POSITION}. the provided domain is {random_domains[random_int_between_0_and_49]}. Please provide your response."

            completion_request.add_user_message(text=user_text, base64_image=[base64_image, base64_image_som], image_first=True)
            completion = completion_request.get_completion_payload()
            irrelevant_plan = completion.choices[0].message.parsed

            # Force the position to match (no adjustment needed)
            irrelevant_plan.text_position_number = FIXED_POSITION

            # ============================================
            # VERIFICATION: All three should have the same position
            # ============================================
            assert misleading_plan_adjusted.text_position_number == FIXED_POSITION
            assert supporting_plan.text_position_number == FIXED_POSITION
            assert irrelevant_plan.text_position_number == FIXED_POSITION

            # ============================================
            # STEP 4: Calculate Text Region (SHARED - based on fixed position)
            # ============================================

            if int(FIXED_POSITION) <= len(mask):
                target_mask = mask[int(FIXED_POSITION) - 1]['segmentation']
                # print(f"Using segmentation region {FIXED_POSITION}")
            else:
                # print(f"text_position_number {FIXED_POSITION} is out of range, using the largest mask")
                target_mask = mask[0]['segmentation']

            label = True
            x, y, w, h = largest_inscribed_rectangle(target_mask, label)
            # print(f"Largest inscribed rectangle [x, y, w, h]: {[x, y, w, h]}")

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
            # STEP 5: Generate Diffusion Images for Each Variant
            # ============================================

            results = {}


            # ---  MISLEADING TEXT DIFFUSION ---

            # RESIZE BY SCALE - adjust rectangle to fit the text aspect ratio
            ml_left, ml_top, ml_right, ml_bottom = find_text_region(
                misleading_plan_adjusted.misleading_text,  # The text to fit
                base_left, base_top, base_right, base_bottom,  # Base coordinates
                font_path="./fonts/arial.ttf",
                font_size=20, 
                aspect_ratio_threshold=0.1
            )
            misleading_bbox_xyxy = [int(ml_left), int(ml_top), int(ml_right), int(ml_bottom)] ## to store in cache
            # print(f"Resized rectangle for misleading text '{misleading_plan_adjusted.misleading_text}': {[(int(left), int(top)), (int(right), int(bottom))]}")

            # Create two-point positions for diffusion
            # point_positions = [(int(left), int(top)), (int(right), int(bottom))]
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
                misleading_plan_adjusted.misleading_text,
                misleading_plan_adjusted.short_caption_with_misleading_text, 
                # radio="Two Points",
                radio="Four Points",
                positive_prompt = (
                    "clear legible text, exact spelling, sharp letters, "
                    "professional typography, sans-serif font, "
                    "high contrast, perfectly straight baseline, "
                    "natural lighting, realistic text integration"
                ),
                scale_factor=3, 
                regional_diffusion=True
            )

            # Resize diffusion images to match original image size
            misleading_diffusion_images = diffusion_result[0]
            misleading_diffusion_images = [img.resize((image.width, image.height)) for img in misleading_diffusion_images]

            results['misleading'] = {
                'plan_origin': misleading_plan_origin,
                'plan_adjusted': misleading_plan_adjusted,
                'diffusion_images': misleading_diffusion_images,
                'coordinates': (left, top, right, bottom)
            }

            # --- SUPPORTING TEXT DIFFUSION ---
            # # RESIZE BY SCALE - adjust rectangle to fit the supporting text aspect ratio
            # left, top, right, bottom = find_text_region(
            #     supporting_plan.supporting_text,  # Different text content
            #     base_left, base_top, base_right, base_bottom,  # SAME base coordinates
            #     font_path="./fonts/arial.ttf",
            #     font_size=20, 
            #     aspect_ratio_threshold=0.1
            # )
            # # print(f"Resized rectangle for supporting text '{supporting_plan.supporting_text}': {[(int(left), int(top)), (int(right), int(bottom))]}")

            # # point_positions = [(int(left), int(top)), (int(right), int(bottom))]
            # point_positions = [
            #     (int(left), int(top)),           # Top-left
            #     (int(right), int(top)),          # Top-right
            #     (int(right), int(bottom)),       # Bottom-right
            #     (int(left), int(bottom))         # Bottom-left
            # ]
            # diffusion_result = self.diffuser.generate(
            #     point_positions, 
            #     image_path, 
            #     supporting_plan.supporting_text,
            #     supporting_plan.short_caption_with_supporting_text, 
            #     # radio="Two Points",
            #     radio="Four Points",
            #     positive_prompt = (
            #         "clear legible text, exact spelling, sharp letters, "
            #         "professional typography, sans-serif font, "
            #         "high contrast, perfectly straight baseline, "
            #         "natural lighting, realistic text integration"
            #     ),
            #     scale_factor=3, 
            #     regional_diffusion=True
            # )

            # supporting_diffusion_images = diffusion_result[0]
            # supporting_diffusion_images = [img.resize((image.width, image.height)) for img in supporting_diffusion_images]

            # results['supporting'] = {
            #     'plan': supporting_plan,
            #     'diffusion_images': supporting_diffusion_images,
            #     'coordinates': (left, top, right, bottom)
            # }

            # --- IRRELEVANT TEXT DIFFUSION ---

            # RESIZE BY SCALE - adjust rectangle to fit the irrelevant text aspect ratio
            ir_left, ir_top, ir_right, ir_bottom = find_text_region(
                irrelevant_plan.irrelevant_text,  # Different text content
                # "Gourmet cuisine",
                base_left, base_top, base_right, base_bottom,  # SAME base coordinates
                font_path="./fonts/arial.ttf",
                font_size=20, 
                aspect_ratio_threshold=0.1
            )
            irrelevant_bbox_xyxy = [int(ir_left), int(ir_top), int(ir_right), int(ir_bottom)]
            # print(f"Resized rectangle for irrelevant text '{irrelevant_plan.irrelevant_text}': {[(int(left), int(top)), (int(right), int(bottom))]}")

            # point_positions = [(int(left), int(top)), (int(right), int(bottom))]
            point_positions = [
                (int(ir_left), int(ir_top)),           # Top-left
                (int(ir_right), int(ir_top)),          # Top-right
                (int(ir_right), int(ir_bottom)),       # Bottom-right
                (int(ir_left), int(ir_bottom))         # Bottom-left
            ]

            diffusion_result = self.diffuser.generate(
                point_positions,
                image_path,
                irrelevant_plan.irrelevant_text,
                irrelevant_plan.short_caption_with_irrelevant_text,
                # radio="Two Points",
                radio="Four Points",
                positive_prompt = (
                    "clear legible text, exact spelling, sharp letters, "
                    "professional typography, sans-serif font, "
                    "high contrast, perfectly straight baseline, "
                    "natural lighting, realistic text integration"
                ),
                scale_factor=3,              # 2–3 is more stable than 4
                regional_diffusion=True,
            )

            irrelevant_diffusion_images = diffusion_result[0]
            irrelevant_diffusion_images = [img.resize((image.width, image.height)) for img in irrelevant_diffusion_images]

            results['irrelevant'] = {
                'plan': irrelevant_plan,
                'diffusion_images': irrelevant_diffusion_images,
                'coordinates': (left, top, right, bottom)
            }


            # --- CORRECT TEXT DIFFUSION ---
            # RESIZE BY SCALE - adjust rectangle to fit the text aspect ratio
            cr_left, cr_top, cr_right, cr_bottom = find_text_region(
                correct_answer,  # The text to fit
                base_left, base_top, base_right, base_bottom,  # Base coordinates
                font_path="./fonts/arial.ttf",
                font_size=20, 
                aspect_ratio_threshold=0.1
            )
            # print(f"Resized rectangle for misleading text '{misleading_plan_adjusted.misleading_text}': {[(int(left), int(top)), (int(right), int(bottom))]}")

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
                correct_answer,
                f"Image showing the answer: {correct_answer}", 
                # radio="Two Points",
                radio="Four Points",
                positive_prompt = (
                    "clear legible text, exact spelling, sharp letters, "
                    "professional typography, sans-serif font, "
                    "high contrast, perfectly straight baseline, "
                    "natural lighting, realistic text integration"
                ),
                scale_factor=3, 
                regional_diffusion=True
            )

            # Resize diffusion images to match original image size
            correct_diffusion_images = diffusion_result[0]
            correct_diffusion_images = [img.resize((image.width, image.height)) for img in correct_diffusion_images]

            results['correct'] = {
                'diffusion_images': correct_diffusion_images,
                'coordinates': (left, top, right, bottom)
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
                "base_bbox_xyxy": base_bbox_xyxy,   # <-- PRE (shared anchor)
                "variants": {
                    "misleading": {
                        "text": misleading_plan_adjusted.misleading_text,
                        "incorrect_answer": misleading_plan_adjusted.incorrect_answer,
                        "caption": misleading_plan_adjusted.short_caption_with_misleading_text,
                        "text_bbox_xyxy": misleading_bbox_xyxy,   # <-- POST (exact intended text box)
                    },
                    # "supporting": {
                    #     "text": supporting_plan.supporting_text,
                    #     "caption": supporting_plan.short_caption_with_supporting_text,
                    #     "text_bbox_xyxy": supporting_bbox_xyxy,
                    # },
                    "irrelevant": {
                        "text": irrelevant_plan.irrelevant_text,
                        "caption": irrelevant_plan.short_caption_with_irrelevant_text,
                        "text_bbox_xyxy": irrelevant_bbox_xyxy,
                    },
                    "correct": {
                        "text": correct_answer,
                        "text_bbox_xyxy": correct_bbox_xyxy,
                    }
                },
                "question": question,
                "correct_answer": correct_answer,
                "timestamp": datetime.now().isoformat()
            }

            
            # Write cache file
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            
            print(f"\n✅ Cache file created: {cache_file}")

            return results

