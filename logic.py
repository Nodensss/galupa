import cv2
import numpy as np
import easyocr
from PIL import Image
import io
import re

# Initialize EasyOCR reader (load once to avoid reloading)
# Adding 'ru' as requested in the prompt explanation, and 'en'
reader = easyocr.Reader(['ru', 'en'], gpu=False)

def process_image(image_bytes):
    """
    Process the image to extract text and identify the correct answer.

    Args:
        image_bytes: Bytes of the uploaded image.

    Returns:
        dict: containing 'question', 'options', 'correct_answer_text', 'full_text', 'original_image', 'processed_image'
    """
    # Load image
    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert('RGB')
    img_np = np.array(image)

    # Convert to BGR for OpenCV
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # OCR
    results = reader.readtext(img_np)
    # results format: [[box, text, confidence], ...]
    # box: [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

    # Sort results by Y coordinate (top to bottom), then X (left to right)
    results.sort(key=lambda x: (x[0][0][1], x[0][0][0]))

    # Merge horizontally close boxes on similar Y
    merged_results = []
    if results:
        current_block = results[0]
        for i in range(1, len(results)):
            next_block = results[i]

            # Check Y overlap
            curr_y1 = current_block[0][0][1]
            curr_y2 = current_block[0][2][1]
            next_y1 = next_block[0][0][1]
            next_y2 = next_block[0][2][1]

            # Check if same line (overlap significantly)
            overlap = min(curr_y2, next_y2) - max(curr_y1, next_y1)
            height = curr_y2 - curr_y1

            # Check X distance
            curr_x2 = current_block[0][1][0]
            next_x1 = next_block[0][0][0]

            x_dist = next_x1 - curr_x2

            if overlap > 0.5 * height and x_dist < 50: # 50px gap max
                # Merge
                # Update box: x1 is min, y1 is min, x2 is max, y2 is max
                new_box = [
                    [min(current_block[0][0][0], next_block[0][0][0]), min(current_block[0][0][1], next_block[0][0][1])],
                    [max(current_block[0][1][0], next_block[0][1][0]), min(current_block[0][1][1], next_block[0][1][1])],
                    [max(current_block[0][2][0], next_block[0][2][0]), max(current_block[0][2][1], next_block[0][2][1])],
                    [min(current_block[0][3][0], next_block[0][3][0]), max(current_block[0][3][1], next_block[0][3][1])]
                ]
                new_text = current_block[1] + " " + next_block[1]
                new_conf = (current_block[2] + next_block[2]) / 2 # average confidence
                current_block = (new_box, new_text, new_conf)
            else:
                merged_results.append(current_block)
                current_block = next_block
        merged_results.append(current_block)
        results = merged_results

    # Detect Green Color
    # Convert to HSV
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Define green range
    # Green in HSV is roughly 60. OpenCV uses H: 0-179, S: 0-255, V: 0-255
    # Standard Green: (35, 50, 50) to (85, 255, 255)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])

    # Mask for green
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Find which text block intersects with green the most
    best_match_idx = -1
    max_intersection = 0

    # Helper to calculate intersection
    def check_green_intersection(box, mask):
        # expand box slightly to catch borders
        # box format from easyocr is list of lists
        x_min_raw = min(p[0] for p in box)
        x_max_raw = max(p[0] for p in box)
        y_min_raw = min(p[1] for p in box)
        y_max_raw = max(p[1] for p in box)
        height = y_max_raw - y_min_raw
        left_pad = max(10, int(height * 2.0))
        y_pad = max(10, int(height * 0.5))
        x_min = int(x_min_raw) - left_pad
        x_max = int(x_max_raw) + 10
        y_min = int(y_min_raw) - y_pad
        y_max = int(y_max_raw) + y_pad

        # Clip to image bounds
        h, w = mask.shape
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(w, x_max)
        y_max = min(h, y_max)

        if x_max <= x_min or y_max <= y_min:
            return 0, 0

        roi = mask[y_min:y_max, x_min:x_max]
        green_pixels = cv2.countNonZero(roi)
        area = roi.shape[0] * roi.shape[1]
        return green_pixels, area

    # Visualize boxes and green detection on a copy
    debug_img = img_np.copy()

    structured_data = []

    best_match_ratio = 0
    for i, (box, text, conf) in enumerate(results):
        green_count, area = check_green_intersection(box, mask)
        green_ratio = (green_count / area) if area > 0 else 0

        # Draw box
        p1 = tuple(map(int, box[0]))
        p3 = tuple(map(int, box[2]))
        cv2.rectangle(debug_img, p1, p3, (255, 0, 0), 2)

        # If green detected, mark it
        if green_count > 0:
             cv2.rectangle(debug_img, p1, p3, (0, 255, 0), 2)

        # We need a decent amount of green to call it the correct answer.
        # Prefer the best green ratio to avoid selecting large boxes with small green areas.
        if green_ratio > best_match_ratio:
            best_match_ratio = green_ratio
            max_intersection = green_count
            best_match_idx = i

        structured_data.append({
            'text': text,
            'box': box,
            'is_correct': False
        })

    # Heuristic for Question vs Options

    question_lines = []
    options_raw = []

    def build_line_metadata(items):
        lines = []
        for item in items:
            box = item['box']
            x_min = min(p[0] for p in box)
            x_max = max(p[0] for p in box)
            y_min = min(p[1] for p in box)
            y_max = max(p[1] for p in box)
            lines.append({
                **item,
                'x_min': x_min,
                'x_max': x_max,
                'y_min': y_min,
                'y_max': y_max,
                'y_center': (y_min + y_max) / 2,
            })
        return lines

    def split_into_blocks(lines, image_height):
        if not lines:
            return []
        lines = sorted(lines, key=lambda l: l['y_center'])
        gaps = [lines[i + 1]['y_center'] - lines[i]['y_center'] for i in range(len(lines) - 1)]
        median_gap = float(np.median(gaps)) if gaps else 0
        split_threshold = max(1.5 * median_gap, 20)
        blocks = []
        current = [lines[0]]
        for i in range(1, len(lines)):
            if lines[i]['y_center'] - lines[i - 1]['y_center'] > split_threshold:
                blocks.append(current)
                current = [lines[i]]
            else:
                current.append(lines[i])
        blocks.append(current)
        return blocks

    def select_options_block(lines, image_height):
        blocks = split_into_blocks(lines, image_height)
        if not blocks:
            return None
        best_block = None
        best_score = None
        for block in blocks:
            count = len(block)
            x_mins = [line['x_min'] for line in block]
            x_min_std = float(np.std(x_mins)) if len(x_mins) > 1 else 0
            y_center = float(np.mean([line['y_center'] for line in block]))
            in_middle = image_height * 0.15 < y_center < image_height * 0.85
            count_bonus = 1 if 3 <= count <= 6 else 0
            score = (count * 2) - (x_min_std / 20) + (2 if in_middle else -1) + count_bonus
            if best_score is None or score > best_score:
                best_score = score
                best_block = block
        return best_block

    def filter_ui_noise(lines):
        ui_pattern = re.compile(
            r"(правильный\s+ответ|неправильный\s+ответ|баллов|назад|далее|завершить)",
            re.IGNORECASE,
        )
        return [line for line in lines if not ui_pattern.search(line['text'])]

    def filter_language_noise(lines):
        filtered = []
        for line in lines:
            text = line['text'].strip()
            if not text:
                continue
            has_cyrillic = bool(re.search(r"[А-Яа-яЁё]", text))
            has_latin = bool(re.search(r"[A-Za-z]", text))
            has_digits = bool(re.search(r"\d", text))
            has_word_chars = bool(re.search(r"[A-Za-zА-Яа-яЁё0-9]", text))
            if not has_word_chars:
                continue
            if has_latin and not has_cyrillic and not has_digits:
                continue
            if has_latin and not has_cyrillic and len(text) > 3:
                continue
            filtered.append(line)
        return filtered

    # Try to identify where options start
    # Common option markers
    # Regex is better: ^[A-D][.)]
    marker_pattern = re.compile(r'^\s*(?:[A-Fa-f]|\d+)[\.\):]\s*')

    line_items = filter_language_noise(filter_ui_noise(build_line_metadata(structured_data)))
    options_block = select_options_block(line_items, img_np.shape[0])

    if options_block:
        options_raw = [
            {key: line[key] for key in ['text', 'box', 'is_correct']}
            for line in options_block
        ]
        options_block_min_y = min(line['y_min'] for line in options_block)
        gaps = sorted(
            [
                abs(line_items[i + 1]['y_center'] - line_items[i]['y_center'])
                for i in range(len(line_items) - 1)
            ]
        )
        median_gap = float(np.median(gaps)) if gaps else 0
        question_candidates = [
            line
            for line in line_items
            if line['y_center'] < options_block_min_y
            and (options_block_min_y - line['y_center']) <= max(3 * median_gap, 80)
        ]
        question_candidates.sort(key=lambda l: l['y_center'])
        question_lines = [
            {key: line[key] for key in ['text', 'box', 'is_correct']}
            for line in question_candidates[-3:]
        ]
        option_lengths = [len(item['text'].strip()) for item in options_raw if item['text'].strip()]
        median_length = float(np.median(option_lengths)) if option_lengths else 0
        question_from_options = None
        for item in options_raw:
            text = item['text'].strip()
            if not text:
                continue
            is_question_like = (
                "?" in text
                or text.endswith(":")
                or len(text) > max(40, int(median_length * 1.6))
            )
            if is_question_like:
                question_from_options = item
                break
        if question_from_options:
            options_raw = [item for item in options_raw if item is not question_from_options]
            if not question_lines:
                question_lines = [question_from_options]
    else:
        option_start_index = -1

        # First pass: look for explicit markers
        for i, item in enumerate(structured_data):
            text_clean = item['text'].strip()
            if marker_pattern.match(text_clean):
                option_start_index = i
                break

        if option_start_index != -1:
            question_lines = structured_data[:option_start_index]
            options_raw = structured_data[option_start_index:]
        else:
            # Fallback: Assume the last 4 items are options if total items > 4
            if len(structured_data) >= 5:
                question_lines = structured_data[:-4]
                options_raw = structured_data[-4:]
            else:
                if len(structured_data) > 0:
                    question_lines = structured_data[:1]
                    options_raw = structured_data[1:]
                else:
                    question_lines = []
                    options_raw = []

    # Format options
    formatted_options = []
    correct_answer_text = None

    # Mapping index to A, B, C, D
    labels = ["A", "B", "C", "D", "E", "F"]

    # Mark the correct answer in structured_data
    # Use a dynamic threshold based on box size? or just absolute
    if best_match_idx != -1 and max_intersection > 20 and best_match_ratio >= 0.01:
        # Check if the best match is inside options_raw or question_lines
        # We assume correct answer is in options
        # Map best_match_idx (which is index in structured_data) to options_raw

        # Find which item in structured_data corresponds to the answer
        for item in options_raw:
             if item == structured_data[best_match_idx]:
                 item['is_correct'] = True
                 break
        # Note: If the green box is on the question (unlikely but possible error), we ignore it for now.

    for i, item in enumerate(options_raw):
        text = item['text'].strip()
        is_correct = item['is_correct']

        # Check if it already starts with a label
        label = labels[i] if i < len(labels) else "?"

        # Clean existing markers
        match = marker_pattern.match(text)
        if match:
            # Remove the marker
            text = text[match.end():].strip()

        formatted_option = f"{label}. {text}"
        if is_correct:
             formatted_option += " ✅"
             correct_answer_text = formatted_option

        formatted_options.append(formatted_option)


    question_text = "\n".join([item['text'] for item in question_lines])

    # Construct full text output
    # Format:
    # Вопрос:
    # [Text]
    #
    # Ответы:
    # A. [Text]
    # B. [Text] ✅

    output_text = f"Вопрос:\n{question_text}\n\nОтветы:\n"
    for opt in formatted_options:
        output_text += f"{opt}\n"

    return {
        'question': question_text,
        'options': formatted_options,
        'correct_answer': correct_answer_text,
        'full_text': output_text,
        'processed_image': debug_img
    }
