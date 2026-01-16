# auto_annotator.py (Optimized Version)
import os
import cv2
from ultralytics import YOLO

def get_classes(model_path):
    """Extract class names from a .pt model, always returning a list[str]"""
    import torch
    print(f"[DEBUG] Loading model to get classes: {model_path}")
    model = torch.load(model_path, map_location='cpu')
    names = None
    if hasattr(model, 'names'):
        names = model.names
        print(f"[DEBUG] Type of model.names: {type(names)}, Preview: {list(names)[:5]}")
    elif 'model' in model and hasattr(model['model'], 'names'):
        names = model['model'].names
        print(f"[DEBUG] Type of model['model'].names: {type(names)}, Preview: {list(names)[:5]}")

    if names is None:
        print("[WARNING] No class information found, using default classes")
        return ["class_0", "class_1"]

    # Key fix: If names is a dict, convert to an ordered list
    if isinstance(names, dict):
        max_key = max(names.keys())
        class_list = [names[i] for i in range(max_key + 1)]
        print(f"[DEBUG] Detected dict for classes, converted to ordered list. Length: {len(class_list)}")
        return class_list
    elif isinstance(names, (list, tuple)):
        return list(names)
    else:
        # Handle other cases (e.g., OrderedDict)
        try:
            return list(names)
        except Exception as e:
            print(f"[ERROR] Failed to convert classes to list: {e}, using defaults")
            return ["class_0", "class_1"]

def run_auto_annotation(model_path, image_dir, label_dir, conf_threshold=0.25, selected_classes=None):
    print(f"[TRACE] auto_annotator.py → Received selected_classes = {selected_classes}")
    print(f"[TRACE] Type: {type(selected_classes)}")

    if not os.path.exists(label_dir):
        os.makedirs(label_dir)

    img_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(img_exts)]
    all_class_names = get_classes(model_path)
    print(f"[TRACE] All model classes (Total: {len(all_class_names)}): {all_class_names[:5]}...")

    model = YOLO(model_path)

    # Use the passed selected_classes directly (it can be an empty list [])
    if selected_classes is None:
        selected_classes = all_class_names # Only use all if explicitly None

    selected_set = set(selected_classes)
    filtered_class_names = [name for name in all_class_names if name in selected_set]
    print(f"[DEBUG] Final classes to annotate: {filtered_class_names}")

    # Key fix: Do not fall back to all_class_names!
    if not filtered_class_names:
        print("[WARNING] No valid classes selected, will generate empty labels")
        filtered_class_names = []

    # Build mapping from old_id to new_id
    old_id_to_new_id = {}
    for new_id, cls_name in enumerate(filtered_class_names):
        old_id = all_class_names.index(cls_name)
        old_id_to_new_id[old_id] = new_id
        print(f"[TRACE] Mapping: Old ID={old_id}({cls_name}) → New ID={new_id}")

    # Write classes.txt
    classes_file = os.path.join(label_dir, 'classes.txt')
    with open(classes_file, 'w', encoding='utf-8') as f:
        for cls in filtered_class_names:
            f.write(f"{cls}\n")
    print(f"[TRACE] Written to classes.txt: {classes_file}")

    total_images = len(image_files)
    processed = 0
    for img_name in image_files:
        img_path = os.path.join(image_dir, img_name)
        image = cv2.imread(img_path)
        if image is None:
            continue

        results = model(image, conf=conf_threshold)
        label_path = os.path.splitext(img_name)[0] + '.txt'
        full_label_path = os.path.join(label_dir, label_path)

        with open(full_label_path, 'w') as f:
            for box in results[0].boxes:
                old_cls_id = int(box.cls.item())
                if old_cls_id not in old_id_to_new_id:
                    cls_name = all_class_names[old_cls_id] if old_cls_id < len(all_class_names) else "unknown"
                    print(f"[DEBUG] Skipping unselected class: ID={old_cls_id}, Name='{cls_name}'")
                    continue
                new_cls_id = old_id_to_new_id[old_cls_id]
                x_center, y_center, box_w, box_h = box.xywhn[0].tolist()
                f.write(f"{new_cls_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}\n")

        processed += 1
        yield processed, total_images

def preview_detection(model_path, image_path, conf_threshold=0.25, selected_classes=None):
    image = cv2.imread(image_path)
    if image is None:
        return None

    model = YOLO(model_path)
    all_class_names = get_classes(model_path) # Reuse

    if selected_classes:
        selected_set = set(selected_classes)
        keep_ids = [i for i, name in enumerate(all_class_names) if name in selected_set]
        results = model(image, conf=conf_threshold, classes=keep_ids)
    else:
        results = model(image, conf=conf_threshold)

    return results[0].plot()