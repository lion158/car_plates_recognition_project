def detect_plate(IMAGE_PATH):
    img = cv2.imread(IMAGE_PATH)
    image_np = np.array(img).astype(np.uint8)

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes']+label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=5,
                min_score_thresh=.7,
                agnostic_mode=False,
                line_thickness=1)

    plt.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
    plt.show()
    return image_np_with_detections

def cut_plate(image_np_with_detections):
    detection_threshold = 0.7
    try:
    score = detections['detection_scores'][detections['detection_scores'] > detection_threshold][0]
    box = detections['detection_boxes'][0]
    except IndexError:
        print("No car plates detected")
        ### tutaj zwróć coś że nie ma tablicy na zdjęciu i nie można dalej wyszukiwać tekstu
    img = image_np_with_detections
    height = img.shape[0]
    width = img.shape[1]
    # "ROI" -"Region of Interest" 
    # box => [y_min, x_min, y_max, x_max]
    roi = box * [height, width, height, width]


    plate_img = img[roi[0].astype(int):roi[2].astype(int), roi[1].astype(int):roi[3].astype(int)]
    #img[box[0].astype(int):box[1].astype(int), box[2].astype(int):box[3].astype(int)]
    plt.imshow(cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB))

def ocr_results(plate_img):
    
    print(f" PYTESSERACT RESULT: { pytesseract.image_to_string(plate_img)}")
    
    ocr_result = reader.readtext(plate_img)
    print(f"EASY_OCR RESULT: {ocr_result}")
    
    
    print(f"KERAS_OCR RESULTS: {pipeline.recognize([plate_img]}") 