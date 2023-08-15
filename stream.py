import onnxruntime
import cv2
import numpy as np
import time
import streamlit as st
import tempfile
import subprocess

def box_iou_batch(
	boxes_a: np.ndarray, boxes_b: np.ndarray
) -> np.ndarray:

    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    area_a = box_area(boxes_a.T)
    area_b = box_area(boxes_b.T)

    top_left = np.maximum(boxes_a[:, None, :2], boxes_b[:, :2])
    bottom_right = np.minimum(boxes_a[:, None, 2:], boxes_b[:, 2:])

    area_inter = np.prod(
    	np.clip(bottom_right - top_left, a_min=0, a_max=None), 2)
        
    return area_inter / (area_a[:, None] + area_b - area_inter)

def non_max_suppression(
   predictions: np.ndarray, iou_threshold: float = 0.5
) -> np.ndarray:
    rows, columns = predictions.shape

    sort_index = np.flip(predictions[:, 4].argsort())
    predictions = predictions[sort_index]

    boxes = predictions[:, :4]
    categories = predictions[:, 5]
    ious = box_iou_batch(boxes, boxes)
    ious = ious - np.eye(rows)
    # print(ious)

    keep = np.ones(rows, dtype=bool)

    for index, (iou, category) in enumerate(zip(ious, categories)):
        if not keep[index]:
            continue

        condition = (iou > iou_threshold) & (categories == category)
        keep = keep & ~condition

    return keep[sort_index.argsort()]

def xywh2xyxy(x):
    # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

st.title('Deploy Yolov8 model to GCP')
st.header(":green[Upload your video and enjoy!]")

video_data = st.file_uploader("upload", ['mp4','mov', 'avi'])

opt_session = onnxruntime.SessionOptions()
opt_session.enable_mem_pattern = False
opt_session.enable_cpu_mem_arena = False
opt_session.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL

model_path = 'model.onnx'
EP_list = ['CPUExecutionProvider']

ort_session = onnxruntime.InferenceSession(model_path, providers=EP_list)

model_inputs = ort_session.get_inputs()
input_names = [model_inputs[i].name for i in range(len(model_inputs))]
input_shape = model_inputs[0].shape

model_output = ort_session.get_outputs()
output_names = [model_output[i].name for i in range(len(model_output))]

write = st.checkbox('Write to video', value=True)
real = st.checkbox('Show real time')

if st.button("Let's started") and (write or real):

    if video_data:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_filename = temp_file.name
            temp_file.write(video_data.read())
        
        my_bar = st.progress(0, text="Detection in progress. Please wait...")

        # read it with cv2.VideoCapture(),
        # so now we can process it with OpenCV functions
        cap = cv2.VideoCapture(temp_filename)

        nums = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if write:
            frame_width = int(cap.get(3))
            frame_height = int(cap.get(4))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = cap.get(cv2.CAP_PROP_FPS)
            out = cv2.VideoWriter('result.mp4', fourcc, fps, (frame_width, frame_height))
        if real:
            end = cv2.imread('images/end.png')
            imagepl = st.empty()

        if cap.isOpened()== False:
            st.write("Error opening video stream or file. Upload another video or another video extension(Support [mp4, mov, avi]).")

        num = 0
        cap = cv2.VideoCapture(temp_filename)

        if (cap.isOpened()== False):
            print("Error opening video stream or file")

        with open('coco.names') as f:
            classes = f.read().split('\n')

        while(cap.isOpened()):
            ret, frame = cap.read()
            my_bar.progress(num / nums, text="Detection in progress. Please wait...")
            num += 1
            if ret == True:
                now = time.time()
                image_height, image_width = frame.shape[:2]
        #         Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                input_height, input_width = input_shape[2:]
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                resized = cv2.resize(image_rgb, (input_width, input_height))

                # Scale input pixel value to 0 to 1
                input_image = resized / 255.0
                input_image = input_image.transpose(2,0,1)
                input_tensor = input_image[np.newaxis, :, :, :].astype(np.float32)

                outputs = ort_session.run(output_names, {input_names[0]: input_tensor})[0]
                predictions = np.squeeze(outputs).T
                conf_thresold = 0.6
                # Filter out object confidence scores below threshold
                scores = np.max(predictions[:, 4:], axis=1)
                predictions = predictions[scores > conf_thresold, :]
                scores = scores[scores > conf_thresold]
                class_ids = np.argmax(predictions[:, 4:], axis=1)

                # Get bounding boxes for each object
                boxes = predictions[:, :4]

                #rescale box
                input_shape = np.array([input_width, input_height, input_width, input_height])
                boxes = np.divide(boxes, input_shape, dtype=np.float32)
                boxes *= np.array([image_width, image_height, image_width, image_height])
                boxes = boxes.astype(np.int32)
                boxes = xywh2xyxy(boxes)
                # print(boxes)
                
                indices = non_max_suppression(np.concatenate((boxes, np.expand_dims(scores, axis=1), np.expand_dims(class_ids, axis=1)), axis=1), 0.3)
                for (bbox, score, label) in zip(boxes[indices], scores[indices], class_ids[indices]):
                    bbox = bbox.round().astype(np.int32).tolist()
                    cls = classes[int(label)]
                    color = (0,255,0)
                    cv2.rectangle(frame, tuple(bbox[:2]), tuple(bbox[2:]), color, 2)
                    cv2.putText(frame,
                                f'{cls}:{int(score*100)}', (bbox[0], bbox[1] - 2),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.60, [225, 255, 255],
                                thickness=1)
                fps = 1 / (time.time() - now)
                cv2.putText(frame,
                            f'FPS:{round(fps, 2)}', (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, [0,0,255],
                            thickness=1)
                if write:
                    out.write(frame)
                if real:
                    imagepl.image(frame)
            else:
                if real:
                    imagepl.image(end)
                break
        cap.release()
        # Destroys all the windows created
        cv2.destroyAllWindows()
        if write:
            out.release()
            subprocess.call(args=f"ffmpeg -y -i result.mp4 -c:v libx264 convert.mp4", shell=True)

            video_file = open('convert.mp4', 'rb')
            video_bytes = video_file.read()

            st.video(video_bytes)

            # Download button
            with open("result.mp4", "rb") as file:
                btn = st.download_button(
                        label="Download video",
                        data=file,
                        file_name="result.mp4",
                        mime="video/mp4"
                    )
    else:
        if write:
            # Download button
            with open("result.mp4", "rb") as file:
                btn = st.download_button(
                        label="Download video",
                        data=file,
                        file_name="result.mp4",
                        mime="video/mp4"
                    )
        if real:
            end = cv2.imread('images/end.png')
            st.image(end)
    

# ╔════════════════════════════════════════════════════════════╗
# ╠═ Uploading 1 file to Google Cloud Storage                 ═╣
# ╚════════════════════════════════════════════════════════════╝