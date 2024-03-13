from ultralytics import YOLO
import cv2 as cv
import os
import argparse

parser = argparse.ArgumentParser(
                    prog='gen_new_labels',
                    description='Generate new labels')
parser.add_argument('-i', '--input', type=str, required=True, help='Input video file')
parser.add_argument('-m', '--model', type=str, required=True, help='Input model')
parser.add_argument('-o', '--outdir', type=str, required=True, help='Output directory')
parser.add_argument('-c', '--count', type=int, required=True, help='Starting image number')
args = parser.parse_args()

model_file = args.model
output_dir = args.outdir
input_file = args.input
count = 669


if not os.path.exists(output_dir):
    os.mkdir(output_dir)
    os.mkdir(f'{output_dir}/images')
    os.mkdir(f'{output_dir}/labels')
    with open(f'{output_dir}/labels/classes.txt', 'w') as f:
        f.write('note')

# Load a model
model = YOLO(model_file, verbose=False)  # load an official model

cap = cv.VideoCapture(input_file)

while True:
    count += 1
    ret, img = cap.read()
    if not ret:
        exit(0)

    # Predict with the model
    results = model(img)  # predict on an image
    for r in results:
        outimg = f'{output_dir}/images/img{count:05}.jpg'
        outlabel = f'{output_dir}/labels/img{count:05}.txt'
        cv.imwrite(outimg, img)
        with open(outlabel, 'w') as writer:
            box = r.boxes
            for cls, xywhn in zip(box.cls, box.xywhn):
                writer.write(f'{int(cls)} {xywhn[0]} {xywhn[1]} {xywhn[2]} {xywhn[3]}')