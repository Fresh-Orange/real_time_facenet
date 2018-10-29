import os

model_dir = "E:\\AI_DATA\\facenet_model"  # replace with your own directory containing the meta_file and ckpt_file

dataset = "./dataset_without_align"
dataset_align = "./dataset_with_align_160"
classifier = "./classifier.pkl"

export_cmd = "export PYTHONPATH=./src"
align_cmd = "python ./src/align_dataset_mtcnn.py {} {}  --image_size 160 --margin 0 --random_order".format(dataset, dataset_align)
train_classifier_cmd = "python ./src/classifier.py TRAIN {} {} {}".format(dataset_align, model_dir, classifier)
predict_cmd = "python ./src/camera_predict.py {} {}".format(model_dir, classifier)

os.system(export_cmd)

print("\n\naligning and cropping images, please wait...\n\n")
print(os.system(align_cmd))

print("\n\ntraining classifier, please wait...\n\n")
os.system(train_classifier_cmd)

print("\n\nloading model and opening your camera, please wait...\n\n")
os.system(predict_cmd)
