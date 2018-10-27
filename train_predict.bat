cd /d %~dp0
pause(0)
python ./src/align_dataset_mtcnn.py ./dataset_without_align ./dataset_without_align_160 --image_size 160 --margin 0 --random_order
pause(0)
python ./src/classifier.py TRAIN ./dataset_without_align_160 E:\AI_DATA\facenet_model%please replace with your own directory containing the meta_file and ckpt_file% ./classifier.pkl
pause(0)
python ./src/camera_predict.py E:\AI_DATA\facenet_model%replace with your own% ./classifier.pkl