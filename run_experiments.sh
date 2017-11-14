# python -m reconstruction -d modelnet10 -o data/test-vgan --epochs 200 --opt voxels-vgan --overwrite -f
# python -m reconstruction -d modelnet10 -o data/test-ugan --epochs 200 --opt voxels-ugan --overwrite -f
# python -m reconstruction -d modelnet10 -o data/test-usegan --epochs 200 --opt voxels-usegan --overwrite -f
python -m reconstruction_final -d arq_dataset -o data/final_model --epochs 400
