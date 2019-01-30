mkdir logdir
python train.py --gpu=0 --embed_dim=256 --dataset=imdb --n_epochs=100
python train.py --gpu=0 --embed_dim=256 --dataset=imdb --n_epochs=100 --use_tt=True --d=3 --ranks=16
python train.py --gpu=0 --embed_dim=256 --dataset=imdb --n_epochs=100 --use_tt=True --d=4 --ranks=16
python train.py --gpu=0 --embed_dim=256 --dataset=imdb --n_epochs=100 --use_tt=True --d=6 --ranks=16
python train.py --gpu=0 --embed_dim=256 --dataset=sst5 --n_epochs=100
python train.py --gpu=0 --embed_dim=256 --dataset=sst5 --n_epochs=100 --use_tt=True --d=3 --ranks=16
python train.py --gpu=0 --embed_dim=256 --dataset=sst5 --n_epochs=100 --use_tt=True --d=4 --ranks=16
python train.py --gpu=0 --embed_dim=256 --dataset=sst5 --n_epochs=100 --use_tt=True --d=6 --ranks=16
