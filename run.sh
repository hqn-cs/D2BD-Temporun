for i in {0..9}
do
    python augment.py --data_path "../mnist/val/" --save_path "mnist/val/$i" --cls $i
done