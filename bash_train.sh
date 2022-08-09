model=FeatureFusionNet # optional from [FeatureFusionNet, ChannelFushionNet]
dataset_mode=patchBlender # optional from [patchBlender, patchCapture]
valid_modals="dot dif" # a list of modals serves as input, choose from [dot, dif, gray] for FeatureFusionNet,
                       # from [dot,dif,geo,gray] for ChannelFusionNet , details are referred to models
data_dir="dir_to_data_root" # !! note: change the data_dir to the direction where the data are stored
out_dir="./Result/"
n_epochs='90'
n_epochs_decay="60"

python train.py --model $model \
                --dataset_mode $dataset_mode \
                --name "${model} ${valid_modals}" \
                --outf "${out_dir}/" \
                --train_csv "${data_dir}/train/data.csv" \
                --val_csv "${data_dir}/test/data.csv" \
                --valid_modals $valid_modals \
                --n_epochs $n_epochs \
                --n_epochs_decay $n_epochs_decay \
                --data_expand 100 \
                # --select_plane "all" \

python test_cm.py --model $model \
        --dataset_mode $dataset_mode \
        --name "${model} ${valid_modals}" \
        --outf "${out_dir}/${model} ${valid_modals}/test_cm/" \
        --load_dir "${out_dir}/${model} ${valid_modals}" \
        --test_csv "${data_dir}/test/data.csv" \
        --data_expand 100 \
        --batch_size 1000 \
        --load_epoch $n_epochs \
        --valid_modals $valid_modals \
        # --select_plane "all" \