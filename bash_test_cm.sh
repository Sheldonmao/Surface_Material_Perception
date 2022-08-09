model=FeatureFusionNet # optional from [FeatureFusionNet, ChannelFushionNet]
dataset_mode=patchBlender # optional from [patchBlender, patchCapture]
valid_modals="dot dif" # a list of modals serves as input, choose from [dot, dif, gray] for FeatureFusionNet,
                       # from [dot,dif,geo,gray] for ChannelFusionNet , details are referred to models
data_dir="dir_to_data_root" # !! note: change the data_dir to the direction where the data are stored
out_dir="./Result/"
n_epochs="90"

python test_cm.py --model $model \
        --dataset_mode $dataset_mode \
        --name "${model} ${valid_modals}" \
        --outf "${out_dir}/${model} ${valid_modals}/test_cm/" \
        --load_dir "${out_dir}/${model} ${valid_modals}" \
        --test_csv "${data_dir}/test/data.csv" \
        --data_expand 100 \
        --batch_size 1000 \
        --load_epoch $n_epochs \
        --select_plane "all" \
        --valid_modals $valid_modals \