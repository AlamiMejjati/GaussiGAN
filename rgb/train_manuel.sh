python rgb_gen_dynamic.py --dataset manuel --data ../datasets/masks --path ../datasets/rgbs/manuel --dataloader SynthLoader --getdata get_data_synth --nb_l 12 --angle 180 --LAMBDA_m 100. --LAMBDA_p 0.5 --nb_epochs 200 --gpu 0 --batch 4 --LAMBDA_KLt 0.01 --LAMBDA_z 0.1 --load /home/ymejjati/Documents/PhDYoop/gaussigan/pretrained/giraffe/mask/Ep200_r0.9_L10_Lm100_Lgm100_Lgmrot50_Lcyc10_ange180_nbl6/checkpoint,./vgg/vgg16.npz