python mask_gen_dynamic.py --dataset giraffe --data ../datasets/masks --path ../datasets/rgbs/giraffe --dataloader SynthLoader --getdata get_data_synth --nb_l 6 --angle 180 --LAMBDA_gm_rot 50. --LAMBDA_gm 100. --LAMBDA_m 100. --LAMBDA_cyc 10. --nb_epochs 200 --ratio .9 --batch 3