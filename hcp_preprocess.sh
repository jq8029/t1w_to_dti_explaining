#!/bin/bash

# Set target dir to the folder of downloaded HCP files
TARGET_DIR="/project/preprocess/data"

for subdir in "$TARGET_DIR"/*/; do
    echo "Processing: $subdir   with $FILE"

	set +e
	
	cd $subdir
	mv $subdir/T1w/T1w_acpc_dc.nii.gz T1_0_brain.nii.gz
	
	dtifit \
		-k T1w/Diffusion/data.nii.gz \
		-o dti \
		-m T1w/Diffusion/nodif_brain_mask.nii.gz \
		-r T1w/Diffusion/bvecs \
		-b T1w/Diffusion/bvals \
		--save_tensor
	
	mv dti_FA.nii.gz T1_1_FA.nii.gz
	mv dti_MD.nii.gz T1_2_MD.nii.gz
	mv dti_L1.nii.gz T1_3_AD.nii.gz
	fslmaths dti_L2.nii.gz -add dti_L3.nii.gz -div 2 T1_4_RD.nii.gz
	
	mv T1w/Diffusion/nodif_brain_mask.nii.gz T1_mask_0_brain.nii.gz
	
	rm -rf dti*.gz
	rm -rf MNINonLinear
	rm -rf T1w
	
	set -e

done

echo "Done!"