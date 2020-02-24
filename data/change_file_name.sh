
for g in {1.0,1.1,1.2,1.3,1.4};
do
	cd /tmp/qTEBD/data/1d_TFI_g${g}/ ;
	for i in mps_chi2*; do git mv $i ${i/mps_chi2/mps_chi2_1st}; done
	for i in mps_chi4*; do git mv $i ${i/mps_chi4/mps_chi4_1st}; done
	cd L10 && for i in mps_chi2*; do git mv $i ${i/mps_chi2/mps_chi2_1st}; done
	cd /tmp/qTEBD/data/1d_TFI_g${g}/ ;
	cd L10 && for i in mps_chi4*; do git mv $i ${i/mps_chi4/mps_chi4_1st}; done
	cd /tmp/qTEBD/data/1d_TFI_g${g}/ ;
	cd L20 && for i in mps_chi2*; do git mv $i ${i/mps_chi2/mps_chi2_1st}; done
	cd /tmp/qTEBD/data/1d_TFI_g${g}/ ;
	cd L20 && for i in mps_chi4*; do git mv $i ${i/mps_chi4/mps_chi4_1st}; done

done;
