Mini project on hate speech detection for EPFL EE-559 Deep Learning course (spring 2025)

Based on TIB-VA by Hakimov, Cheema and Ewerth (https://github.com/TIBHannover/multimodal-misogyny-detection-mami-2022) 
and "A closer look at the explainability of Contrastive language-image pre-training" by Yi Li, Hualiang Wang, Yiqun Duan, Jiheng Zhang and Xiaomeng Li (https://github.com/xmed-lab/CLIP_Surgery)

The MAMI dataset files are under "data". Images need to be downloaded and put under the parent folder "data" as "training_images" and "test_images". Download link : https://drive.google.com/file/d/169qe9n4EbNlVbzFWNMjVX3N74Hh5Jcqr/view


Make sure the folder outputs exists
Clone OpenAI's official CLIP github repo https://github.com/openai/CLIP.git
Clone CLIP_Surgery's github repo (https://github.com/xmed-lab/CLIP_Surgery) and replace the following line
in the file CLIP_Surgery/clip/clip.py (change in pip module version that wasn't updated in clip surgery):
line 6 : from pkg_resources import packaging  -> replace with -> import packaging

To retrain the classifier, change status variable to "train". To test the accuracy of the classifer, change it to "test". To classify an image and produce explainability outputs, change it to "explain".

In explain mode, to choose which image to classify & explain, use the idx variable. Choose an image from the data/test_images/ folder, find its corresponding index in data/test.tsv and change the idx variable to this number.