# AlexNet with TF 2.x

I am pleased to update the AlexNet and its variants from small dataset to the 160GB+ ImageNet dataset to adapt to the need for really big data. Since TensorFlow are fast iterative, the old scripts written by the AlexNet team(in Toronto University) are outdated, I updates TensorFlow from v2.4 to 2.10.0. Furthermore, I divide the AlexNet model in both the OOP and the command style and make client applications to call the AlexNet model because the AlexNet model is not seperated in most of the published scenarios which incur the difficulities for developer's deep learning. Considering AlexNet is the great milestone in the CNN/DNN history. Mr. Hinton has won the Turing Award in 2019 and the Deep Learning feast is not fading away. Therefore, it is important to continue its usability and accessibility to all developers. It is exciting for me to take the opportunity to share the new update. 

# Development Environment

## For the following development environment, develoeprs can direcly run the script. 

  Miniconda v22.9.0
  
  Ubuntu 18.04 LTS
  
  CUDA 11.0 or the above
  
  cuDNN 8.0.1 
  
  TensorFlow 2.10.0
  
  Keras 2.10.0

# Folders 

To use the script in Python, users need to create the folder such as Alexnet_Callback. The application 
automatically downloads the pictures into the created folders. 

# Script running procedure

## Enter into current directory

   $ cd /home/user/Documents/Alexnet_Callback
   
Anaconda defaults the pre-installed Python3 and the Ubuntu 18.04 has both Python2 and Python3. Therefore, 
users need to follow the procedures. 

## Script running command

  In the TensorFlow 2.1.0 environment, please execute the following command in the Ubuntu terminal at the current 
  directory.  
  
  $ python alexnet_classify.py  
  
  or 
  
  In the TensorFlow 2.2.0 environment, please execute the following command. 
  
  $ python3 alexnet_classify.py --cap-add=CAP_SYS_ADMIN
  
  While executing the above-mentioned command, the Linux Terminal shows the arrays of image name ended 
  with jpg. 
  
  Moreover, the Terminal show the complete Model: alex_net". Furthermore, it show "Found 117 images 
  belonging to 2 classes". 
  
  In the meantime, it also address the following warning. However, users can ignore the warning becuase it
  does not influence the script running. 
  
  WARNING:tensorflow:sample_weight modes were coerced from
  ...
    to  
  ['...']
 
  
## Start the TensorBoard

   After completing the script excution, users can start the TensorBoard command in the Linux Terminal 
   at the current directory. 
   
  $ tensorboard --logdir logs/fit
  
  After the above-mentioned command is given, the ｔerminal shows the reminding message as follows. 
  Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all
  TensorBoard 2.2.1 at http://localhost:6006/ (Press CTRL+C to quit)
  
## Enter the weblink in a browser

   http://localhost:6006/

   After entering the above weblink into either Chrome or Firefox browser, the TensorBoard will show the 
   diagrams that the scrip defines. 
   
## Images showing 

   The browser could not show the images. If users want to plot the images, please upload the Python script 
   into the Jupyter Notebook or just directly adopts the original ipython script. 
