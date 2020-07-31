# alexnet_with_tf2.0/2.1/2.2

I am pleased to update the AlexNet and its variants to adapt to TensorFlow 2.0/2.1/2.2. Since the original script of myalexnet_forward written by AlexNet team was build on TensorFlow 1.x in 2017, it has generated many errors during the runtime in the env. of TensorFlow 2.x. Either the AlexNet team has no time to update, or it is very hard for developers to find out the AlexNet variants to adapt to TensorFlow 2.0/2.1 with the Google search. Furthermore, I split the AlexNet model in both the OOP and the pure command style and make client applications to call the AlexNet model because the AlexNet model is not seperated in most of the published scenarios which incur the difficulities for developer's deep learning. Considering that AlexNet is the great milestone in the CNNN/DNN history (Mr. Hinton has won the Turing Award in 2019 and the event is not far from fading away), it is important to continue its usability and accessibility to all developers. It is exciting for me to take the opportunity to share the new update. 

"""
The User Guide: 

Part One. Development Environment

1. For the following development environment, develoeprs can direcly run the script. 

Miniconda 4.8.3
Ubuntu 18.04 LTS
CUDA 10.0.130
cuDNN 7.3.1 
TensorFlow 2.1.0
Keras 2.3.1 


2. For the TensorFlow 2.2.0 environment, please run the scrip and add --cap-add=CAP_SYS_ADMIN

Miniconda 4.8.3
Ubuntu 18.04 LTS
CUDA 11.0
cuDNN 8.0.1
TensorFlow 2.2.0
Keras 2.4.4


Part Two. Folders 

To use the script in Python, users need to create the folder such as Alexnet_Callback. The application 
automatically downloads the pictures into the created folders. 


Part Three. Script running procedure

1. Enter into current directory
   $ cd /home/john/Documents/Alexnet_Callback
   
Anaconda defaults the pre-installed Python3 and the Ubuntu 18.04 has both Python2 and Python3. Therefore, 
users need to follow the procedures. 

2. Script running command

  In the TensorFlow 2.1.0 environment, please execute the following command in the Ubuntu terminal at the current 
  directory.  
  
  $ python alexnet_classifying.py  
  
  or 
  
  In the TensorFlow 2.2.0 environment, please execute the following command. 
  
  $ python3 alexnet_classifying.py --cap-add=CAP_SYS_ADMIN
  
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
 
  
3. Start the TensorBoard
   After completing the script excution, users can start the TensorBoard command in the Linux Terminal 
   at the current directory. 
   
  $ tensorboard --logdir logs/fit
  
  After the above-mentioned command is given, the ｔerminal shows the reminding message as follows. 
  Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all
  TensorBoard 2.2.1 at http://localhost:6006/ (Press CTRL+C to quit)
  
4. Enter the weblink in a browser

   After entering the weblink into either Chrome or Firefox browser, the TensorBoard will show the diagrams
   that the scrip defines. 
   http://localhost:6006/
   
5. Images showing 

   The browser could not show the images. If users want to plot the images, please upload the Python script 
   into the Jupyter Notebook or just directly adopts the original ipython script. 
  
  
Part Four Trouble shooting 

1. The compat issue 

For the error related to the environment of TensorFlow 2.1.0, developers can make the appropriate solutions as follows. 

AttributeError: 
module 'tensorflow' has no attribute 'compat'

Solution: 

It is the conflict between Conda and TensorFlow 2.x if users adopt the Anaconda/Miniconda env. I recommend 
the users to install tensorflow 2.1 and then install tensorflw-estimator as follows. 
$ conda install tensorflow-estimator==2.1.0


2. The CUPTI issue

For the environment of TensorFlow 2.2.0, there is the reminidng error although the scripts runs correctly. 

E tensorflow/core/profiler/internal/gpu/cupti_tracer.cc:1408] function cupti_interface_->Subscribe( &subscriber_, (CUpti_CallbackFunc)ApiCallback, this)failed with error CUPTI_ERROR_INSUFFICIENT_PRIVILEGES
E tensorflow/core/profiler/internal/gpu/cupti_tracer.cc:1447] function cupti_interface_->ActivityRegisterCallbacks( AllocCuptiActivityBuffer, FreeCuptiActivityBuffer)failed with error CUPTI_ERROR_INSUFFICIENT_PRIVILEGES
E tensorflow/core/profiler/internal/gpu/cupti_tracer.cc:1430] function cupti_interface_->EnableCallback( 0 , subscriber_, CUPTI_CB_DOMAIN_DRIVER_API, cbid)failed with error CUPTI_ERROR_INVALID_PARAMETER

According to the current trace report from Nvidia CUDA Profiling Tools Interface(CUPTI), it is only the reminding message. It reminds users of the lacking super user privilege. At the presetn, it is hard to remove the reminding message. Please take the reference of the CUPTI as follows. 

https://docs.nvidia.com/cupti/Cupti/index.html

