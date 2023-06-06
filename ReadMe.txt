READ ME

VARIATIONAL AUTOENCODER

Main_Vae.py

-Run main.py in the terminal (python Main_Vae.py) .
-This is the training part with a batch size of 256, epoch number of 100.
-It will also generate loss curves, FID Values curves and saved model.
-To activate FID calculation, please set Calculate_Fid=True on Main_Vae.py
-Saved model is inside the Model_Save Folder
-And the plots created in the sam directıry with Main_Vae.py

Generator_Vae.py

- Run eval.py in the terminal (python Generator_Vae.py)

- It will load the saved model  and generate 100 image in  the same directory

Model.py

-The AutoEncoder architecture is in this python file. It is called in Main_Vae.py.




GENERA ADVERSARIAL NETWORK

Main_Gan.py

-Run main.py in the terminal (python Main_GAN.py) .
-This is the training part with a batch size of 256, epoch number of 100.
-It will also generate loss curves, FID Values curves and saved model.
-To activate FID calculation, please set Calculate_Fid=True on Main_GAN.py
-Saved model is inside the Model_Save Folder
-And the plots created in the same directıry with Main_GAN.py

Model_Gan.py

-For the WasserStein GAN implementation, please set wasserstein=True in the discriminator function.
-The GAN architecture is gien in this python file and it is called in Main_Gan.py

Generator_Gan.py

- Run Generator_Gan.py in the terminal (python Generator_Gan.py)

- It will load the saved model  and generate 100 image in the same directory



