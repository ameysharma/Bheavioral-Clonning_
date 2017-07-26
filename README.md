# Behaviorial Cloning Project
### Data Collection
* I used udacity generated data set to train my model.
* Images were collected from all three cameras.
* Angle Adjustments of 0.22 were made to left(+.22) and Right Images(-.22)
* Followed By fliping every image vertically and adjusting angles accordingly.
* All these data collection was done using generators to imporve memory utilization

### Data Learing Model
* In this model I first adjusted the image datas to range of -1 to 1 using lamba function in keras.
* Then I used 2 2D-Convolution neural networks of kernal 5X5 and filters 3 and 18 with Maxpooling function of 2 stides.
* This was then followed by 3X3 Kernals with filiter 36 and 64 along with Maxpooling function
* After performing this I used Dropout function to remove noise in the function
* This was followed by flatten function and further followed by Dense function of 100,80 and 1
* After Creating the architecture I used generator fit fuction for execution.
*Note:- Learning Rate= 0.0008 & Epoach=20 , Losses =0.0208, Validation Losses=0.0221

#### Note: Model Suggestion taken from Nvdia Model. Moreover, I would like to thank  my mentor and forums mentors in helping me complete this project.