### Model Utility

Different training epochs on the unlearning model will have different model utility.

We use cumulative reward for Retain Dataset on the unlearned model as our model utility caculation original data. The data is a list which length is 50*19 where 50 refers to running 50 epoch for test process on unlearned model, and 19 refers to we have 19 retained environments (maps). We first applied min-max normalisation on the data. After that we reshape the data to (19,50) then we accumulate 50 epoch reward for each map. Once we got the result, we swap the dimension to make the data have a shape like (50,19). Then we pick the last element (index = 49, the last epoch) in the processed data and calculate the average reward for 19 environments, and the result is our model utility.



### Forget Quality

We will record $R_{truth}$ for each epoch independently (which means no accumulation from epoch to epoch, but we will do later) during normal model test, unlearn model test, retain model test.

We will use two sets of data for the calculation of forget quality. One is the unlearned model test performance on the forget dataset (env = 0, the first map) and the other is the retained model test performance on forget dataset. Firstly, we calculate the sum for each set of data and then for each element in each set of data, we will use that element to divide the sum to get a nomalized value. Once we completed the steps above, we will add each element in the normalized data with all previous elements sum to follow Cumulative Distribution (CDF) requirement then perform a two sample KS-test on these two set of data to calculate a p-value. After that we will log the p-value to get our forget quality.



### Model Utility vs Forget Quality Diagram

For decremental methodology, we select epoch 6 12 18 24 30

### Code Explaination

* decremental.py

  * Add Retain Process Code

  * modify the DQNAgent().get_action(): Now it can return both action and $R_{truth}$

  * Modify the code by accepted command line argument as unlearn model train epoch

  	* for example: python3 decremental.py 10 means run 10 epoch for the unlearn model training

  * Add tqdm for helping us track the training process

  * Modify the env.reset(), now only unlearn model training process will random generator 20 maps all rest training and testing will load our pre-generated map.

  * Add dump feature, now all $R_{truth}$ and reward will be recorded and store as a pkl file.

  	* Note: The saved pkl file follow the following format: `s{seed}-{mode}-RTruth-Unlearn-epoch-{unlearn_epoch}.pkl`

  		* seed: we **<u>are not using</u>** the seed now

  			* s3 (seed 3) used to be the best seed for decremental model


  		
  		  

* ks-test.py

  will load all needed pkl file and calculate model utility(x) and forget quality(y), once we got the result, the result will be saved as `{mode}-result.csv`

* Merge_csv.py

  Generate ` combined_result.csv` by from `dec-result.csv`

* drawimg.py

  Draw the image and saved image in  `Grid_World.pdf` by using the data in `combined_result.csv`

* Drawing_partial.py

	Draw the image for the data on the right top corner

* maps.json

	Saved map data

* Maps-visual.txt

	Visualised saved map data

* trained_data

	All training and testing record data are stored in here

* trained_result

	All ks-test result are stored in here





