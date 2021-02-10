# CS329s Machine Learning Model Deployment Tutorial

TODO: warnings
TODO: thank you's (people who have helped create this)

## What is in here?

Code and files to go along with [CS329s machine learning model deployment tutorial](https://stanford-cs329s.github.io/syllabus.html).

## What will I end up with?

If you go through the steps below without fail, you should end up with a Streamlit-powered web application for classifying images of food (deployed on Google Cloud if you want).

## What do I need?

* A [Google Cloud account](https://cloud.google.com/gcp) and a [Google Cloud Project](https://cloud.google.com/resource-manager/docs/creating-managing-projects)
* [Google Cloud SDK installed](https://cloud.google.com/sdk/docs/install) (gcloud CLI utitly)
* Trained [machine learning models](https://github.com/mrdbourke/cs329s-ml-deployment-tutorial/blob/main/model_training.ipynb), our service uses an image classification model on a number of different classes of food from [Food101 dataset](https://www.kaggle.com/dansbecker/food-101).

**Warning:** Using Google Cloud services costs money. If you don't have credits (you get $300USD when you first sign up), you will be charged. Delete and shutdown your work when finished to avoid charges.

## Okay, I'm in, how can I use it?

We're going to tackle this in 3 parts:
1. Getting the app running (running Streamlit on our local machines)
2. Deploying a machine learning model to AI Platform (getting Google Cloud to host one of our models)
3. Deploying our app to App Engine (getting our app on the internet)

### 1. Getting the app running

1. Clone this repo
```
git clone https://github.com/mrdbourke/cs329s-ml-deployment-tutorial
```

2. Change into the `food-vision` directory
```
cd food-vision
```

3. Create and activate a virtual environment (call it what you want, I called mine "env")
```
pip install virtualenv
virtualenv <ENV-NAME>
source <ENV-NAME>/bin/activate
```
4. Install the required dependencies (Streamlit, TensorFlow, etc)
```
pip install -r requirements.txt
```
5. Activate Streamlit and run `app.py`
```
streamlit run app.py
``` 
Running the above command should result in you seeing the following:
![](https://raw.githubusercontent.com/mrdbourke/cs329s-ml-deployment-tutorial/main/images/streamlit-app-what-you-should-see.png)

This is Food Vision 🍔👁 the app we're making.

6. Try an upload an image (e.g. one of the ones in [`food-images/`](https://github.com/mrdbourke/cs329s-ml-deployment-tutorial/tree/main/food-images) such as [`ice_cream.jpeg`](https://github.com/mrdbourke/cs329s-ml-deployment-tutorial/blob/main/food-images/ice_cream.jpeg) and it should load.

7. Notice a "Predict" button appears when you upload an image to the app, click it and see what happens.

8. The app breaks because it tries to contact Google Cloud Platform (GCP) looking for a machine learning model and it either:
 a. won't be able to find the model (wrong API call or the model doesn't exist)
 b. won't be able to use the existing model because the credentials are wrong (seen below)
![credential error](https://raw.githubusercontent.com/mrdbourke/cs329s-ml-deployment-tutorial/main/images/streamlit-app-first-error-youll-run-into.png)
 
This is a good thing! It means our app is trying to contact GCP (using functions in `food-vision/app.py` and `food-vision/utils.py`. 

Now let's learn how to get a model hosted on GCP.

### 2. Getting a machine learning model hosted on GCP
 
> How do I fix this error? (Streamlit can't access your model) 

* Train a model - use the [model training notebook](https://github.com/mrdbourke/cs329s-ml-deployment-tutorial/blob/main/model_training.ipynb)
* Host it on Google Storage
 * Requires creating a bucket - https://cloud.google.com/storage/docs/creating-buckets
 * Then copy your SavedModel to the bucket you created - https://cloud.google.com/storage/docs/uploading-objects#gsutil
* Connect model in Google Storage to AI platform - https://cloud.google.com/ai-platform/prediction/docs/deploying-models 
 * You can use the Google Cloud Console or use `gcloud` CLI - https://cloud.google.com/sdk/gcloud/reference/ai-platform/models/create 
 * create a model
 * create a version (link your model to version saved in GS)
 
* Create a service account to access AI platform (GCP loves permissions, it's for your security)
 * Download the service account key (**Warning:** keep this private! otherwise people can access your GCP account, add these to your .gitignore (.gitignore tells git what files it should ignore when saving/uploading to GitHub))
 * Create a service account which is allowed to access AI platform - https://cloud.google.com/iam/docs/creating-managing-service-accounts
 * Give the service account permissions to use ML Engine Developer (TODO: add image)
 * Get a key for that service account - https://cloud.google.com/iam/docs/creating-managing-service-account-keys
* Update the variables in `app.py` and `utils.py` with your GCP information
 * You'll need:
  * GCP key (JSON)
  * the name of your hosted model (e.g. `efficientnetb0_10_food_classes`)
  * the region of where your hosted model lives (e.g. `us-central1`)
 * Test to see if it works...
  
 > Okay, I've fixed the permissions error, how do I deploy my model/app?
 
 I'm glad you asked...
 
 * run `make gcloud-deploy`... wait 5-10 mins and your app will be on App Engine (as long as you've activated the App Engine API)
 
 > What happens when you run `make gcloud-deploy`?
 
 * TODO: add the steps here of make/GCLOUD deploy
 
> What do all the files in `food-vision` do?

* TODO: list what each file does...

> Where can I learn all of this?

* TODO: Google Cloud free materials etc/lots and lots and lots of blog posts


