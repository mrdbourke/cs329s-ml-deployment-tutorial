# CS329s Machine Learning Model Deployment Tutorial

TODO: warnings
TODO: thank you's (people who have helped create this)

> What is in here?

Code and files to go along with [CS329s machine learning model deployment tutorial](https://stanford-cs329s.github.io/syllabus.html).

> What will I end up with?

If you go through the steps below without fail, you should end up with a Streamlit-powered web application for classifying images of food.

> What do I need?

* GCP account
* gcloud CLI utitly
* gsutil is also helpful 
* trained machine learning models, our service uses an image classification model on a number of different classes of food from Food101 dataset. (see the notebook)

**Warning:** Using Google Cloud services costs money. If you don't have credits (you get $300USD when you first sign up), you will be charged. Delete and shutdown your work when finished to avoid charges.

> Okay, I'm in, how can I use it?

* clone the repo
* cd into food-vision
* make virtualenv `virtualenv <ENV-NAME>`, e.g. `virtualenv env` , if you don't have `virtualenv` -> `pip install virtualenv` 
* activate env: `source <ENV-NAME>/bin/activate`, e.g `source env/bin/activate`
* pip install -r requirements.txt: `pip install -r requirements.txt`

* streamlit run app.py
 * TODO: what you will see... 
* try to upload an image, e.g. ice_cream.jpeg (notice it loads)
* but it will break when it tries to ping the model hosted on GCP (it'll look for GCP credentials, instead of using yours...)
 * TODO: add image of error...
 * this is a good thing! now let's learn how to get a model hosted on GCP
 
* navigate to local host where Streamlit app lives
  * Note: Streamlit app will break unless, you've got the right GCP setup...
 
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


