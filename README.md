# CS329s Machine Learning Model Deployment Tutorial

**Warning:** Following the steps of what's in here may cost you money (Google Cloud is a paid service), be sure to follow accordingly.

**Thank you to:** [Mark Douthwaite's incredible ML + software engineering blog](https://mark.douthwaite.io/), [Lj Miranda Huyen's amazing post on software engineering tools for data scientists](https://ljvmiranda921.github.io/notebook/2020/11/15/data-science-swe/), [Chip Huyen](https://huyenchip.com/) and Ashik Shafi's gracious feedback on the raw materials of this tutorial.

## What is in here?

Code and files to go along with [CS329s machine learning model deployment tutorial](https://stanford-cs329s.github.io/syllabus.html).

## What will I end up with?

If you go through the steps below without fail, you should end up with a Streamlit-powered web application for classifying images of food (deployed on Google Cloud if you want).

## What do I need?

* A [Google Cloud account](https://cloud.google.com/gcp) and a [Google Cloud Project](https://cloud.google.com/resource-manager/docs/creating-managing-projects)
* [Google Cloud SDK installed](https://cloud.google.com/sdk/docs/install) (gcloud CLI utitly)
* Trained [machine learning models](https://github.com/mrdbourke/cs329s-ml-deployment-tutorial/blob/main/model_training.ipynb), our service uses an image classification model on a number of different classes of food from [Food101 dataset](https://www.kaggle.com/dansbecker/food-101).

**Warning (again):** Using Google Cloud services costs money. If you don't have credits (you get $300USD when you first sign up), you will be charged. Delete and shutdown your work when finished to avoid charges.

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
 * won't be able to find the model (wrong API call or the model doesn't exist)
 * won't be able to use the existing model because the credentials are wrong (seen below)
![credential error](https://raw.githubusercontent.com/mrdbourke/cs329s-ml-deployment-tutorial/main/images/streamlit-app-first-error-youll-run-into.png)
 
This is a good thing! It means our app is trying to contact GCP (using functions in `food-vision/app.py` and `food-vision/utils.py`). 

Now let's learn how to get a model hosted on GCP.

### 2. Getting a machine learning model hosted on GCP
 
> How do I fix this error? (Streamlit can't access your model) 

To fix it, we're going to need a couple of things:
* A trained machine learning model (suited to our problem, we'll be uploading this to Google Storage)
* A Google Storage bucket (to store our trained model)
* A hosted model on Google AI Platform (we'll connect the model in our Google Storage bucket to here)
* A service key to access our hosted model on Google AI Platform

Let's see how we'll can get the above.

1. To train a machine learning model and save it in the [`SavedModel`](https://www.tensorflow.org/guide/saved_model) format (this TensorFlow specific, do what you need for PyTorch), we can follow the steps in [`model_training.ipynb`](https://github.com/mrdbourke/cs329s-ml-deployment-tutorial/blob/main/model_training.ipynb).

2. Once we've got a `SavedModel`, we'll upload it Google Storage but before we do that, we'll need to [create a Google Storage Bucket](https://cloud.google.com/storage/docs/creating-buckets).

TODO: Image

3. With a bucket created, we can [copy our model to the bucket](https://cloud.google.com/storage/docs/uploading-objects#gsutil).
```
## Uploading a model to Google Storage from within Colab ##

# Authorize Colab and initalize gcloud (enter the appropriate inputs when asked)
from google.colab import auth
auth.authenticate_user()
!curl https://sdk.cloud.google.com | bash
!gcloud init

# Upload SavedModel to Google Storage Bucket
!gsutil cp -r <YOUR_MODEL_PATH> <YOUR_GOOGLE_STORAGE_BUCKET>
```

4. [Connect model in bucket to AI Platform](https://cloud.google.com/ai-platform/prediction/docs/deploying-models) (this'll make our model accessible via an API call, if you're not sure what an API call is, imagine writing a function that could trigger our model from anywhere on the internet)
 * Don't like clicking around Google Cloud's console? You can also [use `gcloud` to create a model in AI Platform](https://cloud.google.com/sdk/gcloud/reference/ai-platform/models/create) on the command line 
 
 TODO: image(s)
 
5. Create a [service account to access AI Platform](https://cloud.google.com/iam/docs/creating-managing-service-accounts) (GCP loves permissions, it's for the security of your app)
 * You'll want to make a service account with permissions to use the "ML Engine Developer" role

![ml developer role permission](https://raw.githubusercontent.com/mrdbourke/cs329s-ml-deployment-tutorial/main/images/gcp-ml-engine-permissions.png)

6. Once you've got an active service account, [create and download its key](https://cloud.google.com/iam/docs/creating-managing-service-account-keys) (this will come in the form of a .JSON file)
 * 🔑 **Note:** Service keys grant access to your GCP account, keep this file private (e.g add `*.json` to your `.gitignore` so you don't accidently add it to GitHub)

7. Update the following variables:
 * In `app.py`, change the existing GCP key path to your key path:
```
# Google Cloud Services look for these when your app runs

# Old
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "daniels-dl-playground-4edbcb2e6e37.json"

# New 
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "<PATH_TO_YOUR_KEY>"
```
 * In `app.py`, change the GCP project and region to your GCP project and region
```
# Old
PROJECT = "daniels-dl-playground"
REGION = "us-central1" 

# New
PROJECT = "<YOUR_GCP_PROJECT_NAME>"
REGION = "<YOUR_GCP_REGION>"
```
 * In `utils.py`, change the `"model_name"` key of `"model_1"` to your model name:
 ```
 # Old
 classes_and_models = {
    "model_1": {
        "classes": base_classes,
        "model_name": "efficientnet_model_1_10_classes" 
    }
 }
 
 # New
  classes_and_models = {
    "model_1": {
        "classes": base_classes,
        "model_name": "<YOUR_AI_PLATFORM_MODEL_NAME>" 
    }
 }
```

8. Retry the app to see if it works (refresh the Streamlit app by pressing R or refreshing the page and then reupload an image and click "Predict")

TODO: Image
  
### 3. Deploying the whole app to GCP

> Okay, I've fixed the permissions error, how do I deploy my model/app?
 
I'm glad you asked...
 
1. run `make gcloud-deploy`... wait 5-10 mins and your app will be on App Engine (as long as you've activated the App Engine API)

...and you're done
 
 > But wait, what happens when you run `make gcloud-deploy`?
 
 * TODO: add the steps here of make/GCLOUD deploy
 
## Breaking down `food-vision`

> What do all the files in `food-vision` do?

* TODO: list what each file does...

## Where else your app will break

The app we've deployed is far from perfect, here's where you'll find it also breaking: 
* TODO

## Learn more

> Where can I learn all of this?

* TODO: Google Cloud free materials etc/lots and lots and lots of blog posts

* https://google.qwiklabs.com/
* https://ljvmiranda921.github.io/notebook/2020/11/15/data-science-swe/
* https://mark.douthwaite.io/deploying-streamlit-to-app-engine/

## TODO: Extensions

* CI/CD
* Codify everything!
