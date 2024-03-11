import json

import requests
from bs4 import BeautifulSoup

################################
## DEFINE USERNAME & PASSWORD ##
################################
MLOPS_PLATFORM_USERNAME = ""
MLOPS_PLATFORM_PASSWORD = ""
EMERALDS_TOKEN_URL = "https://emeralds-token.apps.emeralds.ari-aidata.eu"
KUBEFLOW_ENDPOINT = "https://kubeflow.emeralds.ari-aidata.eu"
KSERVE_MODEL_ENDPOINT = "https://model-example.models.kubeflow.emeralds.ari-aidata.eu/v1/models/model-example:predict"


###################################################
#### REQUEST KEYCLOAK TOKEN (DO NOT CHANGE IT) ####
###################################################
data = {"username": MLOPS_PLATFORM_USERNAME, "password": MLOPS_PLATFORM_PASSWORD}

response = requests.post(EMERALDS_TOKEN_URL, data=data)
if response.status_code != 200:
    raise Exception(f'Error {response.status_code}. {response.json()["detail"]}')
access_token = response.json()['access_token']

#####################################################
# GET AUTHSERVICE_SESSION COOKIE (DO NOT CHANGE IT) #
#####################################################
with requests.Session() as s:

    resp = s.get(KUBEFLOW_ENDPOINT, allow_redirects=True)
    resp = s.get(resp.url)

    login_form = BeautifulSoup(resp.text, features="html.parser").findAll(attrs={'id': 'kc-form-login'})[0]
    post_url = login_form.get('action')

    credentials = {'username': MLOPS_PLATFORM_USERNAME, 'password': MLOPS_PLATFORM_PASSWORD}

    resp = s.post(post_url, data=credentials)
    cookies = {c.name: c.value for c in s.cookies}

#############################################
# INFERENCE (CHANGE THIS BLOCK AS YOU WISH) #
#############################################
data = {"instances": [[6.8, 2.8, 4.8, 1.4], [1, 1, 1, 1]]}

resp = requests.post(
    KSERVE_MODEL_ENDPOINT,
    data=json.dumps(data),
    headers={
        'Authorization': f'Bearer {access_token}',
        'Cookie': f'authservice_session={cookies["authservice_session"]}',
    },
)
print(resp.text)
