# Prerequisites in our kubernetes cluster

Before launching an application in your Kubernetes cluster, follow these steps:

# Permissions to pull our docker image

1. Create a namespace specifically for your application.
2. If using GitHub, generate a personal token with the ``read:packages` scope only.
3. Run the following command in your cluster:
``````
kubectl create secret docker-registry github-container-registry --namespace=<namespace> --docker-server=ghcr.io --docker-username=<github-username> --docker-password=<token>
``````
4. In your `inferenceservice.yaml` file, make sure to reference this configuration.
``````
spec:
  predictor:
    containers:
      - name: container-name
        image: ghcr.io/username/package:latest
        imagePullPolicy: Always
    imagePullSecrets:
        - name: github-container-registry
``````

# Authenticate with the cluster

The next step is to allow GitHub Actions to manage resources in our Kubernetes cluster.

1. Find the kubeconfig file and add it to GitHub as a secret. Name this secret 'KUBECONFIG'. It will be used in GitHub Actions.

2. Change the default domain in your kubeflow (This has to be done just once).
``````
kubectl edit configmap config-domain -n knative-serving
``````

Delete the entire "data" section and paste this:
``````
data:
  kubeflow.domain.com : ""
  # Example: mlops.ari-aidata.eu: ""
``````
