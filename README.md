# Tensorflow + TF Serving Examples

## Setup

### Docker

1. Build docker image:
```
docker build --pull -t $USER/tensorflow-serving-devel -f Dockerfile.devel . 
```

2. Run the docker container:
```
docker run --name=tf_container -it $USER/tensorflow-serving-devel
```

Copy the models into the container
```
cd <path to project>
docker cp ./export tf_container:/serving
```
4. Start it again with
```
docker start -i tf_container
```
5. Start serving (after you enter docker container)
```
tensorflow_model_server --port=9000 --model_name=MODELNAME --model_base_path=EXPORTPATH &> logs &
```

### gRPC Client

1. Get container IP address
```
docker network inspect bridge | grep IPv4Address
```
2. Test your client
```
cd <path to project>
python MODEL_client.py --server=<IP ADDRESS>:9000 ...
```
