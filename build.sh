docker build -t fit5225a2 .   
docker tag fit5225a2:latest 052057399122.dkr.ecr.us-east-1.amazonaws.com/fit5225a2:latest
docker push 052057399122.dkr.ecr.us-east-1.amazonaws.com/fit5225a2:latest