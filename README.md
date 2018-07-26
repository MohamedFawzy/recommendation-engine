# Recommendation Engine
Recommender system algorithms and tools , method used and so on collrobative filtering etc .


# Requirements
### Native
- R Language 3.5.x .
- Python 2.7 .
- Recommenderlab library for R language .

### Docker
- Docker installed on your host machine

# Datasets:
- Jester5k e.g of user ratings distribution in R language:
![alt text](https://raw.githubusercontent.com/MohamedFawzy/recommendation-engine/master/imgs/user-ratings.png)

# Algorithms :
- Collaborative Filtering
  - Item Based
    ![alt text](https://raw.githubusercontent.com/MohamedFawzy/recommendation-engine/master/imgs/Rplot.png)
  - User Based


# Install :
- `cd docker`
- `docker compose up`
- You should see something like this
![alt text](https://raw.githubusercontent.com/MohamedFawzy/recommendation-engine/master/imgs/docker.png)

- Confirm everything is working correctly by running the following command
  `docker ps`
- Then you can enter container with the following command `docker exec -it docker_python-microservice_1 /bin/bash`
- Your source code under the following path `/var/www/html`
