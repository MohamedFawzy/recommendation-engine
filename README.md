# Recommendation Engine
Have you wondered how amazon recommend items to you ? or netflix recommend content for you , spotify and youtube Here i will summarize things as much as possible .

Recommendation engine a branch of information retrieval and artificial intelligence , are powerful tools and techniques to analyze huge volumes of data , especially product information and user information , and then provide relevant suggestions based on data-mining approaches .

you can find more here : https://medium.com/@mohamedfawzy_44931/recommendation-engine-explained-c5b8642cc0f


# Requirements
### Native
- R Language 3.5.x.
- Python 3.4.
- Recommenderlab library for R language.
- Spark 2.

### Docker
- Docker installed on your host machine

# Datasets:
- Jester5k e.g of user ratings distribution in R language:
![alt text](https://raw.githubusercontent.com/MohamedFawzy/recommendation-engine/master/imgs/user-ratings.png)
- MovieLens 100K users rating
  - Ratings distribution count
  ![alt text](https://raw.githubusercontent.com/MohamedFawzy/recommendation-engine/master/imgs/Figure_1.png)
  - Movies ratings distribution
  ![alt text](https://raw.githubusercontent.com/MohamedFawzy/recommendation-engine/master/imgs/Figure_2.png)
- Anonymous Microsoft Web Dataset

# Algorithms :
- Collaborative Filtering
  - Item Based
    ![alt text](https://raw.githubusercontent.com/MohamedFawzy/recommendation-engine/master/imgs/Rplot.png)
    - Params Tune:
    ![alt text](https://raw.githubusercontent.com/MohamedFawzy/recommendation-engine/master/imgs/Rplot02.png)

  - User Based
    ![alt text](https://raw.githubusercontent.com/MohamedFawzy/recommendation-engine/master/imgs/Rplot01.png)

  - Item-based/User-based for tuning K-NearestNeighbors parameter.
    ![alt text](https://raw.githubusercontent.com/MohamedFawzy/recommendation-engine/master/imgs/Figure_1_python.png)


# Install :
- `cd docker`
- `docker compose up`
- You should see something like this
![alt text](https://raw.githubusercontent.com/MohamedFawzy/recommendation-engine/master/imgs/docker.png)

- Confirm everything is working correctly by running the following command
  `docker ps`
- Then you can enter container with the following command `docker exec -it docker_python-microservice_1 /bin/bash`
- Your source code under the following path `/var/www/html`
