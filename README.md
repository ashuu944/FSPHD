## Food Security Predictions based on Heterogenous Data (FSPHD) Framework


### 1. Dataset

How to download the dataset

- For **MAC/Linux** users, Open the terminal and navigate to your code directory, e.g. (cd FSPHD). Run the below script to download the data:

```sh
./data.sh
```
- For **Windows** users, Download the data using [Link](https://www.googleapis.com/drive/v3/files/1VJFM0wuljsc2Dhdxus8h0IdcE9-0iJJu?alt=media&key=AIzaSyBo55XtefB47P_CPLKosGvnpEi3pQs5lCk). Unzip the data and paste it in your code (**FSPHD**) folder.


### 2. Configuration (Optional)

The [configuration.py](https://github.com/ashuu944/FSPHD/blob/main/configuration.py) file contains the basic configuration files to setup the directories, variables according to their own dataset.


### 3. Creating Environment

```sh
pip install pandas
pip install -U scikit-learn
pip install GDAL
pip3 install torch torchvision
```

### 4. How to run the application

The application offers multiple running configurations, including:
1. **country:** Options include burkina_faso, rwanda, and tanzania.
2. **algorithm:** Choice between classification or regression.
3. **tt_split:** Data split selection of temporal, spatio-temporal, or percentage methods for dividing input data.

Examples 
For Rwanda
```sh
 python main.py -country=rwanda -algorithm=classification -tt_split=temporal 
```
The output will be found at *'../output/rwanda/results/[model]/temporal'*. 



