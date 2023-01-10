# corner-shop-ml
Repo for dev on Corner shop ML (propensity to pay) pipeline

## Pipeline
<img width="509" alt="image" src="https://user-images.githubusercontent.com/50050912/204278202-de4331ef-fdb6-44a6-932e-16c7cd98ab12.png">

## Set-up
```
cd ../corner-shop-ml
conda env create -f environment.yml
conda activate corner_shop_ml_env
```

## Run
```
cd ../corner-shop-ml
python main.py

# List of arguments
python main.py --help

# To train the model using data from BigQuery 
python main.py --bigqueryData True
```
